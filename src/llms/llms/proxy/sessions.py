"""Per-conversation upstream session simulation.

Genuine opencode mints one ses_ id per conversation (title request first,
agent turns after, prompt_cache_key mirrored). The proxy mirrors that so
each client conversation looks like a normal session:

- responses ingress: ``previous_response_id`` chains continuations; Codex
  sends stable ``thread-id``/``session-id`` headers instead (verified on
  the wire — it never sends previous_response_id).
- chat/messages ingress: no conversation signal exists, so those share
  the stable per-key session (previous behavior).

New conversations on the responses leg get a title warming call in the
background (same shape as a genuine session open; ``stream:true`` because
the gate requires it). Tombstones expire after ``SESSION_TTL_S``; the map
is best-effort and memory-only.
"""

from __future__ import annotations

import logging
import os
import time

import httpx

logger = logging.getLogger("zen_proxy")

SESSION_TTL_S = float(os.getenv("ZEN_SESSION_TTL_S", "1800"))
SESSION_MAX_ENTRIES = int(os.getenv("ZEN_SESSION_MAX_ENTRIES", "10000"))
SESSION_WARMING = os.getenv("ZEN_SESSION_WARMING", "1") == "1"
WARM_MAX_TOKENS = int(os.getenv("ZEN_SESSION_WARM_TOKENS", "32"))


def mint_session_id() -> str:
    """Fresh genuine-shape ses_ id (12 hex + 14 base62, random)."""
    raw = os.urandom(20)
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return "ses_" + raw[:6].hex() + "".join(alphabet[b % 62] for b in raw[6:])


def conversation_ref(ingress: str, body: dict, headers) -> str | None:
    """Opaque conversation key for this request, or None when unknowable."""
    if ingress != "responses" or not isinstance(body, dict):
        return None
    prev = body.get("previous_response_id")
    if isinstance(prev, str) and prev:
        return "chain:" + prev
    get = headers.get if hasattr(headers, "get") else (lambda _k: None)
    originator = str(get("originator") or "")
    if originator.startswith("codex") or get("x-codex-turn-metadata") is not None:
        thread = get("thread-id") or get("session-id")
        if thread:
            return "codex-thread:" + str(thread).strip()
    return None


class SessionTracker:
    """(secret key, conversation ref) -> upstream ses_ id with TTL eviction."""

    def __init__(
        self,
        ttl_s: float = SESSION_TTL_S,
        max_entries: int = SESSION_MAX_ENTRIES,
    ):
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._sessions: dict[tuple[str, str], tuple[str, float]] = {}
        self._last_purge = time.monotonic()

    def _purge(self) -> None:
        now = time.monotonic()
        if now - self._last_purge < 60.0:
            return
        self._last_purge = now
        cutoff = now - self._ttl_s
        self._sessions = {k: v for k, v in self._sessions.items() if v[1] >= cutoff}
        while len(self._sessions) > self._max_entries:
            self._sessions.pop(next(iter(self._sessions)))

    def lookup(self, secret_key: str | None, ref: str) -> str | None:
        """Session id for a known conversation ref, else None."""
        self._purge()
        hit = self._sessions.get((secret_key or "anonymous", ref))
        if hit is not None and hit[1] >= time.monotonic() - self._ttl_s:
            return hit[0]
        return None

    def remember(self, secret_key: str | None, ref: str, session_id: str) -> None:
        """Map a conversation ref (or chained response id) to a session."""
        self._purge()
        self._sessions[(secret_key or "anonymous", ref)] = (
            session_id,
            time.monotonic(),
        )


async def warm_session(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    model: str,
    session_id: str,
    trace_id: str,
) -> None:
    """Fire a tiny title-shaped request so a new session opens like genuine.

    Best-effort: never raises (callers must still swallow BaseException
    cancellation separately). Streaming is required — the gate 403s
    non-streaming even with canonical instructions.
    """
    from llms.proxy.forward import STREAM_TIMEOUT_S
    from llms.proxy.zen_prompts import TITLE_PREFIX

    body = {
        "model": model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "instructions": TITLE_PREFIX,
        "prompt_cache_key": session_id,
        "stream": True,
        "max_output_tokens": WARM_MAX_TOKENS,
    }
    try:
        req = client.build_request("POST", url, headers=headers, json=body)
        req.extensions["timeout"] = {
            "connect": 10.0,
            "read": STREAM_TIMEOUT_S,
            "write": 10.0,
            "pool": 10.0,
        }
        upstream = await client.send(req, stream=True)
        try:
            if upstream.status_code >= 400:
                logger.debug(
                    "[%s] session warm got HTTP %s", trace_id, upstream.status_code
                )
                return
            async for _ in upstream.aiter_bytes():
                pass
        finally:
            try:
                await upstream.aclose()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("[%s] session warm failed: %r", trace_id, exc)
