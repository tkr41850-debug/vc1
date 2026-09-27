"""Cloudflare-friendly slow-request dedup (non-streaming legs only).

Background: mute clients (Claude Code falling back to non-streaming)
sit behind Cloudflare, which kills quiet connections well before our
600s upstream budget. The client retries the identical body — today
that restarts the whole upstream generation from scratch.

With this, a non-streaming request that outruns MAX_TIMEOUT gets
`429 + Retry-After: 20` while its upstream work CONTINUES in the
background. A retry with a matching request hash gets 429 again while
still running, or the held response once done. Claiming a held
response evicts the mapping immediately (one stored response serves
one claim); unclaimed results expire after DEDUP_HOLD_S.
Streaming legs bypass this entirely (heartbeats already keep them
alive, and streams can't be held).

Key design points:
- The hash covers (secret, affinity, ingress, model, canonical body,
  session): identical retries match; distinct conversations (distinct
  sessions) never share results.
- Fresh responses-leg conversations would mint a new session per
  attempt and defeat the hash, so the session block reserves the
  minted id under the session-less hash (see reserve_session) —
  retries then reuse the session AND its prompt-cache affinity.
- Dedup 429s/hits carry an x-llms-dedup marker; the pipeline skips all
  post-processing (ratelimit outcome, provider notes, usage, warming)
  for marked responses so a synthetic 429 can never poison the pool.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

DEDUP_RETRY_AFTER_S = 20
DEDUP_HOLD_S = 300.0
DEDUP_MAX_ENTRIES = 128
DEDUP_INFLIGHT_TTL_S = 900.0
RESERVE_TTL_S = 600.0
RESERVE_MAX_ENTRIES = 10000
DEDUP_HEADER = "x-llms-dedup"


def canonical_body(body: dict) -> str:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)


def request_hash(
    secret_key: str | None,
    affinity: str | None,
    ingress: str,
    model: str,
    body: dict,
    session_id: str | None = None,
) -> str:
    raw = "\x00".join(
        [
            secret_key or "",
            affinity or "",
            ingress,
            model,
            canonical_body(body),
            session_id or "",
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def reserve_key(
    secret_key: str | None,
    affinity: str | None,
    ingress: str,
    model: str,
    body: dict,
) -> str:
    return request_hash(secret_key, affinity, ingress, model, body, None)


class _Entry:
    __slots__ = ("body", "expires", "headers", "status", "task", "timed_out")

    def __init__(self, task: asyncio.Task | None) -> None:
        self.task = task
        self.status: int | None = None
        self.body: bytes = b""
        self.headers: dict = {}
        self.expires = time.monotonic() + DEDUP_INFLIGHT_TTL_S
        # Set when the foreground waiter gives up (TimeoutError): only then
        # does the background completion own usage/conversation/warming
        # bookkeeping. Fast completions leave those to the normal tail.
        self.timed_out = False

    @property
    def done(self) -> bool:
        return self.status is not None


_HOP_BY_HOP = frozenset(
    {"content-length", "connection", "server", "date", "transfer-encoding"}
)


class DedupTable:
    """In-flight + recently completed non-streaming responses by hash."""

    def __init__(
        self,
        max_entries: int = DEDUP_MAX_ENTRIES,
        hold_s: float = DEDUP_HOLD_S,
    ) -> None:
        self._entries: dict[str, _Entry] = {}
        self._max_entries = max(1, max_entries)
        self._hold_s = hold_s

    def _purge(self) -> None:
        now = time.monotonic()
        dead = [k for k, e in self._entries.items() if e.expires <= now]
        for k in dead:
            self._entries.pop(k, None)
        while len(self._entries) > self._max_entries:
            self._entries.pop(next(iter(self._entries)))

    def lookup(self, key: str) -> _Entry | None:
        self._purge()
        return self._entries.get(key)

    def track(self, key: str, task: asyncio.Task) -> _Entry:
        """Register an in-flight task; drops the entry if the task dies."""
        self._purge()
        entry = _Entry(task)

        def _drop_on_failure(done_task: asyncio.Task) -> None:
            try:
                done_task.result()
            except BaseException:
                self._entries.pop(key, None)

        task.add_done_callback(_drop_on_failure)
        self._entries[key] = entry
        while len(self._entries) > self._max_entries:
            self._entries.pop(next(iter(self._entries)))
        return entry

    def complete(self, key: str, status: int, body: bytes, headers: dict) -> None:
        """Store a finished result, replacing any in-flight entry."""
        self._purge()
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(None)
            self._entries[key] = entry
        entry.task = None
        entry.status = status
        entry.body = bytes(body)
        entry.headers = dict(headers)
        entry.expires = time.monotonic() + self._hold_s
        while len(self._entries) > self._max_entries:
            self._entries.pop(next(iter(self._entries)))

    def drop(self, key: str) -> None:
        self._entries.pop(key, None)

    @staticmethod
    def store_headers(headers) -> dict:
        out = {}
        for k, v in dict(headers).items():
            if k.lower() not in _HOP_BY_HOP:
                out[k] = v
        return out


def inflight_response():
    """429 telling a mute client to come back in DEDUP_RETRY_AFTER_S."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=429,
        content={"error": {"message": "upstream request still running, retry shortly"}},
        headers={"retry-after": str(DEDUP_RETRY_AFTER_S), DEDUP_HEADER: "inflight"},
    )


def replay_response(status: int, body: bytes, headers: dict):
    """Rebuild a held response; callers must evict the mapping after."""
    from fastapi.responses import JSONResponse

    try:
        content = json.loads(bytes(body).decode())
    except Exception:
        content = {"error": {"message": "stored response unreadable"}}
    replay_headers = {
        k: v for k, v in dict(headers).items() if k.lower() not in _HOP_BY_HOP
    }
    replay_headers[DEDUP_HEADER] = "hit"
    return JSONResponse(status_code=status, content=content, headers=replay_headers)


class SessionReservations:
    """Fresh-mint session ids keyed by session-less request hash.

    A brand-new responses conversation has no ref to track, so without
    this every client retry would mint a different session — defeating
    both the dedup hash and prompt-cache affinity. The first attempt
    reserves its minted id; retries reuse it.
    """

    def __init__(
        self,
        max_entries: int = RESERVE_MAX_ENTRIES,
        ttl_s: float = RESERVE_TTL_S,
    ) -> None:
        self._entries: dict[str, tuple[str, float]] = {}
        self._max_entries = max(1, max_entries)
        self._ttl_s = ttl_s

    def _purge(self) -> None:
        now = time.monotonic()
        dead = [k for k, v in self._entries.items() if v[1] <= now]
        for k in dead:
            self._entries.pop(k, None)
        while len(self._entries) > self._max_entries:
            self._entries.pop(next(iter(self._entries)))

    def lookup(self, key: str) -> str | None:
        self._purge()
        hit = self._entries.get(key)
        if hit is not None and hit[1] > time.monotonic():
            return hit[0]
        return None

    def remember(self, key: str, session_id: str) -> None:
        self._purge()
        self._entries[key] = (session_id, time.monotonic() + self._ttl_s)
