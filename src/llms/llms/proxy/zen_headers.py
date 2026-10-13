from __future__ import annotations

import datetime as _datetime
import hashlib
import time as _time

from llms.proxy.config import Settings

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def descending_session_id(now_ms: int | None = None, rand: bytes | None = None) -> str:
    """Genuine-shape ses_ id with a real timestamp head.

    Faithful port of identifier.ts ``create(true)``: the 12-hex head is
    ``~((timestamp_ms * 0x1000) + 1)`` (descending IDs sort newest
    first), followed by 14 random base62 chars. The head MUST decode
    to a plausible timestamp — Zen's gate rejects ids whose time part
    is garbage (probed Oct 2026: random-head ids 403 even with a
    perfect wire shape; 30-day-old time-ordered ids 200). Freshness
    beyond plausibility is not enforced, so stability buckets (day)
    are safe; randomness never is.
    """
    import os as _os

    ts = int(now_ms if now_ms is not None else _time.time() * 1000)
    value = (~(ts * 0x1000 + 1)) & ((1 << 48) - 1)
    head = "".join(f"{(value >> (40 - 8 * i)) & 0xFF:02x}" for i in range(6))
    raw = rand if rand is not None and len(rand) >= 14 else _os.urandom(14)
    return "ses_" + head + "".join(_BASE62[b % 62] for b in raw[:14])


def stable_session_id(api_key: str, now_s: float | None = None) -> str:
    # Stable per proxy key per UTC day (usage attribution groups by key
    # anyway): the random tail is key-derived, but the head is a real
    # descending timestamp for the day so the gate's plausibility check
    # passes. A fully-stable id would carry a hash-garbage head and
    # 403 (see descending_session_id); a fully-fresh id would cold the
    # upstream cache on every untracked request.
    now = now_s if now_s is not None else _time.time()
    day = _datetime.datetime.fromtimestamp(now, tz=_datetime.UTC).date()
    day_start_ms = int(
        _datetime.datetime(
            day.year, day.month, day.day, tzinfo=_datetime.UTC
        ).timestamp()
        * 1000
    )
    seed = api_key if api_key else "anonymous-free-tier"
    digest = hashlib.sha256(f"{seed}\x00{day.isoformat()}".encode()).digest()
    return descending_session_id(now_ms=day_start_ms, rand=digest[6:20])


def build_zen_headers(
    settings: Settings, incoming_auth: str | None = None, session_id: str | None = None
) -> dict[str, str]:
    # Client credentials (sk- secrets, harness dummy keys) must never reach
    # the upstream gateway. Only the operator ZEN_API_KEY authenticates
    # upstream; without it requests ride the anonymous free tier as
    # `Bearer public` (observed genuine v2 wire — the header is required,
    # omitting it 403s even with everything else correct).
    _ = incoming_auth
    api_key = settings.zen_api_key
    session_id = session_id or stable_session_id(api_key)
    headers = {
        "User-Agent": (
            f"opencode/{settings.opencode_channel}/"
            f"{settings.opencode_version}/{settings.opencode_client}"
        ),
        "x-opencode-client": settings.opencode_client,
        "x-opencode-project": settings.opencode_project,
        "x-opencode-session": session_id,
        "x-opencode-session-id": session_id,
        "x-session-affinity": session_id,
        "x-session-id": session_id,
        # NOTE: no x-opencode-request — upstream dropped it
        # (model-request.ts no longer sends it) and its presence 403s
        # per the original bisect, so the gateway leaves it out.
        "Content-Type": "application/json",
    }
    headers["Authorization"] = f"Bearer {api_key}" if api_key else "Bearer public"
    return headers
