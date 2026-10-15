from __future__ import annotations

import datetime as _datetime
import hashlib
import re as _re
import time as _time

from llms.proxy.config import Settings

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_SESSION_RE = _re.compile(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}\Z")


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


def plausible_session_id(
    value: object,
    now_ms: int | None = None,
    max_age_ms: int = 60 * 86400 * 1000,
) -> bool:
    """True when value decodes to a plausible descending session id.

    Shape plus timestamp plausibility (identifier.ts descending(): the
    head carries ``~(ts*0x1000+c)`` mod 2**48, so the stamp survives
    only mod 2**36 ms — checked on that ring against now, accepting up
    to ``max_age_ms`` old or 1h future skew; the counter borrow (<=1
    ring unit) is absorbed by the window). Uniform-garbage heads land
    in-window ~7% of the time at the default 60d — fine for a
    trust-but-verify gate whose fallback is a fresh mint (never leak a
    gate-killing id upstream, never reject a genuine long session).
    """
    if not isinstance(value, str) or not _SESSION_RE.match(value):
        return False
    low = (((1 << 48) - 1 - int(value[4:16], 16)) // 0x1000) % 2**36
    now = int(now_ms if now_ms is not None else _time.time() * 1000) % 2**36
    age = (now - low) % 2**36
    return age <= max_age_ms or age >= 2**36 - 3600_000


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
