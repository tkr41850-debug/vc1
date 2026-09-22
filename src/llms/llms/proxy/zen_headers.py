from __future__ import annotations

import hashlib
import uuid

from llms.proxy.config import Settings

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def stable_session_id(api_key: str) -> str:
    # Genuine opencode mints ses_ + 12 lowercase-hex time part + 14 base62
    # chars (see packages/schema/src/identifier.ts: descending() IDs, one
    # per session). Ours stays stable per proxy key (usage attribution
    # groups by key anyway) but matches the wire shape exactly.
    seed = api_key if api_key else "anonymous-free-tier"
    digest = hashlib.sha256(seed.encode()).digest()
    time_part = digest[:6].hex()
    rand_part = "".join(_BASE62[b % 62] for b in digest[6:20])
    return f"ses_{time_part}{rand_part}"


def new_request_id() -> str:
    return f"msg_{uuid.uuid4().hex[:16]}"


def build_zen_headers(
    settings: Settings, incoming_auth: str | None = None
) -> dict[str, str]:
    # Client credentials (sk- secrets, harness dummy keys) must never reach
    # the upstream gateway. Only the operator ZEN_API_KEY authenticates
    # upstream; without it requests ride the anonymous free tier.
    _ = incoming_auth
    api_key = settings.zen_api_key
    session_id = stable_session_id(api_key)
    headers = {
        "User-Agent": (
            f"opencode/{settings.opencode_channel}/"
            f"{settings.opencode_version}/{settings.opencode_client}"
        ),
        "x-opencode-client": settings.opencode_client,
        "x-opencode-project": settings.opencode_project,
        "x-opencode-session": session_id,
        "x-session-affinity": session_id,
        "x-session-id": session_id,
        "x-opencode-request": new_request_id(),
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
