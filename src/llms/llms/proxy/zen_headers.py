from __future__ import annotations

import hashlib
import uuid

from llms.proxy.config import Settings


def stable_session_id(api_key: str) -> str:
    seed = api_key if api_key else "anonymous-free-tier"
    digest = hashlib.sha256(seed.encode()).hexdigest()[:16]
    return f"ses_{digest}"


def new_request_id() -> str:
    return f"msg_{uuid.uuid4().hex[:16]}"


def build_zen_headers(settings: Settings, incoming_auth: str | None) -> dict[str, str]:
    # The client's sk- secret key must never reach the upstream gateway.
    # Only the operator ZEN_API_KEY (or an explicitly allowed non-sk- client
    # bearer) authenticates upstream.
    api_key = settings.zen_api_key
    if (
        not api_key
        and settings.allow_client_keys
        and incoming_auth
        and incoming_auth.lower().startswith("bearer ")
    ):
        presented = incoming_auth.split(" ", 1)[1].strip()
        if not presented.startswith("sk-"):
            api_key = presented
    headers = {
        "User-Agent": f"opencode/{settings.opencode_version}",
        "x-opencode-client": settings.opencode_client,
        "x-opencode-project": settings.opencode_project,
        "x-opencode-session": stable_session_id(api_key),
        "x-opencode-request": new_request_id(),
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
