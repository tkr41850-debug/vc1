"""Local -> upstream-llms relay (borrow an upstream gateway's load balancing).

When UPSTREAM_RELAY_URL + UPSTREAM_RELAY_KEY are set, the local gateway
forwards its anonymous-shaped upstream request to the upstream llms
gateway INSTEAD of opencode.ai/zen — and presents as a genuine opencode
client so the upstream passes through without re-shaping.

Why this shape: the upstream gateway's `is_genuine_opencode` UA check
decides passthrough. A non-opencode UA gets the upstream's anonymous
shaping applied ON TOP of the local shaping (double notice, double
tool injection). So the local hop keeps the full opencode identity
headers (UA, x-opencode-*, ses_ session) and swaps ONLY the
Authorization to the relay sk- key (the upstream is an llms gateway:
it expects sk- there, resolves it via its own middleware, and
re-derives the anonymous zen shape for its own upstream hop).

Credential rule: the relay key lives ONLY in server env
(UPSTREAM_RELAY_KEY). Client sk- secrets never leave the local box —
the zen_headers rule (only the operator key authenticates upstream)
extends here: only the relay key authenticates the local->upstream
hop.

Half-configuration (URL without key or vice versa) is a ValueError,
never a silent direct — a relay that silently stops relaying would
burn anonymous quota while looking healthy.
"""

from __future__ import annotations

from llms.proxy.config import Settings
from llms.proxy.router import ENDPOINT_PATH
from llms.proxy.zen_headers import build_zen_headers


def relay_enabled(settings: Settings) -> bool:
    """True when BOTH relay env knobs are set (else direct-to-zen)."""
    return bool(
        getattr(settings, "upstream_relay_url", "").strip()
        and getattr(settings, "upstream_relay_key", "").strip()
    )


def relay_url_for(settings: Settings, egress: str) -> str:
    """Upstream URL for one egress dialect: relay gateway or direct zen.

    Raises ValueError on half-configuration (URL without key or vice
    versa) — callers fail closed instead of silently going direct.
    """
    url = (getattr(settings, "upstream_relay_url", "") or "").strip()
    key = (getattr(settings, "upstream_relay_key", "") or "").strip()
    if bool(url) != bool(key):
        raise ValueError(
            "upstream relay needs both UPSTREAM_RELAY_URL and UPSTREAM_RELAY_KEY"
        )
    if url:
        return url.rstrip("/") + ENDPOINT_PATH[egress]
    return settings.zen_base_url.rstrip("/") + ENDPOINT_PATH[egress]


def relay_headers_for(settings: Settings, session_id: str) -> dict[str, str]:
    """Headers for the local->upstream hop: opencode identity + relay key.

    build_zen_headers supplies the full genuine shape (UA,
    x-opencode-*, ses_ session, Content-Type); only the Authorization
    swaps from `Bearer public` to the relay sk- key. The upstream
    gateway resolves the key via its own middleware and re-derives the
    zen shape itself.
    """
    headers = build_zen_headers(settings, session_id=session_id)
    headers["Authorization"] = f"Bearer {settings.upstream_relay_key.strip()}"
    return headers
