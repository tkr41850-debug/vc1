from __future__ import annotations

import os
from dataclasses import dataclass, field

from fastapi import Request


@dataclass(frozen=True)
class Settings:
    zen_base_url: str = field(
        default_factory=lambda: os.getenv("ZEN_BASE_URL", "https://opencode.ai/zen/v1")
    )
    zen_api_key: str = field(default_factory=lambda: os.getenv("ZEN_API_KEY", ""))
    opencode_version: str = field(
        default_factory=lambda: os.getenv("ZEN_GATEWAY_OPENCODE_VERSION", "1.18.4")
    )
    opencode_client: str = field(
        default_factory=lambda: os.getenv("ZEN_GATEWAY_CLIENT", "cli")
    )
    opencode_project: str = field(
        default_factory=lambda: os.getenv("ZEN_GATEWAY_PROJECT", "global")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv(
            "ZEN_DEFAULT_MODEL", "muse-spark-1.3-contributor-free"
        )
    )
    default_chat_model: str = field(
        default_factory=lambda: os.getenv(
            "ZEN_DEFAULT_CHAT_MODEL", "muse-spark-1.3-contributor-free"
        )
    )
    default_messages_model: str = field(
        default_factory=lambda: os.getenv(
            "ZEN_DEFAULT_MESSAGES_MODEL", "claude-haiku-4-5"
        )
    )
    free_models: tuple = field(default_factory=lambda: _free_models())
    allow_client_keys: bool = field(
        default_factory=lambda: os.getenv("ZEN_ALLOW_CLIENT_KEYS", "0") == "1"
    )
    num_buckets: int = field(
        default_factory=lambda: int(os.getenv("NUM_BUCKETS", "1024"))
    )
    num_slots: int = field(default_factory=lambda: int(os.getenv("NUM_SLOTS", "8")))
    slot_cooldown_s: float = field(
        default_factory=lambda: float(os.getenv("SLOT_COOLDOWN_S", "60"))
    )
    egress_mode: str = field(default_factory=lambda: os.getenv("EGRESS_MODE", "direct"))
    vsp_base_url: str = field(default_factory=lambda: os.getenv("VSP_BASE_URL", ""))
    vsp_token: str = field(default_factory=lambda: os.getenv("VSP_TOKEN", ""))
    port: int = field(
        default_factory=lambda: int(os.getenv("ZEN_GATEWAY_PORT", "8789"))
    )
    request_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("ZEN_TIMEOUT_S", "120"))
    )


def _free_models() -> tuple:
    from llms.proxy.catalog import BY_ID

    override = os.getenv("ZEN_FREE_MODELS", "").strip()
    if override:
        return tuple(m.strip() for m in override.split(",") if m.strip())
    return tuple(BY_ID)


def get_settings() -> Settings:
    return Settings()


def settings_from_app(request: Request) -> Settings:
    return request.app.state.settings
