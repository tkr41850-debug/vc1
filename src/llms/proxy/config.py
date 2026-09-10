from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    allow_client_keys: bool = field(
        default_factory=lambda: os.getenv("ZEN_ALLOW_CLIENT_KEYS", "0") == "1"
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("ZEN_GATEWAY_PORT", "8789"))
    )
    request_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("ZEN_TIMEOUT_S", "120"))
    )


def get_settings() -> Settings:
    return Settings()
