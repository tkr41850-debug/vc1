from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

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
    model_aliases: tuple = field(default_factory=lambda: _model_aliases())
    num_buckets: int = field(
        default_factory=lambda: int(os.getenv("NUM_BUCKETS", "1024"))
    )
    num_slots: int = field(default_factory=lambda: int(os.getenv("NUM_SLOTS", "8")))
    slot_cooldown_s: float = field(
        default_factory=lambda: float(os.getenv("SLOT_COOLDOWN_S", "60"))
    )
    egress_mode: str = field(default_factory=lambda: os.getenv("EGRESS_MODE", "direct"))
    warp_exits: int = field(
        default_factory=lambda: int(
            os.getenv("WARP_EXITS", os.getenv("WARP_SLOTS", "8"))
        )
    )
    warp_hold_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("WARP_HOLD_TIMEOUT", "10"))
    )
    warp_reg_interval_sec: int = field(
        default_factory=lambda: int(os.getenv("WARP_REG_INTERVAL_SEC", "28800"))
    )
    warp_boot_retry_sec: int = field(
        default_factory=lambda: int(os.getenv("WARP_BOOT_RETRY_SEC", "300"))
    )
    warp_base_socks_port: int = field(
        default_factory=lambda: int(os.getenv("WARP_BASE_SOCKS_PORT", "40001"))
    )
    warp_protocol: str = field(
        default_factory=lambda: os.getenv("WARP_PROTOCOL", "MASQUE")
    )
    warp_masque: str = field(default_factory=lambda: os.getenv("WARP_MASQUE", ""))
    warp_net_mtu: str = field(default_factory=lambda: os.getenv("WARP_NET_MTU", ""))
    port: int = field(
        default_factory=lambda: int(os.getenv("ZEN_GATEWAY_PORT", "8789"))
    )
    request_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("ZEN_TIMEOUT_S", "120"))
    )
    data_dir: str = field(default_factory=lambda: _default_data_dir())
    static_dir: str = field(
        default_factory=lambda: os.getenv(
            "STATIC_DIR",
            str(
                Path(__file__).resolve().parent.parent.parent
                / "llms"
                / "proxy"
                / "static"
            ),
        )
    )
    github_client_id: str = field(
        default_factory=lambda: os.getenv("GITHUB_CLIENT_ID", "")
    )
    github_client_secret: str = field(
        default_factory=lambda: os.getenv("GITHUB_CLIENT_SECRET", "")
    )
    github_redirect_uri: str = field(
        default_factory=lambda: os.getenv(
            "GITHUB_REDIRECT_URI", "http://localhost:8789/api/admin/callback"
        )
    )
    admin_github_users: tuple = field(
        default_factory=lambda: tuple(
            u.strip().lower()
            for u in os.getenv("ADMIN_GITHUB_USERS", "").split(",")
            if u.strip()
        )
    )
    admin_session_secret: str = field(
        default_factory=lambda: os.getenv("ADMIN_SESSION_SECRET", "")
    )


def _model_aliases() -> tuple:
    aliases: list = []
    for pair in os.getenv("MODEL_ALIASES", "").split(","):
        if "=" not in pair:
            continue
        pattern, _, target = pair.partition("=")
        if pattern.strip() and target.strip():
            aliases.append((pattern.strip(), target.strip()))
    return tuple(aliases)


def _default_data_dir() -> str:
    if explicit := os.getenv("DATA_DIR"):
        return explicit
    return str(Path(__file__).resolve().parent.parent.parent.parent.parent / "data")


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
