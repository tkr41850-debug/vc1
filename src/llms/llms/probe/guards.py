from __future__ import annotations

import os


def require_messages_live() -> None:
    if os.getenv("PROBE_MESSAGES_LIVE", "0") != "1":
        raise NotImplementedError(
            "messages live probe not implemented: no free messages-capable models on Zen yet"
        )
