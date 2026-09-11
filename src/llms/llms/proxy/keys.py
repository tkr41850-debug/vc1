from __future__ import annotations

import os
import time
from pathlib import Path

from starlette.requests import Request

from llms.proxy.config import Settings
from llms.proxy.store import Store

# Reload keys.yaml at most this often; file edits take effect within the
# interval without re-parsing YAML on every request.
KEYS_TTL_S = float(os.getenv("KEYS_CACHE_TTL_S", "1.0"))

_cache: dict = {"mtime": 0.0, "checked": 0.0, "keys": {}}


def _snapshot(settings: Settings) -> dict[str, bool]:
    """{key: enabled} cache, reloaded on mtime change (checked at most per TTL).

    Raises StoreError when keys.yaml exists but cannot be parsed — callers
    fail closed; healthz surfaces the condition separately.
    """
    path = Store(data_dir=Path(settings.data_dir)).keys_path()
    now = time.monotonic()
    if now - _cache["checked"] < KEYS_TTL_S and _cache["mtime"]:
        return _cache["keys"]
    _cache["checked"] = now
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _cache.update(mtime=now, keys={})
        return {}
    if mtime == _cache["mtime"] and _cache["keys"] is not None:
        return _cache["keys"]
    store = Store(data_dir=path.parent)
    keys = {k.key: k.enabled for k in store.load_keys()}
    _cache.update(mtime=mtime, keys=keys)
    return keys


def resolve_secret_key(request: Request, settings: Settings) -> str | None:
    """The sk- secret key for this request, or None if missing/unknown/disabled.

    The sk- key lives only on the header (Authorization: Bearer / x-api-key).
    It is authenticated here, never forwarded upstream (see zen_headers) and
    never mixed with the ak- affinity prefix (see middleware).
    """
    from llms.proxy.auth import presented_secret_key
    from llms.proxy.logging import setup_logging

    presented = presented_secret_key(request)
    if presented is None:
        return None
    snapshot = _snapshot(settings)
    if snapshot.get(presented, False):
        return presented
    canonical = _strip_ant_infix(presented)
    if canonical is not None and snapshot.get(canonical, False):
        setup_logging().info("accepted sk-ant- key form for an allowlisted key")
        return canonical
    return None


def _strip_ant_infix(value: str) -> str | None:
    if value.startswith("sk-ant-"):
        return "sk-" + value[len("sk-ant-") :]
    return None


def reset_cache() -> None:
    global _cache
    _cache = {"mtime": 0.0, "checked": 0.0, "keys": {}}
