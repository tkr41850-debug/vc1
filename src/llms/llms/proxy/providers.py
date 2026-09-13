from __future__ import annotations

import asyncio
import copy
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROVIDERS_FILE = "providers.yaml"
WARPS_DIR = "warps"
WARP_STATUS_FILE = "status.json"
RECENT_CAP = 10
HEALTH_TTL_S = 30.0


@dataclass
class Provider:
    id: str
    label: str = ""
    kind: str = "warp"  # "noproxy" or "warp"
    models: list[str] = field(default_factory=list)  # prefix patterns ("*" = all)
    enabled: bool = True
    exits: int = 8  # local warp exits owned by llms (per-provider pool size)

    def serves(self, model: str) -> bool:
        name = model.strip().lower()
        for pattern in self.models:
            p = pattern.strip().lower()
            if not p:
                continue
            if p == "*" or (p.endswith("*") and name.startswith(p[:-1])) or p == name:
                return True
        return False


@dataclass
class WarpExit:
    idx: int
    ready: bool = False
    status: str = "unknown"
    reason: str = ""
    socks: int = 0
    registered: bool = False
    error: str = ""


@dataclass
class ProviderHealth:
    exits: list[WarpExit] = field(default_factory=list)
    fetched_at: float = 0.0
    error: str = ""


@dataclass
class RecentRequest:
    ts: float
    model: str
    status: int
    ms: float
    warp_idx: int | None = None
    error: str = ""


class ProviderRuntime:
    """Live per-provider state: health snapshot, RetryIn, recent requests, SSE."""

    def __init__(self) -> None:
        self.health = ProviderHealth()
        self.retry_until: float = 0.0
        self.retry_reason: str = ""
        self.recent: deque[RecentRequest] = deque(maxlen=RECENT_CAP)
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    def retry_in(self) -> float:
        return max(0.0, self.retry_until - time.monotonic())

    def note_ratelimited(self, retry_after: float | None, reason: str = "") -> None:
        wait = 60.0
        if retry_after is not None:
            wait = max(wait, retry_after)
        self.retry_until = time.monotonic() + wait
        if reason:
            self.retry_reason = reason[:200]

    async def record(self, entry: RecentRequest) -> None:
        async with self._lock:
            self.recent.append(entry)
            dead = []
            for q in self._subscribers:
                try:
                    q.put_nowait(entry)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subscribers.discard(q)

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    def recent_snapshot(self) -> list[dict]:
        return [
            {
                "ts": r.ts,
                "model": r.model,
                "status": r.status,
                "ms": round(r.ms, 1),
                "warp_idx": r.warp_idx,
                "error": r.error,
            }
            for r in list(self.recent)
        ]


class ProviderRegistry:
    def __init__(self, data_dir: str | Path, settings=None, supervisor=None) -> None:
        self.data_dir = Path(data_dir)
        self._settings = settings
        self._supervisor = supervisor
        self._runtimes: dict[str, ProviderRuntime] = {}
        self._health_checked: dict[str, float] = {}
        # Populated by ProviderEgress.resolve() so refresh_health() can keep
        # each warp egress's slot spread in sync with ready exits.
        self._egresses: dict | None = None

    def _pool_config(self, provider: Provider):
        from llms.proxy import warp as _warp

        return _warp.WarpPoolConfig(
            exits=provider.exits
            if provider.exits
            else int(getattr(self._settings, "warp_exits", 8) or 8),
            hold_timeout_s=float(
                getattr(self._settings, "warp_hold_timeout_s", 10.0) or 10.0
            ),
            reg_interval_sec=int(
                getattr(self._settings, "warp_reg_interval_sec", 28800) or 28800
            ),
            boot_retry_sec=int(
                getattr(self._settings, "warp_boot_retry_sec", 300) or 300
            ),
            base_socks_port=int(
                getattr(self._settings, "warp_base_socks_port", 40001) or 40001
            ),
            protocol=str(
                getattr(self._settings, "warp_protocol", "MASQUE") or "MASQUE"
            ),
            masque=str(getattr(self._settings, "warp_masque", "") or ""),
        )

    def pool_for(self, provider: Provider):
        """The local WarpPool for a warp provider, or None without one."""
        if provider.kind != "warp" or self._supervisor is None:
            return None
        return self._supervisor.get(provider.id)

    async def ensure_pool(self, provider: Provider):
        """Get the started local WarpPool (starts it on first use)."""
        from llms.proxy import warp as _warp

        if provider.kind != "warp":
            return None
        if self._supervisor is None:
            self._supervisor = _warp.WarpSupervisor(self.data_dir)
        config = _warp.WarpPoolConfig(
            exits=provider.exits
            if provider.exits
            else int(getattr(self._settings, "warp_exits", 8) or 8),
            hold_timeout_s=float(
                getattr(self._settings, "warp_hold_timeout_s", 10.0) or 10.0
            ),
            reg_interval_sec=int(
                getattr(self._settings, "warp_reg_interval_sec", 28800) or 28800
            ),
            boot_retry_sec=int(
                getattr(self._settings, "warp_boot_retry_sec", 300) or 300
            ),
            base_socks_port=int(
                getattr(self._settings, "warp_base_socks_port", 40001) or 40001
            ),
            protocol=str(
                getattr(self._settings, "warp_protocol", "MASQUE") or "MASQUE"
            ),
            masque=str(getattr(self._settings, "warp_masque", "") or ""),
        )
        return await self._supervisor.ensure(provider.id, config)

    def path(self) -> Path:
        return self.data_dir / PROVIDERS_FILE

    def warp_dir(self, provider_id: str) -> Path:
        return self.data_dir / WARPS_DIR / provider_id

    def ensure_warp_dir(self, provider_id: str) -> Path:
        path = self.warp_dir(provider_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def drop_warp_dir(self, provider_id: str) -> None:
        import shutil

        shutil.rmtree(self.warp_dir(provider_id), ignore_errors=True)

    def save_warp_status(self, provider_id: str, health: ProviderHealth) -> None:
        path = self.ensure_warp_dir(provider_id) / WARP_STATUS_FILE
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "fetched_at": health.fetched_at,
                    "exits": [{"idx": w.idx, "ready": w.ready} for w in health.exits],
                }
            )
        )
        tmp.replace(path)

    def load_warp_status(self, provider_id: str) -> dict:
        path = self.warp_dir(provider_id) / WARP_STATUS_FILE
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def ready_exits(self, provider_id: str) -> int | None:
        """Persisted ready-exit count, or None when never polled."""
        status = self.load_warp_status(provider_id)
        exits = status.get("exits")
        if not isinstance(exits, list):
            return None
        return sum(1 for w in exits if isinstance(w, dict) and w.get("ready"))

    def _seed(self) -> list[Provider]:
        from llms.proxy.router import FREE_MODELS

        return [
            Provider(
                id="noproxy",
                label="Direct (Zen free tier)",
                kind="noproxy",
                models=list(FREE_MODELS),
                enabled=True,
            )
        ]

    def load(self) -> list[Provider]:
        from llms.proxy.store import StoreError

        path = self.path()
        if not path.exists():
            return self._seed()
        try:
            raw = yaml.safe_load(path.read_text()) or []
        except yaml.YAMLError as exc:
            raise StoreError(f"unparsable {PROVIDERS_FILE}: {exc}") from exc
        if not isinstance(raw, list):
            raise StoreError(f"unparsable {PROVIDERS_FILE}: expected a list")
        out = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            if str(item.get("kind", "warp")) == "warp" and item.get("base_url"):
                raise StoreError(
                    f"provider {item.get('id')}: remote pool base_url is no longer "
                    "supported — remove it and set exits (llms manages warp-cli "
                    "datadirs in-process)"
                )
            # `exits` sizes the local exit pool; `slots` is accepted as a
            # legacy alias from the slots-named era.
            try:
                exits = max(1, int(item.get("exits", item.get("slots", 8))))
            except (TypeError, ValueError):
                raise StoreError(f"provider {item.get('id')}: exits must be an integer")
            out.append(
                Provider(
                    id=str(item["id"]),
                    label=str(item.get("label", "")),
                    kind=str(item.get("kind", "warp")),
                    models=[str(m) for m in item.get("models", []) or []],
                    enabled=bool(item.get("enabled", True)),
                    exits=exits,
                )
            )
        if not any(p.id == "noproxy" for p in out):
            out = self._seed() + out
        return out

    def save(self, providers: list[Provider]) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(
                [
                    {
                        "id": p.id,
                        "label": p.label,
                        "kind": p.kind,
                        "models": p.models,
                        "enabled": p.enabled,
                        "exits": p.exits,
                    }
                    for p in providers
                ],
                sort_keys=False,
            )
        )
        tmp.replace(path)

    def runtime(self, provider_id: str) -> ProviderRuntime:
        rt = self._runtimes.get(provider_id)
        if rt is None:
            rt = ProviderRuntime()
            self._runtimes[provider_id] = rt
        return rt

    async def drop_egress(self, provider_id: str) -> bool:
        egresses = self._egresses
        if not egresses:
            return False
        egress = egresses.pop(provider_id, None)
        if egress is None:
            return False
        try:
            await egress.aclose()
        except Exception:
            pass
        return True

    def _health_summary(self, health: ProviderHealth) -> dict:
        return {
            "ready": sum(1 for w in health.exits if w.ready),
            "exits": len(health.exits),
            "error": health.error,
        }

    async def reconnect(self, provider: Provider) -> dict:
        """Manual reconnect: bounce local exits, clear backoff, re-poll."""
        before = self._health_summary(self.runtime(provider.id).health)
        try:
            pool = await self.ensure_pool(provider)
            result = await pool.reconnect()
        except Exception as exc:
            result = {"ok": False, "error": str(exc)[:300]}
        self.runtime(provider.id).retry_until = 0.0
        self.runtime(provider.id).retry_reason = ""
        health = await self.refresh_health(provider, force=True)
        return {
            "ok": result.get("ok", False),
            "before": before,
            "after": self._health_summary(health),
        }

    def route(self, model: str) -> Provider | None:
        """First enabled provider serving the model; noproxy seed always present."""
        for p in self.load():
            if p.enabled and p.serves(model):
                return p
        return None

    async def refresh_health(
        self, provider: Provider, force: bool = False
    ) -> ProviderHealth:
        rt = self.runtime(provider.id)
        now = time.monotonic()
        if (
            not force
            and now - rt.health.fetched_at < HEALTH_TTL_S
            and rt.health.exits
            and any(w.ready for w in rt.health.exits)
        ):
            return rt.health
        health = ProviderHealth(fetched_at=now)
        if provider.kind != "warp":
            rt.health = health
            return health
        try:
            pool = await self.ensure_pool(provider)
            await pool.refresh_statuses()
            snap = pool.snapshot()
            health.error = str(snap.get("error", "") or "")
            for w in snap.get("exits", []) or []:
                health.exits.append(
                    WarpExit(
                        idx=int(w.get("idx", 0)),
                        ready=bool(w.get("ready", False)),
                        status=str(w.get("status", "")),
                        reason=str(w.get("reason", "")),
                        socks=int(w.get("socks", 0)),
                        registered=bool(w.get("registered", False)),
                        error=str(w.get("error", ""))[:200],
                    )
                )
        except Exception as exc:
            health.error = str(exc)[:300]
        rt.health = health
        # Keep the egress slot spread in sync with ready exits (0 until any).
        # The egress object is created by ProviderEgress.resolve() from the
        # pool snapshot, so after a late promotion (slot flips ready during
        # this very refresh) push the ready SOCKS ports here — otherwise the
        # egress keeps zero ports and traffic fails open to direct even with
        # a connected tunnel (live finding: probe9/10 final status.json
        # showed ready=true while the request went direct).
        egress = self._egresses.get(provider.id) if self._egresses is not None else None
        if egress is not None:
            ready_ports = sorted(w.socks for w in health.exits if w.ready)
            egress.set_num_slots(len(ready_ports))
            if ready_ports:
                egress.set_socks_ports(ready_ports)
        try:
            self.save_warp_status(provider.id, health)
        except OSError:
            pass
        return health

    async def fetch_debug_config(self, provider: Provider) -> dict:
        """Warp-cli metadata from the local supervisor (best-effort)."""
        if provider.kind != "warp":
            return {}
        try:
            pool = await self.ensure_pool(provider)
            return await pool.debug_config()
        except Exception as exc:
            return {"error": str(exc)[:300]}

    async def aclose(self) -> None:
        if self._supervisor is not None:
            await self._supervisor.aclose()
            self._supervisor = None


def provider_snapshot(provider: Provider, rt: ProviderRuntime) -> dict:
    return {
        "id": provider.id,
        "label": provider.label,
        "kind": provider.kind,
        "models": provider.models,
        "enabled": provider.enabled,
        "exits": provider.exits,
        "deletable": provider.id != "noproxy",
        "retry_in": round(rt.retry_in(), 1),
        "retry_reason": rt.retry_reason,
        "health": {
            "fetched_at": rt.health.fetched_at,
            "error": rt.health.error,
            "exits": [
                {
                    "idx": w.idx,
                    "ready": w.ready,
                    "status": w.status,
                    "reason": w.reason,
                    "socks": w.socks,
                    "registered": w.registered,
                    "error": w.error,
                }
                for w in rt.health.exits
            ],
        },
    }


def snapshot_all(registry: ProviderRegistry) -> list[dict]:
    return [provider_snapshot(p, registry.runtime(p.id)) for p in registry.load()]


def snapshot_copy(data: dict) -> dict:
    return copy.deepcopy(data)
