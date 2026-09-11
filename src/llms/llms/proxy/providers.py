from __future__ import annotations

import asyncio
import base64
import copy
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

PROVIDERS_FILE = "providers.yaml"
RECENT_CAP = 10
HEALTH_TTL_S = 30.0


@dataclass
class Provider:
    id: str
    label: str = ""
    kind: str = "warp"  # "noproxy" or "warp"
    base_url: str = ""
    token: str = ""
    models: list[str] = field(default_factory=list)  # prefix patterns ("*" = all)
    enabled: bool = True

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
    active: int = 0
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
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._runtimes: dict[str, ProviderRuntime] = {}
        self._health_checked: dict[str, float] = {}
        self._client: httpx.AsyncClient | None = None
        # Populated by ProviderEgress.resolve() so refresh_health() can keep
        # each warp egress's slot spread in sync with ready exits.
        self._egresses: dict | None = None

    def path(self) -> Path:
        return self.data_dir / PROVIDERS_FILE

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
            out.append(
                Provider(
                    id=str(item["id"]),
                    label=str(item.get("label", "")),
                    kind=str(item.get("kind", "warp")),
                    base_url=str(item.get("base_url", "")),
                    token=str(item.get("token", "")),
                    models=[str(m) for m in item.get("models", []) or []],
                    enabled=bool(item.get("enabled", True)),
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
                        "base_url": p.base_url,
                        "token": p.token,
                        "models": p.models,
                        "enabled": p.enabled,
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

    def route(self, model: str) -> Provider | None:
        """First enabled provider serving the model; noproxy seed always present."""
        for p in self.load():
            if p.enabled and p.serves(model):
                return p
        return None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def refresh_health(
        self, provider: Provider, force: bool = False
    ) -> ProviderHealth:
        rt = self.runtime(provider.id)
        now = time.monotonic()
        if not force and now - rt.health.fetched_at < HEALTH_TTL_S and rt.health.exits:
            return rt.health
        health = ProviderHealth(fetched_at=now)
        if provider.kind != "warp" or not provider.base_url:
            rt.health = health
            return health
        try:
            client = await self._http()
            headers = {}
            if provider.token:
                headers["Authorization"] = f"Bearer {provider.token}"
            resp = await client.get(
                provider.base_url.rstrip("/") + "/health", headers=headers
            )
            resp.raise_for_status()
            payload = resp.json()
            health.active = int(payload.get("active", 0))
            for w in payload.get("warps", []) or []:
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
        # Keep the egress slot spread in sync with ready exits (at least 1).
        egress = self._egresses.get(provider.id) if self._egresses is not None else None
        if egress is not None:
            ready = sum(1 for w in health.exits if w.ready)
            egress.set_num_slots(max(1, ready or len(health.exits) or 1))
        return health

    async def fetch_debug_config(self, provider: Provider) -> dict:
        """Warp-cli metadata via the pool's debug/config surface (best-effort)."""
        if provider.kind != "warp" or not provider.base_url:
            return {}
        try:
            client = await self._http()
            headers = {}
            if provider.token:
                headers["Authorization"] = f"Bearer {provider.token}"
            resp = await client.get(
                provider.base_url.rstrip("/") + "/debug/config", headers=headers
            )
            if resp.status_code == 404:
                return {}
            resp.raise_for_status()
            payload = resp.json()
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            return {"error": str(exc)[:300]}

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def pool_active_warp(provider_health: ProviderHealth) -> int | None:
    """The pool's currently-active warp exit, if it reports one healthy.

    The pool routes every /fetch through its own active exit internally, so
    this is an observability snapshot — not a pin. Returns None when the
    pool reports no ready exits or the active exit isn't ready.
    """
    ready = {w.idx for w in provider_health.exits if w.ready}
    if provider_health.active in ready:
        return provider_health.active
    return None


def warp_exit_for(
    provider_health: ProviderHealth, bucket: int = 0, slot: int = 0
) -> int | None:
    """Deprecated alias of pool_active_warp (kept for tests)."""
    _ = (bucket, slot)
    return pool_active_warp(provider_health)


def fetch_spec(
    url: str, headers: dict, body: bytes, token: str = ""
) -> tuple[str, dict, bytes]:
    """Build a pool /fetch POST: returns (path, headers, json_body)."""
    out_headers = {
        k: v for k, v in headers.items() if k.lower() not in {"host", "content-length"}
    }
    if token:
        out_headers["Authorization"] = f"Bearer {token}"
    spec = {
        "url": url,
        "headers": out_headers,
        "body_b64": base64.b64encode(body).decode() if body else "",
    }
    return "/fetch", out_headers, json.dumps(spec).encode()


def parse_fetch_result(payload: dict) -> tuple[int, dict, bytes]:
    """Unwrap a pool /fetch response into (status, headers, body)."""
    if not payload.get("ok"):
        raise ValueError(str(payload.get("error", "pool fetch failed"))[:300])
    body = base64.b64decode(payload.get("body_b64", "") or "")
    headers = payload.get("headers", {}) or {}
    return int(payload.get("status", 502)), headers, body


def provider_snapshot(provider: Provider, rt: ProviderRuntime) -> dict:
    return {
        "id": provider.id,
        "label": provider.label,
        "kind": provider.kind,
        "base_url": provider.base_url,
        "has_token": bool(provider.token),
        "models": provider.models,
        "enabled": provider.enabled,
        "deletable": provider.id != "noproxy",
        "retry_in": round(rt.retry_in(), 1),
        "retry_reason": rt.retry_reason,
        "health": {
            "active": rt.health.active,
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
