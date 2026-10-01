from __future__ import annotations

import asyncio
import ipaddress
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llms.proxy.admin_hub import get_hub
from llms.proxy.auth import require_admin
from llms.proxy.config import Settings, settings_from_app
from llms.proxy.providers import Provider, ProviderRegistry, snapshot_all
from llms.proxy.store import StoreError

router = APIRouter()

operator_router = APIRouter()

# Upper bound on warp exits per provider: each exit allocates a SOCKS
# port, a slot, datadirs and a daemon, so an uncapped admin input fans
# straight out to the pool (verified frontend finding).
MAX_EXITS = 32


def _clamp_exits(exits: int) -> int:
    return max(1, min(MAX_EXITS, int(exits)))

IP_ECHO_URL = "https://api.ipify.org?format=json"
IP6_ECHO_URL = "https://api64.ipify.org?format=json"
IP_TIMEOUT_S = 10.0
IP_CACHE_TTL_S = 60.0
_ip_cache: dict[str, tuple[float, dict]] = {}


def _registry(request: Request) -> ProviderRegistry:
    registry = getattr(request.app.state, "providers", None)
    if registry is None:
        settings: Settings = request.app.state.settings
        registry = ProviderRegistry(data_dir=settings.data_dir)
        request.app.state.providers = registry
    return registry


def _find(registry: ProviderRegistry, provider_id: str) -> Provider:
    for p in registry.load():
        if p.id == provider_id:
            return p
    raise HTTPException(status_code=404, detail="provider not found")


def _sync_slots(request: Request) -> None:
    egress = getattr(request.app.state, "egress", None)
    table = getattr(request.app.state, "bucket_table", None)
    sync = getattr(egress, "sync_bucket_slots", None)
    if callable(sync) and table is not None:
        sync(table)


class ProviderBody(BaseModel):
    id: str = ""
    label: str = ""
    kind: str = "warp"
    models: list[str] = []
    enabled: bool = True
    exits: int = 8


class ProviderPatch(BaseModel):
    label: str | None = None
    models: list[str] | None = None
    enabled: bool | None = None
    exits: int | None = None


@router.get("/api/admin/providers")
async def list_providers(request: Request, _admin: str = Depends(require_admin)):
    return providers_snapshot(request)


def providers_snapshot(request: Request) -> dict:
    """Shared builder for GET /api/admin/providers and GET /api/admin/providers/sse."""
    try:
        return {"providers": snapshot_all(_registry(request))}
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/api/admin/providers", status_code=201)
async def create_provider(
    request: Request,
    body: ProviderBody,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    if not body.id:
        raise HTTPException(status_code=400, detail="id is required")
    if body.kind not in ("noproxy", "warp"):
        raise HTTPException(status_code=400, detail="kind must be noproxy or warp")
    if body.kind == "warp" and (body.exits < 1 or body.exits > MAX_EXITS):
        raise HTTPException(
            status_code=400, detail=f"exits must be 1..{MAX_EXITS}"
        )
    registry = _registry(request)
    _ = settings
    try:
        providers = registry.load()
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if any(p.id == body.id for p in providers):
        raise HTTPException(status_code=409, detail="provider already exists")
    provider = Provider(
        id=body.id,
        label=body.label,
        kind=body.kind,
        models=list(body.models),
        enabled=body.enabled,
        exits=_clamp_exits(body.exits),
    )
    providers.append(provider)
    registry.save(providers)
    if body.kind == "warp":
        registry.ensure_warp_dir(body.id)
    _sync_slots(request)
    await get_hub(request).publish("providers")
    return {"id": body.id}


@router.put("/api/admin/providers/{provider_id:path}")
async def update_provider(
    request: Request,
    provider_id: str,
    body: ProviderPatch,
    _admin: str = Depends(require_admin),
):
    registry = _registry(request)
    try:
        providers = registry.load()
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    for p in providers:
        if p.id == provider_id:
            if body.label is not None:
                p.label = body.label
            if body.exits is not None:
                p.exits = _clamp_exits(body.exits)
            if body.models is not None:
                p.models = list(body.models)
            if body.enabled is not None:
                p.enabled = body.enabled
            registry.save(providers)
            _sync_slots(request)
            await get_hub(request).publish("providers")
            return {"id": p.id, "enabled": p.enabled}
    raise HTTPException(status_code=404, detail="provider not found")


@router.delete("/api/admin/providers/{provider_id:path}")
async def delete_provider(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    if provider_id == "noproxy":
        raise HTTPException(status_code=403, detail="default provider is not deletable")
    registry = _registry(request)
    try:
        providers = [p for p in registry.load() if p.id != provider_id]
        if len(providers) == len(registry.load()):
            raise HTTPException(status_code=404, detail="provider not found")
    except StoreError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    registry.save(providers)
    # Datadirs stay on disk: Cloudflare rate-limits re-registration, so a
    # same-id re-create reuses them instead of burning registration budget.
    # Only the live pool (daemons + clients) is dropped.
    supervisor = getattr(registry, "_supervisor", None)
    if supervisor is not None:
        await supervisor.drop(provider_id)
    invalidate_ips(provider_id)
    _sync_slots(request)
    await get_hub(request).publish("providers")
    return {"status": "ok"}


@router.get("/api/admin/providers/{provider_id:path}/health")
async def provider_health(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    registry = _registry(request)
    provider = _find(registry, provider_id)
    health = await registry.refresh_health(provider, force=True)
    _sync_slots(request)
    # Force-refresh mutates the shared snapshot; push it so SSE clients
    # (no polling) converge without waiting for the next CRUD/reconnect.
    await get_hub(request).publish("providers")
    debug = await registry.fetch_debug_config(provider)
    rt = registry.runtime(provider_id)
    from llms.proxy.providers import provider_snapshot

    settings = getattr(request.app.state, "settings", None)
    snap = provider_snapshot(
        provider,
        rt,
        float(getattr(settings, "warp_auto_cycle_cooldown_s", 300) or 300),
        registry,
    )
    snap["debug"] = debug
    snap["health"]["fetched_at"] = health.fetched_at
    return snap


async def _do_reconnect(request: Request, provider_id: str) -> dict:
    registry = _registry(request)
    provider = _find(registry, provider_id)
    if provider.kind != "warp":
        raise HTTPException(status_code=400, detail="only warp providers reconnect")
    result = await registry.reconnect(provider)
    invalidate_ips(provider_id)
    _sync_slots(request)
    # Reconnect changes health/retry state in the snapshot; push it so SSE
    # clients (no polling) see the new state without refetching.
    await get_hub(request).publish("providers")
    return {"id": provider_id, **result}


@router.post("/api/admin/providers/{provider_id:path}/reconnect")
async def provider_reconnect(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    return await _do_reconnect(request, provider_id)


@operator_router.post("/api/providers/{provider_id:path}/reconnect")
async def provider_reconnect_local(request: Request, provider_id: str):
    if getattr(request.state, "secret_key", None) is None:
        raise HTTPException(status_code=401, detail="missing or invalid secret key")
    return await _do_reconnect(request, provider_id)


async def _fetch_ip(proxy_url: str | None) -> str:
    """Egress IP as the outside world sees it, via an optional SOCKS exit.

    Separated for tests: production passes socks5:// URLs matching the
    real traffic path (see egress._socks_client); tests patch this out.
    """
    import httpx

    kwargs: dict = {"timeout": IP_TIMEOUT_S, "trust_env": False}
    if proxy_url:
        kwargs["proxy"] = proxy_url
    async with httpx.AsyncClient(**kwargs) as client:
        response = await client.get(IP_ECHO_URL)
        response.raise_for_status()
        payload = response.json()
    ip = payload.get("ip") if isinstance(payload, dict) else None
    if not isinstance(ip, str) or not ip:
        raise ValueError("ip echo returned no ip")
    return ip


def _is_ipv6(value: object) -> bool:
    try:
        return isinstance(value, str) and ipaddress.ip_address(value).version == 6
    except ValueError:
        return False


async def _fetch_ip6(proxy_url: str | None) -> str | None:
    """Egress IPv6 as the outside world sees it, or None without v6 egress.

    api.ipify.org is IPv4-only, so the v4 probe above can never show the
    v6 address WARP typically assigns. api64.ipify.org answers over
    whichever family the exit actually dials out on; anything that is
    not a v6 literal (no v6 route, or v4 fallback echoing the v4
    address) maps to None. Best-effort by design: missing v6 is normal,
    so failures never raise and never flip the row to error.
    Separated for tests like _fetch_ip.
    """
    import httpx

    kwargs: dict = {"timeout": IP_TIMEOUT_S, "trust_env": False}
    if proxy_url:
        kwargs["proxy"] = proxy_url
    try:
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.get(IP6_ECHO_URL)
            response.raise_for_status()
            payload = response.json()
        ip = payload.get("ip") if isinstance(payload, dict) else None
        return ip if _is_ipv6(ip) else None
    except Exception:
        return None


async def _do_ips(request: Request, provider_id: str) -> dict:
    registry = _registry(request)
    provider = _find(registry, provider_id)
    now = time.monotonic()
    ports_key = _egress_ports_key(registry, provider)
    hit = _ip_cache.get(provider_id)
    # Besides the TTL, exit churn busts the cache: reconnects, bounces
    # and restarts change the ready-port set, and a fresh circuit can
    # present a different IP even on a reused port.
    if hit is not None and now - hit[0] < IP_CACHE_TTL_S and hit[2] == ports_key:
        return {**hit[1], "cached": True}
    if provider.kind == "warp":
        rt = registry.runtime(provider_id)

        async def _one(exit):
            if not exit.ready or not exit.socks:
                return {
                    "idx": exit.idx,
                    "port": exit.socks or None,
                    "ip": None,
                    "ipv6": None,
                    "error": "exit not ready",
                }
            proxy_url = f"socks5://127.0.0.1:{exit.socks}"
            try:
                ip = await _fetch_ip(proxy_url)
            except Exception as exc:
                return {
                    "idx": exit.idx,
                    "port": exit.socks,
                    "ip": None,
                    "ipv6": None,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            return {
                "idx": exit.idx,
                "port": exit.socks,
                "ip": ip,
                "ipv6": await _fetch_ip6(proxy_url),
                "error": None,
            }

        entries = await asyncio.gather(
            *(_one(w) for w in sorted(rt.health.exits, key=lambda w: w.idx))
        )
    else:

        async def _direct():
            try:
                ip = await _fetch_ip(None)
            except Exception as exc:
                return {
                    "idx": None,
                    "port": None,
                    "ip": None,
                    "ipv6": None,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                }
            return {
                "idx": None,
                "port": None,
                "ip": ip,
                "ipv6": await _fetch_ip6(None),
                "error": None,
            }

        entries = [await _direct()]
    payload = {
        "id": provider_id,
        "kind": provider.kind,
        "ips": list(entries),
        "cached": False,
    }
    _ip_cache[provider_id] = (now, payload, ports_key)
    return payload


def _egress_ports_key(registry, provider) -> tuple:
    """Cache-busting key for the current exit set (see _do_ips)."""
    if provider.kind != "warp":
        return ("direct",)
    try:
        exits = registry.runtime(provider.id).health.exits
    except Exception:
        return ()
    return tuple(sorted(w.socks for w in exits if w.ready and w.socks))


def invalidate_ips(provider_id: str) -> None:
    """Drop a provider's cached egress IPs (reconnect / pool churn)."""
    _ip_cache.pop(provider_id, None)


@router.get("/api/admin/providers/{provider_id:path}/ips")
async def provider_ips(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    return await _do_ips(request, provider_id)


@operator_router.get("/api/providers/{provider_id:path}/ips")
async def provider_ips_local(request: Request, provider_id: str):
    if getattr(request.state, "secret_key", None) is None:
        raise HTTPException(status_code=401, detail="missing or invalid secret key")
    return await _do_ips(request, provider_id)


@router.get("/api/admin/providers/{provider_id:path}/recent")
async def provider_recent(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    registry = _registry(request)
    _find(registry, provider_id)
    return {"recent": registry.runtime(provider_id).recent_snapshot()}


@router.get("/api/admin/providers/{provider_id:path}/stream")
async def provider_stream(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    registry = _registry(request)
    _find(registry, provider_id)
    rt = registry.runtime(provider_id)

    async def gen():
        # Subscribe BEFORE the first snapshot (mirrors the collection
        # streams). The snapshot is taken under the same lock, so a
        # record() landing in between is either in the snapshot or in
        # the queue — never both, never neither.
        q, snap = await rt.subscribe_snapshot()
        try:
            yield f"data: {json.dumps({'recent': snap})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=15.0)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield f"data: {json.dumps({'request': {'ts': entry.ts, 'model': entry.model, 'status': entry.status, 'ms': round(entry.ms, 1), 'warp_idx': entry.warp_idx, 'error': entry.error}})}\n\n"
        finally:
            await rt.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")
