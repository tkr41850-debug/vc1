from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llms.proxy.auth import require_admin
from llms.proxy.config import Settings, settings_from_app
from llms.proxy.providers import Provider, ProviderRegistry, snapshot_all
from llms.proxy.store import StoreError

router = APIRouter()


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
    slots: int = 8
    models: list[str] = []
    enabled: bool = True


class ProviderPatch(BaseModel):
    label: str | None = None
    slots: int | None = None
    models: list[str] | None = None
    enabled: bool | None = None


@router.get("/api/admin/providers")
async def list_providers(request: Request, _admin: str = Depends(require_admin)):
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
    if body.kind == "warp" and body.slots < 0:
        raise HTTPException(status_code=400, detail="slots must be >= 0")
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
        slots=max(0, body.slots),
        models=list(body.models),
        enabled=body.enabled,
    )
    providers.append(provider)
    registry.save(providers)
    if body.kind == "warp":
        registry.ensure_warp_dir(body.id)
    _sync_slots(request)
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
            if body.slots is not None:
                p.slots = max(0, body.slots)
            if body.models is not None:
                p.models = list(body.models)
            if body.enabled is not None:
                p.enabled = body.enabled
            registry.save(providers)
            _sync_slots(request)
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
    registry.drop_warp_dir(provider_id)
    _sync_slots(request)
    return {"status": "ok"}


@router.get("/api/admin/providers/{provider_id:path}/health")
async def provider_health(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    registry = _registry(request)
    provider = _find(registry, provider_id)
    health = await registry.refresh_health(provider, force=True)
    _sync_slots(request)
    debug = await registry.fetch_debug_config(provider)
    rt = registry.runtime(provider_id)
    from llms.proxy.providers import provider_snapshot

    snap = provider_snapshot(provider, rt)
    snap["debug"] = debug
    snap["health"]["fetched_at"] = health.fetched_at
    return snap


@router.post("/api/admin/providers/{provider_id:path}/reconnect")
async def provider_reconnect(
    request: Request, provider_id: str, _admin: str = Depends(require_admin)
):
    registry = _registry(request)
    provider = _find(registry, provider_id)
    if provider.kind != "warp":
        raise HTTPException(status_code=400, detail="only warp providers reconnect")
    result = await registry.reconnect(provider)
    _sync_slots(request)
    return {"id": provider_id, **result}


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
        yield f"data: {json.dumps({'recent': rt.recent_snapshot()})}\n\n"
        q = await rt.subscribe()
        try:
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
