from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from llms.proxy.admin_hub import get_hub
from llms.proxy.auth import require_admin
from llms.proxy.config import Settings, settings_from_app
from llms.proxy.routes.admin import keys_snapshot, models_snapshot
from llms.proxy.routes.providers import providers_snapshot
from llms.proxy.store import Store

router = APIRouter()

# Collection live-update streams for the admin UI. They sit next to the
# REST list endpoints they mirror (same snapshot shape as GET
# /api/admin/{keys,models,providers}) so the browser holds one SSE per
# collection instead of polling GETs. Heartbeat cadence mirrors the
# per-provider recent-request stream
# (GET /api/admin/providers/{id}/stream): SSE comments only, never fake
# events, so proxies don't kill idle connections.
HEARTBEAT_S = 15.0


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


async def _collection_stream(
    request: Request, topic: str, snapshot
) -> StreamingResponse:
    async def gen():
        # Subscribe BEFORE the first snapshot: a CRUD publishing in between
        # stays queued and triggers an immediate second snapshot below,
        # so no write in that window is ever lost.
        q = await get_hub(request).subscribe(topic)
        try:
            yield _frame(snapshot())
            while True:
                if await request.is_disconnected():
                    break
                try:
                    await asyncio.wait_for(q.get(), timeout=HEARTBEAT_S)
                except TimeoutError:
                    yield ": ping\n\n"
                    continue
                yield _frame(snapshot())
        finally:
            await get_hub(request).unsubscribe(topic, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/admin/keys/sse")
async def keys_stream(
    request: Request,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = Store(data_dir=Path(settings.data_dir))

    def snapshot() -> dict:
        # Usage is live (in-memory counters), so re-read it on every event;
        # closing over the connect-time snapshot would go stale.
        return keys_snapshot(store, request.app.state.usage.snapshot())

    return await _collection_stream(request, "keys", snapshot)


@router.get("/api/admin/models/sse")
async def models_stream(
    request: Request,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = Store(data_dir=Path(settings.data_dir))
    return await _collection_stream(request, "models", lambda: models_snapshot(store))


@router.get("/api/admin/providers/sse")
async def providers_stream(request: Request, _admin: str = Depends(require_admin)):
    return await _collection_stream(
        request, "providers", lambda: providers_snapshot(request)
    )
