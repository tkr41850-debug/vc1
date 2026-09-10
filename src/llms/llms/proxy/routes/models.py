from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from llms.proxy.catalog import BY_ID
from llms.proxy.config import Settings, get_settings

router = APIRouter()


@router.get("/v1/models", response_model=None)
async def list_models_v1(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await handle_models(settings)


@router.get("/models", response_model=None)
async def list_models_bare(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await handle_models(settings)


def entry_for(model_id: str, created: int) -> dict:
    entry = {"id": model_id, "object": "model", "created": created, "owned_by": "llms"}
    known = BY_ID.get(model_id)
    if known:
        entry.update({k: v for k, v in known.items() if k != "id"})
    return entry


async def handle_models(settings: Settings) -> Response:
    created = int(time.time())
    return JSONResponse(
        content={
            "object": "list",
            "data": [entry_for(m, created) for m in settings.free_models],
        }
    )
