from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

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


async def handle_models(settings: Settings) -> Response:
    created = int(time.time())
    return JSONResponse(
        content={
            "object": "list",
            "data": [
                {"id": model, "object": "model", "created": created, "owned_by": "llms"}
                for model in settings.free_models
            ],
        }
    )
