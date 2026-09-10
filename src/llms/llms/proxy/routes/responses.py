from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from llms.proxy.config import Settings, settings_from_app
from llms.proxy.pipeline import run

router = APIRouter()


@router.post("/v1/responses", response_model=None)
async def create_response_v1(
    request: Request, settings: Settings = Depends(settings_from_app)
) -> Response:
    return await run(request, settings, "responses")


@router.post("/responses", response_model=None)
async def create_response_bare(
    request: Request, settings: Settings = Depends(settings_from_app)
) -> Response:
    return await run(request, settings, "responses")
