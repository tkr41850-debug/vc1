from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from llms.proxy.config import Settings, get_settings
from llms.proxy.pipeline import run

router = APIRouter()


@router.post("/v1/messages", response_model=None)
async def create_message_v1(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await run(request, settings, "messages")


@router.post("/messages", response_model=None)
async def create_message_bare(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await run(request, settings, "messages")
