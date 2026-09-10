from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from proxy.config import Settings, get_settings
from proxy.pipeline import run

router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_v1(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await run(request, settings, "chat")


@router.post("/chat/completions", response_model=None)
async def create_chat_bare(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await run(request, settings, "chat")
