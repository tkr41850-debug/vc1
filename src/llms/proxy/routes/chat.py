from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from proxy.config import Settings, get_settings
from proxy.forward import forward, parse_body
from proxy.logging import log_ingress, log_upstream, new_trace_id, setup_logging
from proxy.zen_headers import build_zen_headers
from proxy.zen_request import build_zen_chat_request

router = APIRouter()
logger = setup_logging()


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_v1(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await handle_chat(request, settings)


@router.post("/chat/completions", response_model=None)
async def create_chat_bare(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await handle_chat(request, settings)


async def handle_chat(request: Request, settings: Settings) -> Response:
    trace_id = new_trace_id()
    body = await parse_body(request)
    if isinstance(body, JSONResponse):
        return body
    body = build_zen_chat_request(body, settings)
    log_ingress(trace_id, request.url.path, body)
    headers = build_zen_headers(settings, request.headers.get("authorization"))
    url = settings.zen_base_url.rstrip("/") + "/chat/completions"
    log_upstream(trace_id, url, headers, body)
    client = request.app.state.upstream_client
    return await forward(client, url, headers, body, trace_id)
