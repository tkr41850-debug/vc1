from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from proxy.config import Settings, get_settings
from proxy.logging import (
    log_ingress,
    log_response,
    log_upstream,
    new_trace_id,
    setup_logging,
)
from proxy.zen_headers import build_zen_headers

router = APIRouter()
logger = setup_logging()


def upstream_path() -> str:
    return "/responses"


def apply_model_default(body: dict, settings: Settings) -> dict:
    out = dict(body)
    if not out.get("model"):
        out["model"] = settings.default_model
    return out


async def get_upstream_client(request: Request) -> AsyncIterator[httpx.AsyncClient]:
    yield request.app.state.upstream_client


def is_cost_frame(line: bytes) -> bool:
    return b"inference-cost" in line


async def stream_upstream(upstream: httpx.Response, trace_id: str):
    async for line in upstream.aiter_lines():
        if not line:
            yield b": ping\n\n"
            continue
        raw = line.encode() if isinstance(line, str) else line
        if is_cost_frame(raw):
            continue
        yield raw + b"\n"
    try:
        await upstream.aclose()
    except Exception:
        pass


@router.post("/v1/responses", response_model=None)
async def create_response_v1(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await handle_responses(request, settings)


@router.post("/responses", response_model=None)
async def create_response_bare(
    request: Request, settings: Settings = Depends(get_settings)
) -> Response:
    return await handle_responses(request, settings)


async def handle_responses(request: Request, settings: Settings) -> Response:
    trace_id = new_trace_id()
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400, content={"error": {"message": "invalid JSON body"}}
        )
    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "body must be a JSON object"}},
        )
    body = apply_model_default(body, settings)
    log_ingress(trace_id, request.url.path, body)
    headers = build_zen_headers(settings, request.headers.get("authorization"))
    url = settings.zen_base_url.rstrip("/") + upstream_path()
    log_upstream(trace_id, url, headers, body)
    client: httpx.AsyncClient = request.app.state.upstream_client
    want_stream = body.get("stream") is True
    if want_stream:
        req = client.build_request("POST", url, headers=headers, json=body)
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            logger.error("[%s] upstream connect failed: %s", trace_id, exc)
            return JSONResponse(
                status_code=502, content={"error": {"message": "upstream unreachable"}}
            )
        log_response(trace_id, upstream.status_code, -1)
        if upstream.status_code >= 400:
            try:
                payload = await upstream.aread()
            finally:
                await upstream.aclose()
            try:
                content = json.loads(payload.decode())
            except Exception:
                content = {
                    "error": {"message": payload.decode(errors="replace")[:2000]}
                }
            return JSONResponse(status_code=upstream.status_code, content=content)
        media = upstream.headers.get("content-type", "text/event-stream")
        return StreamingResponse(stream_upstream(upstream, trace_id), media_type=media)
    try:
        upstream = await client.post(url, headers=headers, json=body)
    except httpx.HTTPError as exc:
        logger.error("[%s] upstream connect failed: %s", trace_id, exc)
        return JSONResponse(
            status_code=502, content={"error": {"message": "upstream unreachable"}}
        )
    log_response(trace_id, upstream.status_code, len(upstream.content))
    try:
        payload = upstream.json()
    except Exception:
        payload = {"error": {"message": upstream.text[:2000]}}
    return JSONResponse(status_code=upstream.status_code, content=payload)
