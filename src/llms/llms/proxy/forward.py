from __future__ import annotations

import json

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llms.proxy.logging import log_response, setup_logging

logger = setup_logging()


def is_cost_frame(line: bytes) -> bool:
    return b"inference-cost" in line


def passthrough_headers(upstream_headers) -> dict:
    out: dict = {}
    retry_after = upstream_headers.get("retry-after")
    if retry_after:
        out["retry-after"] = retry_after
    return out


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


async def parse_body(request: Request):
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
    return body


async def forward(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: dict,
    trace_id: str,
    convert=None,
    translate_stream=None,
    via_warp: dict | None = None,
) -> Response:
    # via_warp sends the Zen request through a client already bound to the
    # local warp SOCKS exit (httpx proxy=...): direct in-process egress, no
    # loopback relay. Streams flow through SOCKS like any other request.
    warped = via_warp is not None
    if body.get("stream") is True:
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
            return JSONResponse(
                status_code=upstream.status_code,
                content=content,
                headers=passthrough_headers(upstream.headers),
            )
        if translate_stream is not None:
            lines = [line async for line in upstream.aiter_lines()]
            try:
                await upstream.aclose()
            except Exception:
                pass
            return StreamingResponse(
                translate_stream(lines, trace_id), media_type="text/event-stream"
            )
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
    if convert is not None and upstream.status_code < 400:
        try:
            payload = convert(payload)
        except Exception as exc:
            logger.error("[%s] response conversion failed: %s", trace_id, exc)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "response conversion failed"}},
            )
    response = JSONResponse(
        status_code=upstream.status_code,
        content=payload,
        headers=passthrough_headers(upstream.headers)
        if upstream.status_code >= 400
        else None,
    )
    if warped:
        # Observability only: the warp provider at send time, and the
        # pool's currently-active exit snapshot (not a pin — SOCKS
        # selection is slot-spread inside the egress layer).
        response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
        if via_warp.get("pool_active_warp") is not None:
            response.headers["x-pool-active-warp"] = str(via_warp["pool_active_warp"])
    return response
