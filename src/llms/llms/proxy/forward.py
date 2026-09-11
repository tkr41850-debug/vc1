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
    via_pool: dict | None = None,
) -> Response:
    # via_pool relays the Zen request through a vsp warp pool /fetch endpoint:
    # {"base_url", "token"}. The pool returns {ok, status, headers, body_b64};
    # streams are not supported through the relay (pools buffer /fetch).
    if via_pool is not None:
        from llms.proxy.providers import fetch_spec, parse_fetch_result

        raw = json.dumps(body).encode()
        _path, _relay_headers, spec_body = fetch_spec(
            url, headers, raw, via_pool.get("token", "")
        )
        pool_url = via_pool["base_url"].rstrip("/") + "/fetch"
        try:
            upstream = await client.post(
                pool_url,
                headers={"Content-Type": "application/json"},
                content=spec_body,
            )
        except httpx.HTTPError as exc:
            logger.error("[%s] pool connect failed: %s", trace_id, exc)
            return JSONResponse(
                status_code=502, content={"error": {"message": "pool unreachable"}}
            )
        try:
            wrapped = upstream.json()
        except Exception:
            wrapped = {"ok": False, "error": upstream.text[:500]}
        if not wrapped.get("ok"):
            status = 502
            try:
                status = int(wrapped.get("status", 502))
            except (TypeError, ValueError):
                pass
            content = {
                "error": {
                    "message": str(wrapped.get("error", "pool fetch failed"))[:2000]
                }
            }
            if status == 429 or "ratelimit" in str(wrapped.get("error", "")).lower():
                return JSONResponse(
                    status_code=429,
                    content=content,
                    headers={"retry-after": str(wrapped.get("retry_after", "60"))},
                )
            return JSONResponse(status_code=status, content=content)
        status, resp_headers, resp_body = parse_fetch_result(wrapped)
        try:
            payload = json.loads(resp_body.decode())
        except Exception:
            payload = {"error": {"message": resp_body.decode(errors="replace")[:2000]}}
        if convert is not None and status < 400:
            try:
                payload = convert(payload)
            except Exception as exc:
                logger.error("[%s] response conversion failed: %s", trace_id, exc)
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "response conversion failed"}},
                )
        response = JSONResponse(
            status_code=status,
            content=payload,
            headers=passthrough_headers(resp_headers) if status >= 400 else None,
        )
        response.headers["x-egress-provider"] = via_pool.get("provider_id", "")
        if via_pool.get("warp_idx") is not None:
            response.headers["x-egress-warp"] = str(via_pool["warp_idx"])
        return response
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
    return JSONResponse(
        status_code=upstream.status_code,
        content=payload,
        headers=passthrough_headers(upstream.headers)
        if upstream.status_code >= 400
        else None,
    )
