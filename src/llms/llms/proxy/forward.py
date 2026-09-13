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


def sniff_stream_usage(lines: list[str], ingress: str):
    """Run the IR stream parser over buffered lines; return the StreamDone.

    Returns None when parsing yields no terminal frame (shouldn't happen —
    parsers always emit a trailing StreamDone — but stay total).
    """
    from llms.proxy.ir import StreamDone
    from llms.proxy.stream_translate import PARSERS

    done = None
    for delta in PARSERS[ingress](lines):
        if isinstance(delta, StreamDone):
            done = delta
    return done


class TappedStream:
    """Passthrough byte stream with an IR usage tap.

    An async-iterable object (not a bare generator) so per-chunk state and
    the usage sink live on the instance — immune to generator-frame teardown
    ordering (e.g. Starlette cancelling the response task while the client
    is still draining). Yields upstream bytes identically (minus cost
    frames, like stream_upstream); when exhausted, parses the seen lines
    through the IR stream parser and hands the StreamDone to usage_sink.

    Line splitting is incremental: each received chunk is appended to a
    buffer and only newline-terminated lines are emitted/parsed, so a usage
    frame split across TCP segments still reassembles.
    """

    def __init__(self, upstream: httpx.Response, ingress: str, usage_sink=None):
        self._upstream = upstream
        self._ingress = ingress
        self._sink = usage_sink
        self._seen: list[str] = []
        self._buf = bytearray()

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        try:
            # Small chunks so a terminal usage frame split across TCP
            # segments still reassembles before the stream ends. (Default
            # chunking can deliver >100KB at once; the reassembly below
            # handles both intact and split deliveries.)
            async for chunk in self._upstream.aiter_bytes(chunk_size=4096):
                # aiter_lines() buffers a trailing unterminated line until
                # stream close (httpx LineDecoder), so terminal usage frames
                # could miss the tap. Reassemble lines here; emit only
                # newline-terminated ones (the trailing fragment, if any, is
                # flushed as a final line at stream end).
                self._buf.extend(bytes(chunk))
                while True:
                    nl = self._buf.find(b"\n")
                    if nl < 0:
                        break
                    raw = bytes(self._buf[:nl])
                    del self._buf[: nl + 1]
                    line = raw.decode(errors="replace")
                    if not line:
                        yield b": ping\n\n"
                        continue
                    if is_cost_frame(raw):
                        continue
                    self._seen.append(line)
                    yield raw + b"\n"
            if self._buf:
                tail = bytes(self._buf)
                self._buf.clear()
                line = tail.decode(errors="replace")
                if line and not is_cost_frame(tail):
                    self._seen.append(line)
                    yield tail + b"\n"
        finally:
            try:
                await self._upstream.aclose()
            except Exception:
                pass
            if self._sink is not None:
                self._record()

    def _record(self) -> None:
        from llms.proxy.ir import StreamDone
        from llms.proxy.stream_translate import PARSERS

        done = None
        try:
            for delta in PARSERS[self._ingress](self._seen):
                if isinstance(delta, StreamDone):
                    done = delta
        except Exception as exc:
            logger.debug("stream usage sniff failed: %s", exc)
        try:
            self._sink(done)
        except Exception as exc:
            logger.debug("stream usage sink failed: %s", exc)


async def tap_stream_usage(upstream: httpx.Response, ingress: str, usage_sink=None):
    """Build a TappedStream for passthrough responses (see class docs)."""
    return TappedStream(upstream, ingress, usage_sink)


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
    stream_ingress: str | None = None,
    stream_usage_sink=None,
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
            response = StreamingResponse(
                translate_stream(lines, trace_id), media_type="text/event-stream"
            )
            # Lines are already buffered: sniff IR usage now so _record_usage
            # can attribute streamed tokens without waiting on the client.
            if stream_ingress is not None:
                try:
                    response.stream_usage = sniff_stream_usage(lines, stream_ingress)
                except Exception as exc:
                    logger.debug(
                        "[%s] translate-path usage sniff failed: %s", trace_id, exc
                    )
            if warped:
                response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
                if via_warp.get("pool_active_warp") is not None:
                    response.headers["x-pool-active-warp"] = str(
                        via_warp["pool_active_warp"]
                    )
            return response
        media = upstream.headers.get("content-type", "text/event-stream")
        if stream_ingress is not None:
            tapped = await tap_stream_usage(
                upstream, stream_ingress, usage_sink=stream_usage_sink
            )
            response = StreamingResponse(tapped, media_type=media)
            if warped:
                response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
                if via_warp.get("pool_active_warp") is not None:
                    response.headers["x-pool-active-warp"] = str(
                        via_warp["pool_active_warp"]
                    )
            return response
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
