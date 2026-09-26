from __future__ import annotations

import asyncio
import json
import os

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llms.proxy.logging import log_response, setup_logging

logger = setup_logging()

# Idle keepalive for streaming translate responses: the translate path
# buffers the full upstream body before emitting, so without interim bytes
# a slow Zen reply holds the downstream connection silent until the tunnel
# idle timeout (120s) kills it. SSE comments keep the connection alive;
# the Anthropic SDK ignores them (same pattern as providers.py:231).
STREAM_HEARTBEAT_S = float(os.getenv("STREAM_HEARTBEAT_S", "30"))
# Heartbeats only help downstream: give the upstream Zen leg room past the
# default request timeout for long generations with slow first tokens.
STREAM_TIMEOUT_S = float(os.getenv("STREAM_TIMEOUT_S", "600"))

# Sentinel for racing upstream reads against the heartbeat timer.
_END: object = object()


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

    Idle keepalive: upstream reads race a heartbeat timer that is never
    cancelled on timeout (asyncio.wait leaves the read running), so a
    silent upstream yields SSE comments instead of holding the downstream
    connection quiet past the tunnel idle timeout.
    """

    def __init__(self, upstream: httpx.Response, ingress: str, usage_sink=None):
        self._upstream = upstream
        self._ingress = ingress
        self._sink = usage_sink
        self._seen: list[str] = []
        self._buf = bytearray()

    def __aiter__(self):
        return self._gen()

    def _emit_chunk(self, chunk: bytes):
        """Reassemble lines from one raw chunk; returns complete line bytes."""
        # aiter_lines() buffers a trailing unterminated line until
        # stream close (httpx LineDecoder), so terminal usage frames
        # could miss the tap. Reassemble lines here; emit only
        # newline-terminated ones (the trailing fragment, if any, is
        # flushed as a final line at stream end).
        self._buf.extend(bytes(chunk))
        out: list[bytes] = []
        while True:
            nl = self._buf.find(b"\n")
            if nl < 0:
                break
            raw = bytes(self._buf[:nl])
            del self._buf[: nl + 1]
            line = raw.decode(errors="replace")
            if not line:
                out.append(b": ping\n\n")
                continue
            if is_cost_frame(raw):
                continue
            self._seen.append(line)
            out.append(raw + b"\n")
        return out

    def _flush_tail(self):
        if not self._buf:
            return None
        tail = bytes(self._buf)
        self._buf.clear()
        line = tail.decode(errors="replace")
        if line and not is_cost_frame(tail):
            self._seen.append(line)
            return tail + b"\n"
        return None

    async def _gen(self):
        read_task: asyncio.Task | None = None
        try:
            # Small chunks so a terminal usage frame split across TCP
            # segments still reassembles before the stream ends. (Default
            # chunking can deliver >100KB at once; the reassembly below
            # handles both intact and split deliveries.)
            it = self._upstream.aiter_bytes(chunk_size=4096)
            while True:
                if read_task is None:
                    read_task = asyncio.create_task(anext(it, _END))
                done, _ = await asyncio.wait({read_task}, timeout=STREAM_HEARTBEAT_S)
                if not done:
                    yield b": ping\n\n"
                    continue
                chunk = read_task.result()
                read_task = None
                if chunk is _END:
                    break
                for line in self._emit_chunk(chunk):
                    yield line
            tail = self._flush_tail()
            if tail is not None:
                yield tail
        finally:
            if read_task is not None and not read_task.done():
                read_task.cancel()
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


async def translate_streaming(
    upstream: httpx.Response,
    ingress: str,
    egress: str,
    trace_id: str,
    model: str,
    stream_ingress: str | None,
    usage_sink=None,
):
    """Translate upstream SSE incrementally as frames arrive (pure asyncio).

    A stateful parser/emitter pair consumes one upstream line at a time,
    so translated bytes reach the client with true streaming TTFB instead
    of buffering the whole body. Silence past the heartbeat interval
    yields SSE comments (ignored by clients). Upstream reads race the
    timer via a persistent task that is never cancelled on timeout, so
    slow frames survive intact. No threads: safe under high concurrency.

    Dialect note: the parser reads the UPSTREAM (egress) dialect and the
    emitter writes the DOWNSTREAM (ingress) dialect.
    """
    from llms.proxy.ir import StreamDone
    from llms.proxy.stream_translate import (
        STREAM_EMITTERS,
        STREAM_PARSERS,
        SseFramer,
    )

    parser = STREAM_PARSERS[egress]()
    emitter = STREAM_EMITTERS[ingress](trace_id, model)
    framer = SseFramer()
    seen_done = None

    def _run_payloads(payloads: list[str]) -> list[bytes]:
        nonlocal seen_done
        out: list[bytes] = []
        for payload in payloads:
            for delta in parser.feed_payload(payload):
                if isinstance(delta, StreamDone):
                    seen_done = delta
                out.extend(emitter.feed_delta(delta))
        return out

    it = upstream.aiter_lines()
    read_task: asyncio.Task | None = None
    try:
        while True:
            if read_task is None:
                read_task = asyncio.create_task(anext(it, _END))
            done_wait, _ = await asyncio.wait({read_task}, timeout=STREAM_HEARTBEAT_S)
            if not done_wait:
                yield b": ping\n\n"
                continue
            line = read_task.result()
            read_task = None
            if line is _END:
                break
            for chunk in _run_payloads(framer.feed(line)):
                yield chunk
        for chunk in _run_payloads(framer.finish()):
            yield chunk
        terminal = parser.finish()
        if terminal is not None:
            if seen_done is None:
                seen_done = terminal
            for chunk in emitter.feed_delta(terminal):
                yield chunk
    except BaseException:
        if read_task is not None and not read_task.done():
            read_task.cancel()
        raise
    finally:
        try:
            await upstream.aclose()
        except Exception:
            pass
        if usage_sink is not None:
            try:
                usage_sink(seen_done)
            except Exception as exc:
                logger.debug("translate-path usage sink failed: %s", exc)


async def slow_send_stream(
    send_task: asyncio.Task,
    ingress: str | None,
    egress: str | None,
    model: str,
    trace_id: str,
    stream_ingress: str | None,
    usage_sink=None,
):
    """Finish a slow upstream send while heartbeating, then stream the body.

    Entered only after the send grace (one heartbeat interval) expires, at
    which point the response has committed to 200 + SSE — upstream error
    statuses can no longer become JSONResponses, so they end the stream
    instead (fast errors never reach here; they keep the JSON path).
    """
    try:
        while not send_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(send_task), timeout=STREAM_HEARTBEAT_S
                )
            except TimeoutError:
                yield b": ping\n\n"
        try:
            upstream = send_task.result()
        except httpx.HTTPError as exc:
            logger.error("[%s] upstream connect failed: %s", trace_id, exc)
            return
    except BaseException:
        if not send_task.done():
            send_task.cancel()
        raise
    log_response(trace_id, upstream.status_code, -1)
    if upstream.status_code >= 400:
        try:
            await upstream.aread()
        finally:
            try:
                await upstream.aclose()
            except Exception:
                pass
        return
    if ingress is not None and egress is not None:
        async for chunk in translate_streaming(
            upstream, ingress, egress, trace_id, model, stream_ingress, usage_sink
        ):
            yield chunk
        return
    if stream_ingress is not None:
        tapped = await tap_stream_usage(upstream, stream_ingress, usage_sink=usage_sink)
        async for chunk in tapped:
            yield chunk
        return
    async for chunk in stream_upstream(upstream, trace_id):
        yield chunk


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
    translate_dialects: tuple[str, str, str] | None = None,
    synthesize_json=None,
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
        # Per-request upstream budget: heartbeats only keep the downstream
        # leg alive, so long Zen generations need room past the client's
        # default timeout. httpx takes this via request extensions.
        req.extensions["timeout"] = {
            "connect": 10.0,
            "read": STREAM_TIMEOUT_S,
            "write": 10.0,
            "pool": 10.0,
        }
        # Send in the background: a slow Zen TTFB must not hold the
        # downstream connection silent. Fast sends take the normal path
        # below (errors keep their JSON contract); a send slower than one
        # heartbeat commits to SSE with pings until Zen answers.
        send_task = asyncio.create_task(client.send(req, stream=True))
        try:
            await asyncio.wait_for(
                asyncio.shield(send_task), timeout=STREAM_HEARTBEAT_S
            )
        except TimeoutError:
            slow_ingress, slow_egress, slow_model = translate_dialects or (
                None,
                None,
                "",
            )
            response = StreamingResponse(
                slow_send_stream(
                    send_task,
                    slow_ingress,
                    slow_egress,
                    slow_model,
                    trace_id,
                    stream_ingress,
                    stream_usage_sink,
                ),
                media_type="text/event-stream",
            )
            if warped:
                response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
                if via_warp.get("warp_idx") is not None:
                    response.headers["x-pool-active-warp"] = str(via_warp["warp_idx"])
            return response
        except httpx.HTTPError as exc:
            logger.error("[%s] upstream connect failed: %s", trace_id, exc)
            return JSONResponse(
                status_code=502, content={"error": {"message": "upstream unreachable"}}
            )
        except BaseException:
            if not send_task.done():
                send_task.cancel()
            raise
        try:
            upstream = send_task.result()
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
        if translate_dialects is not None:
            ingress, egress, model = translate_dialects
            response = StreamingResponse(
                translate_streaming(
                    upstream,
                    ingress,
                    egress,
                    trace_id,
                    model,
                    stream_ingress,
                    stream_usage_sink,
                ),
                media_type="text/event-stream",
            )
            if warped:
                response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
                if via_warp.get("warp_idx") is not None:
                    response.headers["x-pool-active-warp"] = str(via_warp["warp_idx"])
            return response
        media = upstream.headers.get("content-type", "text/event-stream")
        if stream_ingress is not None:
            tapped = await tap_stream_usage(
                upstream, stream_ingress, usage_sink=stream_usage_sink
            )
            response = StreamingResponse(tapped, media_type=media)
            if warped:
                response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
                if via_warp.get("warp_idx") is not None:
                    response.headers["x-pool-active-warp"] = str(via_warp["warp_idx"])
            return response
        return StreamingResponse(stream_upstream(upstream, trace_id), media_type=media)
    if synthesize_json is not None:
        # Anonymous responses leg: Zen requires stream:true even when the
        # client asked for one JSON body. Stream upstream, fold the SSE
        # into a ResponseIR, and emit a single JSON document downstream.
        # (Downstream stays silent while collecting — JSON has no
        # heartbeat channel; the streaming paths above cover that case.)
        stream_body = dict(body, stream=True)
        req = client.build_request("POST", url, headers=headers, json=stream_body)
        req.extensions["timeout"] = {
            "connect": 10.0,
            "read": STREAM_TIMEOUT_S,
            "write": 10.0,
            "pool": 10.0,
        }
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
        lines = [line async for line in upstream.aiter_lines()]
        try:
            await upstream.aclose()
        except Exception:
            pass
        try:
            payload = synthesize_json(lines, trace_id)
        except Exception as exc:
            logger.error("[%s] response synthesis failed: %s", trace_id, exc)
            return JSONResponse(
                status_code=502,
                content={"error": {"message": "response synthesis failed"}},
            )
        response = JSONResponse(status_code=upstream.status_code, content=payload)
        if warped:
            response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
            if via_warp.get("warp_idx") is not None:
                response.headers["x-pool-active-warp"] = str(via_warp["warp_idx"])
        return response
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
        # request slot's position in the ready-exit spread (slot % ready —
        # not a pool pin; each slot dials its own exit).
        response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
        if via_warp.get("warp_idx") is not None:
            response.headers["x-pool-active-warp"] = str(via_warp["warp_idx"])
    return response
