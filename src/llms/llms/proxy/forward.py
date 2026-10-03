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


def _slow_error_frame(status: int, message: str) -> bytes:
    """Terminal SSE error event for the slow-send path.

    The downstream response already committed to 200 + text/event-stream,
    so the status cannot travel as HTTP. Emit it as a data frame the
    client can treat as failure instead of ending the stream silently
    (an empty 200 reads as success to SDKs and records as one).
    """
    return (
        b"data: "
        + json.dumps(
            {"type": "error", "error": {"status": status, "message": message}}
        ).encode()
        + b"\n\n"
    )


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


def fold_stream_calls(lines: list[str], ingress: str) -> list[dict]:
    """Model function calls in one folded SSE body, with wire arguments.

    Runs the incremental IR parser over buffered lines and collects one
    entry per announced call: {call_id, name, arguments}. Used by the
    streaming steer fold to detect calls the client never declared —
    same lowered-name match semantics as the synthesize path.
    """
    from llms.proxy.ir import StreamDone, ToolArgsDelta
    from llms.proxy.stream_translate import STREAM_PARSERS, SseFramer

    parser = STREAM_PARSERS[ingress]()
    framer = SseFramer()
    names: dict[str, str] = {}
    args: dict[str, str] = {}
    for line in lines:
        for payload in framer.feed(line):
            for delta in parser.feed_payload(payload):
                if isinstance(delta, ToolArgsDelta):
                    name = parser.names.get(delta.call_id, delta.name)
                    if delta.call_id not in names:
                        names[delta.call_id] = name
                        args[delta.call_id] = ""
                    elif not names[delta.call_id] and name:
                        names[delta.call_id] = name
                    args[delta.call_id] += delta.args_chunk
    for payload in framer.finish():
        for delta in parser.feed_payload(payload):
            if isinstance(delta, ToolArgsDelta):
                name = parser.names.get(delta.call_id, delta.name)
                if delta.call_id not in names:
                    names[delta.call_id] = name
                    args[delta.call_id] = ""
                args[delta.call_id] += delta.args_chunk
    done = parser.finish()
    if isinstance(done, StreamDone) and getattr(done, "has_tool_calls", False):
        for call_id in list(names):
            if call_id not in args:
                args[call_id] = ""
    flush = getattr(parser, "flush_pending_calls", None)
    if callable(flush):
        try:
            for delta in flush():
                if isinstance(delta, ToolArgsDelta) and delta.call_id not in names:
                    names[delta.call_id] = delta.name
                    args[delta.call_id] = delta.args_chunk
        except Exception:
            pass
    return [
        {"call_id": cid, "name": names[cid], "arguments": args.get(cid, "")}
        for cid in names
    ]


async def fold_and_steer_streaming(
    upstream: httpx.Response,
    *,
    client,
    url: str,
    headers: dict,
    outbound: dict,
    trace_id: str,
    client_names: set[str] | None,
    usage_sink=None,
    client_tools: tuple = (),
):
    """Fold one streaming turn, steer undeclared calls, replay clean SSE.

    True-steer twin of the synthesize path for streaming downstream:
    the first upstream body buffers to lines (heartbeat pings flow
    while collecting, bounded by the STREAM_TIMEOUT_S collect budget);
    the fold detects model calls naming tools the client never declared
    (case-insensitive, declared passes); on hit the redirect +
    re-request loop runs upstream (same contract as _steer_genuine_calls:
    STEER_MAX_ITERS, final-turn-only usage); only the final clean turn
    replays downstream — as its own verbatim bytes (never re-framed
    through an emitter), so the client sees exactly what it would have
    seen had the model behaved: same response id, same call ids, same
    frame fields. A turn with mixed declared + undeclared calls is
    replaced wholesale (synthesize-path semantics): the declared frames
    never replay, but the dead turn stays upstream as input history and
    the model re-issues whatever declared call it still needs in the
    clean turn (observed live). Without undeclared calls (or without
    client_names) the first turn's bytes replay untouched: zero behavior
    change for the 99% case. Steer failure (re-request error or
    exhausted budget) fails closed with a terminal SSE error frame —
    never the violating turn. Usage attributes the final turn only;
    the replayed turn's upstream response id chains the session like
    TappedStream. Replay framing matches TappedStream (empty lines as
    ping comments, cost frames dropped) so folded and passthrough legs
    look identical on the wire modulo timing (TTFB becomes full-turn
    latency) and line-ending normalization.
    """

    class _FoldOverflow(Exception):
        """First-turn collect past the budget; carries the buffered prefix.

        Raised (never returned) so the prefix travels on the exception,
        not shared mutable state — concurrent folds each carry their own.
        Carries the still-pending line read (`pending_read`): the upstream
        body stays open and exactly one consumer ever pulls the iterator
        (concurrent anext() on one async generator raises), so the
        fallback awaits the in-flight read, then keeps pulling the same
        iterator for the remainder.
        """

        def __init__(self, prefix: list[str], iterator, pending_read):
            super().__init__("streaming fold budget exhausted")
            self.prefix = prefix
            self.iterator = iterator
            self.pending_read = pending_read

    async def _collect(
        resp: httpx.Response, deadline: float | None = None
    ) -> list[str]:
        """Buffer one upstream body to lines; raises _FoldOverflow past budget.

        Heartbeat pings flow while collecting. When `deadline`
        (loop-time seconds) passes before the body completes, the live
        line iterator, the still-pending read, and the consumed prefix
        raise to the caller: the first-turn fallback replays the prefix
        verbatim then keeps the tap draining the same iterator (a steer
        re-request, whose verdict is unknown, fails closed instead).
        """
        prefix: list[str] = []
        it = resp.aiter_lines()
        read_task: asyncio.Task | None = None
        try:
            while True:
                if read_task is None:
                    read_task = asyncio.create_task(anext(it, _END))
                done_wait, _ = await asyncio.wait(
                    {read_task}, timeout=STREAM_HEARTBEAT_S
                )
                if not done_wait:
                    prefix.append(": ping")
                    if deadline is not None:
                        try:
                            now = asyncio.get_running_loop().time()
                        except RuntimeError:
                            now = deadline
                        if now >= deadline:
                            raise _FoldOverflow(prefix, it, read_task)
                    continue
                line = read_task.result()
                read_task = None
                if line is _END:
                    break
                if not line:
                    # Tap-identical framing: blank separators replay as
                    # ping comments (see TappedStream._emit_chunk).
                    prefix.append(": ping")
                    continue
                raw = line.encode() if isinstance(line, str) else line
                if is_cost_frame(raw):
                    continue
                prefix.append(line)
        except _FoldOverflow:
            # Handing the live iterator to the fallback: its pending read
            # stays un-cancelled (cancelling it would strand the generator
            # mid-yield). Any other exception cancels the read above.
            raise
        except BaseException:
            if read_task is not None and not read_task.done():
                read_task.cancel()
            raise
        else:
            try:
                await resp.aclose()
            except Exception:
                pass
        return prefix

    def _replay(lines: list[str]) -> list[bytes]:
        # Verbatim replay of the final turn's own bytes: no emitter
        # re-framing (a fresh emitter drops wire fields such as
        # function_call.call_id/status and mints a new response id,
        # which codex needs to dispatch the call and chain the turn —
        # live probe finding). Tap-identical framing per line.
        out: list[bytes] = []
        for line in lines:
            if line == ": ping":
                out.append(b": ping\n\n")
                continue
            raw = line.encode() if isinstance(line, str) else line
            out.append(raw + b"\n")
        return out

    try:
        collect_deadline: float | None = (
            asyncio.get_running_loop().time() + STREAM_TIMEOUT_S
        )
    except RuntimeError:
        collect_deadline = None
    try:
        lines = await _collect(upstream, collect_deadline)
    except _FoldOverflow as over:
        # First-turn overflow: verdict unknown, so no steer decision is
        # possible — replay the buffered prefix verbatim, then keep the
        # tap draining the SAME live iterator (plus its pending read) for
        # the remainder. The tap's usage record covers prefix +
        # remainder, so nothing is double-billed or unbilled.
        logger.warning(
            "[%s] streaming fold budget exhausted; passing through",
            trace_id,
        )
        tapped = await tap_stream_usage(
            upstream,
            "responses",
            usage_sink=usage_sink,
            seen_prefix=over.prefix,
            line_iter=over.iterator,
            pending_read=over.pending_read,
        )
        for chunk in _replay(over.prefix):
            yield chunk
        async for chunk in tapped:
            yield chunk
        return
    lowered = (
        {n.lower() for n in client_names if isinstance(n, str)}
        if client_names is not None
        else None
    )
    max_iters = 3
    try:
        from llms.proxy.pipeline import STEER_MAX_ITERS as _MAX

        max_iters = int(_MAX)
    except Exception:
        pass
    _steer_failed = False
    from llms.proxy.client_tools import owned_tool_names as _owned_names
    from llms.proxy.pipeline import _classify_calls as _classify
    from llms.proxy.pipeline import _valid_json as _valid_args

    _owned = _owned_names(client_tools)
    _defs = {t.name.lower(): t for t in client_tools if t.name}
    for _ in range(max_iters):
        calls = fold_stream_calls(lines, "responses") if lowered is not None else []
        # Shared classifier: owned-valid calls pass (converted casing),
        # owned-invalid and undeclared calls steer. Without client_tools
        # the legacy lowered-membership rule applies.
        if client_tools:
            passthrough, steer = _classify(calls, _owned, _defs)
        else:
            passthrough, steer = (
                [],
                [
                    c
                    for c in calls
                    if not (
                        isinstance(c.get("name"), str)
                        and c["name"].lower() in (lowered or set())
                    )
                ],
            )
        if passthrough and not steer:
            # Every call is client-owned and valid: rewrite the replayed
            # lines' function_call name fields to the declared casing so
            # the client dispatches its own declarations. Only data lines
            # parse as JSON (pings/blank separators skip); frame bytes stay
            # otherwise verbatim.
            import json as _json

            for call in passthrough:
                target = call["name"]
                for i, line in enumerate(lines):
                    # Rename only the function_call name field for this
                    # call's frames (matched by call id in the same line).
                    # Same frame grammar as SseFramer: "data:" with or
                    # without the space.
                    cid = call.get("call_id", "")
                    if cid and cid in line and f'"name":"{target}"' not in line:
                        if not line.startswith("data:"):
                            continue
                        stripped = line[len("data:") :].strip()
                        try:
                            payload = _json.loads(stripped)
                        except Exception as exc:
                            logger.debug(
                                "[%s] fold rename skipped non-JSON line: %r",
                                trace_id,
                                exc,
                            )
                            continue
                        item = payload.get("item", {})
                        if (item.get("id") == cid or item.get("call_id") == cid) and (
                            isinstance(item.get("name"), str)
                            and item["name"].lower() == target.lower()
                        ):
                            item["name"] = target
                            lines[i] = "data: " + _json.dumps(payload)
            break
        if not steer or client_names is None:
            break
        names = sorted({str(c.get("name", "")) for c in steer if c.get("name")})
        logger.info(
            "[%s] steering streaming tool call(s) %s back to client tools",
            trace_id,
            names,
        )
        followups: list = []
        for call in steer:
            call_id = str(call.get("call_id") or call.get("id") or "")
            args = call.get("arguments", "")
            args = args if isinstance(args, str) else ""
            followups.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": str(call.get("name", "")),
                    # Steer followups replay upstream: never leak a raw ""
                    # or non-JSON payload that the validator 400s.
                    "arguments": args if _valid_args(args) else "{}",
                }
            )
            followups.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": (
                        f"Tool '{call.get('name', '')}' is not available in "
                        f"this session."
                        + (
                            f" Use one of these tools instead: "
                            f"{', '.join(sorted(client_names))}."
                            if client_names
                            else ""
                        )
                        + " If none fits, answer directly without calling a tool."
                    ),
                }
            )
        outbound = dict(outbound, input=list(outbound.get("input", [])) + followups)
        req = client.build_request("POST", url, headers=headers, json=outbound)
        req.extensions["timeout"] = {
            "connect": 10.0,
            "read": STREAM_TIMEOUT_S,
            "write": 10.0,
            "pool": 10.0,
        }
        try:
            follow_resp = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            logger.error("[%s] steer re-request failed: %s", trace_id, exc)
            # Fail closed: never replay a turn known to carry undeclared
            # calls. Emit a terminal error frame (slow_send_stream
            # precedent) while the dead turn stays upstream as history.
            lines = []
            _steer_failed = True
            break
        if follow_resp.status_code >= 400:
            logger.warning(
                "[%s] steer re-request upstream status=%s; failing closed",
                trace_id,
                follow_resp.status_code,
            )
            try:
                await follow_resp.aclose()
            except Exception:
                pass
            lines = []
            _steer_failed = True
            break
        if collect_deadline is not None:
            try:
                over = asyncio.get_running_loop().time() >= collect_deadline
            except RuntimeError:
                over = False
            if over:
                logger.warning(
                    "[%s] streaming steer budget exhausted; failing closed",
                    trace_id,
                )
                try:
                    await follow_resp.aclose()
                except Exception:
                    pass
                lines = []
                _steer_failed = True
                break
        try:
            lines = await _collect(follow_resp, collect_deadline)
        except _FoldOverflow:
            # Re-request overflow: clean verdict unknown — fail closed
            # rather than emit an unjudged turn. The unconsumed body
            # holds its pooled connection, so close it (same guard as the
            # sibling branches) — _collect deliberately left it open for
            # the first-turn handover, which does not apply here.
            logger.warning(
                "[%s] streaming steer re-request overflow; failing closed",
                trace_id,
            )
            try:
                await follow_resp.aclose()
            except Exception:
                pass
            lines = []
            _steer_failed = True
            break
    if _steer_failed:
        chunks = [_slow_error_frame(502, "steer re-request failed")]
    else:
        chunks = _replay(lines)
    if usage_sink is not None:
        try:
            from dataclasses import replace as _replace

            from llms.proxy.ir import StreamDone as _StreamDone

            done = sniff_stream_usage(lines, "responses")
            if _steer_failed and isinstance(done, _StreamDone):
                done = _replace(done, status="incomplete")
            elif isinstance(done, _StreamDone) and done.response_id is None:
                rid = _stream_response_id("responses", lines)
                if rid:
                    done = _replace(done, response_id=rid)
            usage_sink(done)
        except Exception as exc:
            logger.debug("fold-path usage sink failed: %s", exc)
    for chunk in chunks:
        yield chunk


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

    def __init__(
        self,
        upstream: httpx.Response,
        ingress: str,
        usage_sink=None,
        seen_prefix: list[str] | None = None,
        line_iter=None,
        pending_read=None,
    ):
        self._upstream = upstream
        self._ingress = ingress
        self._sink = usage_sink
        self._seen: list[str] = list(seen_prefix) if seen_prefix else []
        # Fold-overflow fallback hands over its live line iterator plus
        # the still-pending read, so the remainder forwards from the same
        # stream (no re-iteration) with exactly one iterator consumer.
        self._line_iter = line_iter
        self._pending_read = pending_read
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
            if self._line_iter is not None:
                # Fold-overflow fallback: the remainder forwards from the
                # same live line iterator the fold was consuming (httpx
                # forbids re-iterating the body). The fold's still-pending
                # read resolves first — exactly one consumer ever pulls
                # the iterator (concurrent anext() raises). Tap-identical
                # framing per line; the usage record covers prefix +
                # remainder.
                it = self._line_iter
                read_task = self._pending_read
                self._pending_read = None
                while True:
                    if read_task is None:
                        read_task = asyncio.create_task(anext(it, _END))
                    done, _ = await asyncio.wait(
                        {read_task}, timeout=STREAM_HEARTBEAT_S
                    )
                    if not done:
                        yield b": ping\n\n"
                        self._seen.append(": ping")
                        continue
                    line = read_task.result()
                    read_task = None
                    if line is _END:
                        break
                    if not line:
                        yield b": ping\n\n"
                        self._seen.append(": ping")
                        continue
                    raw = line.encode() if isinstance(line, str) else line
                    if is_cost_frame(raw):
                        continue
                    self._seen.append(line)
                    yield raw + b"\n"
                return
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
        from dataclasses import replace

        from llms.proxy.ir import StreamDone
        from llms.proxy.stream_translate import PARSERS

        done = None
        try:
            for delta in PARSERS[self._ingress](self._seen):
                if isinstance(delta, StreamDone):
                    done = delta
        except Exception as exc:
            logger.debug("stream usage sniff failed: %s", exc)
        if done is not None:
            try:
                rid = _stream_response_id(self._ingress, self._seen)
            except Exception as exc:
                logger.debug("stream id sniff failed: %s", exc)
                rid = None
            if rid:
                done = replace(done, response_id=rid)
        try:
            self._sink(done)
        except Exception as exc:
            logger.debug("stream usage sink failed: %s", exc)


async def tap_stream_usage(
    upstream: httpx.Response,
    ingress: str,
    usage_sink=None,
    seen_prefix: list[str] | None = None,
    line_iter=None,
    pending_read=None,
):
    """Build a TappedStream for passthrough responses (see class docs)."""
    return TappedStream(
        upstream,
        ingress,
        usage_sink,
        seen_prefix=seen_prefix,
        line_iter=line_iter,
        pending_read=pending_read,
    )


def _stream_response_id(ingress: str, seen: list[str]) -> str | None:
    """Upstream response id sniffed from streamed SSE lines.

    Same-dialect legs pass upstream ids straight to the client, which
    echoes them back as previous_response_id; capturing the id lets the
    follow-up turn reuse this session so the prompt cache stays warm.
    Best-effort: returns None when no id frame is seen.
    """
    import json as _json

    for line in seen:
        text = line.strip()
        if text.startswith("data:"):
            text = text[len("data:") :].strip()
        if not text.startswith("{"):
            continue
        try:
            payload = _json.loads(text)
        except Exception as exc:
            logger.debug("stream id sniff skipped non-JSON line: %r", exc)
            continue
        if not isinstance(payload, dict):
            continue
        if ingress == "responses":
            resp = payload.get("response")
            if isinstance(resp, dict) and isinstance(resp.get("id"), str):
                return resp["id"]
        elif ingress == "chat":
            if isinstance(payload.get("id"), str):
                return payload["id"]
        elif ingress == "messages":
            msg = payload.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("id"), str):
                return msg["id"]
    return None


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
    statuses can no longer become JSONResponses, so they surface as a
    terminal SSE error event instead (fast errors never reach here; they
    keep the JSON path). A bare empty 200 would read as success to SDKs.
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
            yield _slow_error_frame(502, "upstream unreachable")
            return
    except BaseException:
        if not send_task.done():
            send_task.cancel()
        raise
    log_response(trace_id, upstream.status_code, -1)
    if upstream.status_code >= 400:
        try:
            payload = await upstream.aread()
        finally:
            try:
                await upstream.aclose()
            except Exception:
                pass
        try:
            body = json.loads(payload.decode())
            message = (
                str(
                    body.get("error", {}).get("message", "")
                    if isinstance(body, dict)
                    else ""
                )
                or payload.decode(errors="replace")[:500]
            )
        except Exception:
            message = payload.decode(errors="replace")[:500]
        yield _slow_error_frame(upstream.status_code, message or "upstream error")
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
    client_names: set[str] | None = None,
    steer_streaming: bool = False,
    client_tool_defs: tuple = (),
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
            if steer_streaming and ingress == "responses" and egress == "responses":
                # Streaming steer fold (refold design): buffer turn 1,
                # steer undeclared calls via redirect + re-request,
                # replay only the clean turn. Same response id, no
                # dead-call bytes downstream. Falls back to plain
                # translate_streaming without client_names.
                response = StreamingResponse(
                    fold_and_steer_streaming(
                        upstream,
                        client=client,
                        url=url,
                        headers=headers,
                        outbound=dict(body),
                        trace_id=trace_id,
                        client_names=client_names,
                        usage_sink=stream_usage_sink,
                        client_tools=client_tool_defs,
                    ),
                    media_type="text/event-stream",
                )
            else:
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
        if steer_streaming and stream_ingress == "responses":
            # Same-dialect responses streaming (codex path): fold +
            # steer instead of raw tap passthrough. stream_ingress
            # doubles as the fold dialect here.
            response = StreamingResponse(
                fold_and_steer_streaming(
                    upstream,
                    client=client,
                    url=url,
                    headers=headers,
                    outbound=dict(body),
                    trace_id=trace_id,
                    client_names=client_names,
                    usage_sink=stream_usage_sink,
                    client_tools=client_tool_defs,
                ),
                media_type=media,
            )
            if warped:
                response.headers["x-egress-provider"] = via_warp.get("provider_id", "")
                if via_warp.get("warp_idx") is not None:
                    response.headers["x-pool-active-warp"] = str(via_warp["warp_idx"])
            return response
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
