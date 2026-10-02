from __future__ import annotations

import asyncio
import json
import math
import os
import time

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llms.proxy import dedup, sessions
from llms.proxy.admin_hub import get_hub
from llms.proxy.affinity import bucket_for
from llms.proxy.config import Settings
from llms.proxy.forward import forward, parse_body
from llms.proxy.ir import RequestIR
from llms.proxy.logging import log_ingress, log_upstream, new_trace_id, setup_logging
from llms.proxy.providers import RecentRequest
from llms.proxy.rate_limit import classify
from llms.proxy.router import ENDPOINT_PATH, pick, resolve_alias
from llms.proxy.stream_translate import (
    SseFramer,
    new_chat_id,
    new_msg_id,
    new_resp_id,
)
from llms.proxy.translate import (
    from_chat,
    from_messages,
    from_responses,
    to_zen_chat,
    to_zen_messages,
    to_zen_responses,
    with_model,
)
from llms.proxy.translate_response import (
    EMITTERS,
    convert_response,
    deltas_to_response_ir,
)
from llms.proxy.zen_fingerprint import note_free_tier_error
from llms.proxy.zen_headers import build_zen_headers, stable_session_id
from llms.proxy.zen_prompts import TITLE_PREFIX
from llms.proxy.zen_tools import GENUINE_TOOLS

logger = setup_logging()

FROM = {"responses": from_responses, "chat": from_chat, "messages": from_messages}
TO = {"responses": to_zen_responses, "chat": to_zen_chat, "messages": to_zen_messages}
DEFAULT_MODEL_ATTR = {
    "responses": "default_model",
    "chat": "default_chat_model",
    "messages": "default_messages_model",
}


def _convert_for(ingress: str, egress: str, model: str):
    if ingress == egress:
        return None
    return lambda payload: convert_response(egress, ingress, payload, model)


def _stream_for(ingress: str, egress: str, model: str):
    if ingress == egress:
        return None
    if (ingress, egress) not in {
        ("chat", "responses"),
        ("responses", "chat"),
        ("messages", "responses"),
        ("responses", "messages"),
        ("chat", "messages"),
        ("messages", "chat"),
    }:
        return None
    return (ingress, egress, model)


def _deltas_with_pending(parser, lines):
    """Yield parser deltas plus pending-call flush (fold path only).

    Lives here (not in the parser module) so the incremental emitter
    path can never call it by accident: only _synthesize_for drains
    pending calls, after the whole stream parsed.
    """
    framer = SseFramer()
    for line in lines:
        for payload in framer.feed(line):
            yield from parser.feed_payload(payload)
    for payload in framer.finish():
        yield from parser.feed_payload(payload)
    flush = getattr(parser, "flush_pending_calls", None)
    if callable(flush):
        yield from flush()
    done = parser.finish()
    if done is not None:
        yield done


def _synthesize_for(ingress: str, egress: str, model: str):
    """Fold upstream SSE into one downstream JSON body (responses leg only).

    Anonymous Zen requires stream:true even for single-shot callers; the
    deltas accumulate into a ResponseIR that the normal response emitters
    render in the ingress dialect.
    """
    if egress != "responses":
        return None

    def run(lines, trace_id):
        from llms.proxy.stream_translate import STREAM_PARSERS

        parser = STREAM_PARSERS[egress]()
        deltas = list(_deltas_with_pending(parser, lines))
        return EMITTERS[ingress](deltas_to_response_ir(iter(deltas), model), model)

    return run


STEER_MAX_ITERS = int(os.getenv("ZEN_STEER_MAX_ITERS", "3"))


def _inflight_track(request, provider_id: str | None) -> None:
    """Increment the provider's in-flight count (§3, event loop only)."""
    if not provider_id:
        return
    try:
        registry = getattr(request.app.state, "providers", None)
        if registry is not None:
            registry.runtime(provider_id).in_flight += 1
    except Exception:
        pass


def _inflight_release(request, provider_id: str | None) -> None:
    """Decrement with a max(0, …) guard; schedules a throttled push."""
    if not provider_id:
        return
    try:
        registry = getattr(request.app.state, "providers", None)
        if registry is None:
            return
        rt = registry.runtime(provider_id)
        rt.in_flight = max(0, rt.in_flight - 1)
        hub = getattr(request.app.state, "admin_hub", None)
        publish = getattr(hub, "publish_throttled", None)
        if callable(publish):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(publish("providers"))
    except Exception:
        pass


def _wrap_inflight(
    request,
    response,
    provider_id: str | None,
    *,
    model: str | None = None,
    warp_idx: int | None = None,
    started: float | None = None,
):
    """Wrap a StreamingResponse so body completion releases in-flight.

    Covers all four stream generators in forward.py at once; `finally`
    covers abrupt disconnect. JSON/synthesize legs decrement via
    _inflight_release after _record_usage instead.

    A fully-consumed stream also records the recent-request ring entry
    (model/warp_idx/started supplied by the call site): without this only
    JSON legs ever appear in Last requests (live), so a successful
    streaming probe promotes probation -> ready yet stays invisible.
    Disconnects record nothing — the request never completed.
    """
    if provider_id is None or not isinstance(response, StreamingResponse):
        return response
    iterator = response.body_iterator

    async def _tracking_iterator():
        completed = False
        try:
            async for chunk in iterator:
                yield chunk
            completed = True
        finally:
            _inflight_release(request, provider_id)
            # Stream probes settle at body exhaustion: a fully-consumed
            # probe promotes probation -> ready (the outcome-429 path ran
            # at headers time, so settle only needs the success arm here).
            # Disconnects (CancelledError / early close) stay armed — the
            # next request re-probes instead of riding an unverified exit.
            if completed:
                try:
                    registry = getattr(request.app.state, "providers", None)
                    if registry is not None:
                        _settle_probation(request, registry, provider_id, response)
                        if model is not None:
                            outcome = getattr(
                                getattr(request, "state", None),
                                "stream_outcome",
                                None,
                            )
                            error = (
                                ""
                                if outcome in (None, "completed")
                                else f"stream {outcome}"
                            )
                            elapsed_ms = (
                                (time.monotonic() - started) * 1000.0
                                if started is not None
                                else 0.0
                            )
                            await registry.runtime(provider_id).record(
                                RecentRequest(
                                    ts=time.time(),
                                    model=model,
                                    status=(
                                        response.status_code
                                        if hasattr(response, "status_code")
                                        else 0
                                    ),
                                    ms=elapsed_ms,
                                    warp_idx=warp_idx,
                                    error=str(error)[:200],
                                )
                            )
                except Exception:
                    pass

    response.body_iterator = _tracking_iterator()
    return response


def _genuine_calls_in(response: Response, client_names: set[str]) -> list[dict]:
    """Assistant function_calls the client did NOT declare (steer candidates).

    Matching is case-insensitive against the client's own set: a call
    naming a client-declared tool — even one colliding with a genuine
    tool name in a different case (client "Read" vs genuine "read") —
    is the client's to resolve and is never steered. Only undeclared
    names (undeclared genuine tools, hallucinations) steer.
    """
    if not isinstance(response, JSONResponse) or response.status_code >= 400:
        return []
    try:
        payload = json.loads(response.body.decode())
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    calls = []
    lowered = {n.lower() for n in client_names if isinstance(n, str)}
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        name = item.get("name")
        if isinstance(name, str) and name.lower() in lowered:
            continue
        calls.append(item)
    return calls


async def _steer_genuine_calls(
    response: Response,
    *,
    client,
    url: str,
    headers: dict,
    outbound: dict,
    synthesize,
    trace_id: str,
    client_names: set[str],
) -> tuple[Response, dict]:
    """Answer undeclared tool calls with a redirect error and re-request.

    Only calls naming tools the client did NOT declare steer (undeclared
    genuine tools get a redirect listing client tools; hallucinations
    with no client tools get an answer-directly nudge). A call naming a
    client-declared tool — even one colliding with a genuine tool name
    in a different case — passes straight back for the client to
    resolve. Only for
    non-streaming downstream (streaming passes calls through —
    mid-stream steering is a follow-up). Bounded; usage attributes the
    final turn only.
    """
    for _ in range(STEER_MAX_ITERS):
        calls = _genuine_calls_in(response, client_names)
        if not calls:
            break
        names = sorted({str(c.get("name", "")) for c in calls})
        logger.info(
            "[%s] steering genuine tool call(s) %s back to client tools",
            trace_id,
            names,
        )
        followups: list = []
        for call in calls:
            call_id = str(call.get("call_id") or call.get("id") or "")
            followups.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": str(call.get("name", "")),
                    "arguments": str(call.get("arguments", "")),
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
        response = await forward(
            client,
            url,
            headers,
            outbound,
            trace_id,
            synthesize_json=synthesize,
        )
    return response, outbound


def is_genuine_opencode(headers) -> bool:
    """True when the downstream client identifies as genuine opencode.

    Genuine opencode already carries the exact wire identity Zen's gate
    wants (canonical instructions + genuine tool set), so the anonymous
    shaping below (tool injection, chat sysprompt prefix) must not
    mangle it — passthrough instead. Other clients (Claude Code, Codex,
    DSH) send no opencode product headers and keep the shaping.
    """
    ua = headers.get("user-agent", "")
    return ua.startswith("opencode/") or bool(headers.get("x-opencode-client"))


def _with_genuine_tools(outbound: dict, client_names: set[str] | None = None) -> None:
    """Prepend the genuine tool set ahead of client extras (no duplicates).

    The free-tier gate fuzzy-matches the set: the 12 genuine definitions
    must go out byte-identical, in order, ahead of any extras — bare or
    renamed sets 403. A client tool reusing a genuine name in the SAME
    case keeps the CLIENT's definition in that slot (so the model calls
    the client's shape and the call passes back for the client to
    resolve); case-variant collisions ("Read" vs genuine "read") keep
    the genuine definition untouched (renaming it breaks the gate) and
    the client tool appends after as an extra. All 12 genuine names stay
    present at least once, always with their genuine definition.
    Mutates outbound in place.

    Scoped overlay: when the caller supplies the client's own tool names,
    a genuine tool the client ALSO declares under the exact same name is
    skipped from the head — the client's definition already occupies the
    slot, so prepending the genuine twin only dangles an unexecutable
    same-name double in front of the model (live codex finding: model
    called overlay 'shell' instead of declared 'exec_command', and codex
    failed the turn with 'unsupported call: shell'). Skipped names still
    satisfy the gate: the slot carries the client's definition under the
    genuine name. Case-variant collisions ('Read' vs 'read') keep the
    genuine definition untouched (renaming it breaks the gate) with the
    client tool appended after. Without client_names every genuine tool
    prepends (legacy behavior for callers that don't track declarations).
    """
    genuine_names = {t.get("name") for t in GENUINE_TOOLS}
    by_name = {t.get("name"): t for t in outbound.get("tools", []) or []}
    head = []
    for g in GENUINE_TOOLS:
        gname = g.get("name", "")
        if gname in by_name:
            # Client declares this exact name: its own definition
            # occupies the slot — skip the genuine twin so the model
            # never sees an unexecutable same-name double (live codex
            # finding: model called overlay 'shell' instead of the
            # declared tool, failing the turn 'unsupported call').
            continue
        head.append(by_name.get(gname, g))
    extras = [
        t for t in outbound.get("tools", []) or [] if t.get("name") not in genuine_names
    ]
    # Declared-name client tools whose name collides with a genuine
    # tool ride in their overlay slot position, ahead of extras — the
    # slot keeps the client's definition.
    slots = [by_name[gname] for gname in genuine_names if gname in by_name]
    outbound["tools"] = [*head, *slots, *extras]


def _ensure_chat_system(outbound: dict) -> None:
    """Lead chat system content with the canonical prefix (anonymous only).

    The chat leg gates like responses (bisected live: canonical system
    streams 200, bare fails). Mutates outbound in place.
    """
    from llms.proxy.zen_prompts import SEAM

    messages = outbound.get("messages", [])
    original = ""
    rest = messages
    if (
        messages
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        first, rest = messages[0], messages[1:]
        content = first.get("content", "")
        if isinstance(content, str):
            original = content
        elif isinstance(content, list):
            original = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    text = TITLE_PREFIX if not original else TITLE_PREFIX + SEAM + original
    outbound["messages"] = [{"role": "system", "content": text}, *rest]


def _note_free_tier_error(
    response: Response, settings: Settings, trace_id: str
) -> None:
    """Background a fingerprint re-check when Zen rejects the wire identity."""
    if not isinstance(response, JSONResponse):
        return
    try:
        payload = json.loads(response.body.decode())
    except Exception:
        return
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and payload["error"].get("type") == "FreeTierError"
    ):
        note_free_tier_error(settings.data_dir, trace_id, settings.zen_base_url)


def _outcome_of(response: Response) -> tuple[str, float | None]:
    if isinstance(response, JSONResponse):
        try:
            payload = json.loads(response.body.decode())
        except Exception:
            payload = None
        return classify(response.status_code, payload, dict(response.headers))
    return "ok", None


def _reserved_session(request, ingress: str, model: str, body: dict) -> str | None:
    """Session id reserved by an earlier identical fresh attempt, if any."""
    table = getattr(request.app.state, "reservations", None)
    if table is None:
        return None
    return table.lookup(
        dedup.reserve_key(
            getattr(request.state, "secret_key", None),
            getattr(request.state, "affinity", None),
            ingress,
            model,
            body,
        )
    )


def _remember_reservation(
    request, ingress: str, model: str, body: dict, session_id: str
) -> None:
    table = getattr(request.app.state, "reservations", None)
    if table is None:
        return
    try:
        table.remember(
            dedup.reserve_key(
                getattr(request.state, "secret_key", None),
                getattr(request.state, "affinity", None),
                ingress,
                model,
                body,
            ),
            session_id,
        )
    except Exception as exc:
        logger.debug("session reservation failed: %r", exc)


def _dedup_background_finish(task, **kwargs) -> None:
    """Settle a deduped upstream task that outlived its foreground waiter.

    Stores JSON results for later claims; on timeout abandonment also runs
    the usage/conversation/warming bookkeeping the skipped tail would have
    done (fast completions leave those to the normal tail — exactly once
    either way). Best-effort: never raises.
    """
    table = kwargs["table"]
    key = kwargs["key"]
    entry = kwargs["entry"]
    try:
        if table.lookup(key) is not entry:
            try:
                task.result()
            except BaseException:
                pass
            return
        response = task.result()
    except BaseException:
        try:
            table.drop(key)
        except Exception:
            pass
        return
    try:
        from fastapi.responses import JSONResponse

        if not isinstance(response, JSONResponse):
            table.drop(key)
            return
        if not 200 <= response.status_code < 400:
            # Never hold errors: a replayed 429/5xx would mask recovery
            # (rebalance, failover, fresh retry-after). Retries rerun live.
            table.drop(key)
            return
        table.complete(
            key,
            response.status_code,
            bytes(response.body),
            dedup.DedupTable.store_headers(response.headers),
        )
        if not entry.timed_out:
            return
        request = kwargs["request"]
        try:
            _record_usage(request, kwargs["ingress"], kwargs["model"], response)
        except Exception as exc:
            logger.debug("dedup background usage failed: %r", exc)
        try:
            _record_conversation(
                response,
                kwargs["session_tracker"],
                kwargs["secret_key"],
                kwargs["session_id"],
                None,
            )
        except Exception as exc:
            logger.debug("dedup background conversation failed: %r", exc)
        try:
            _warm_new_conversation(
                response,
                kwargs["is_new_conversation"],
                kwargs["egress"],
                kwargs["client"],
                kwargs["url"],
                kwargs["headers"],
                kwargs["model"],
                kwargs["session_id"],
                kwargs["trace_id"],
            )
        except Exception as exc:
            logger.debug("dedup background warm failed: %r", exc)
    except Exception as exc:
        logger.debug("dedup background finish failed: %r", exc)


async def _publish_providers(request) -> None:
    """Best-effort immediate providers push (track-time/refetch changes).

    Release paths use throttled publish (high-frequency coalescing);
    track-time and health-change pushes bypass the throttle so the UI
    converges without polling.
    """
    try:
        hub = getattr(request.app.state, "admin_hub", None)
        publish = getattr(hub, "publish", None)
        if callable(publish):
            await publish("providers")
    except Exception:
        pass


def _record_conversation(
    response: Response,
    session_tracker,
    secret_key: str | None,
    session_id: str,
    stream_id: str | None = None,
) -> None:
    """Remember downstream response ids so chained continuations reuse this session."""
    if session_tracker is None or secret_key is None:
        return
    response_id = stream_id
    if response_id is None and isinstance(response, JSONResponse):
        if response.status_code >= 400:
            return
        try:
            payload = json.loads(response.body.decode())
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            response_id = payload["id"]
    # Passthrough streams reuse upstream ids the tap never parses, so only
    # translate legs (explicit stream_id) and JSON bodies map here.
    if response_id is not None:
        session_tracker.remember(secret_key, "chain:" + response_id, session_id)


def _warm_new_conversation(
    response: Response,
    is_new_conversation: bool,
    egress: str,
    client,
    url: str,
    headers: dict,
    model: str,
    session_id: str,
    trace_id: str,
) -> None:
    """Background a title warming call for a brand-new responses conversation."""
    if not is_new_conversation or egress != "responses" or not sessions.SESSION_WARMING:
        return
    if not isinstance(response, StreamingResponse):
        if not isinstance(response, JSONResponse) or response.status_code >= 400:
            return
    elif getattr(response, "status_code", 200) >= 400:
        return

    async def _run() -> None:
        try:
            await sessions.warm_session(
                client, url, headers, model, session_id, trace_id
            )
        except BaseException as exc:
            logger.debug("[%s] session warm task ended: %r", trace_id, exc)

    asyncio.create_task(_run())


async def run(request: Request, settings: Settings, ingress: str) -> Response:
    trace_id = new_trace_id()
    body = await parse_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        req: RequestIR = FROM[ingress](body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})
    if not req.model:
        req = with_model(req, getattr(settings, DEFAULT_MODEL_ATTR[ingress]))
    requested = req.model
    req = with_model(req, resolve_alias(req.model, settings.model_aliases))
    egress = pick(req.model, ingress)
    outbound = TO[egress](req)
    affinity = getattr(request.state, "affinity", None)
    secret_key = getattr(request.state, "secret_key", None)
    # Upstream session: per-conversation simulation on the responses leg
    # (previous_response_id chains, codex thread headers), stable per-key
    # fallback everywhere else. Zen's free-tier gate requires body
    # prompt_cache_key to equal the x-opencode-session header (both
    # well-formed ses_ IDs); mint one id here and share it with headers.
    # Chat has no such field/gate; messages needs a real API key instead.
    session_id = stable_session_id(settings.zen_api_key)
    is_new_conversation = False
    session_tracked = False
    session_tracker = getattr(request.app.state, "sessions", None)
    if session_tracker is not None and ingress == "responses":
        ref = sessions.conversation_ref(ingress, body, request.headers)
        if ref is None:
            # No trackable ref: reuse a session reserved by an earlier
            # identical attempt (client retry after a dedup 429) so the
            # retry matches the in-flight hash and its cache affinity.
            reserved = _reserved_session(request, ingress, req.model, body)
            if reserved is not None:
                session_id = reserved
            else:
                session_id = sessions.mint_session_id()
                _remember_reservation(request, ingress, req.model, body, session_id)
                is_new_conversation = True
        else:
            hit = session_tracker.lookup(secret_key, ref)
            if hit is not None:
                session_id = hit
                session_tracked = True
            elif not ref.startswith("chain:"):
                session_id = sessions.mint_session_id()
                session_tracker.remember(secret_key, ref, session_id)
                is_new_conversation = True
    genuine = is_genuine_opencode(request.headers)
    if egress == "responses":
        outbound["prompt_cache_key"] = session_id
        # Anonymous free tier matches the genuine tool set (bisected
        # live: full set + any client extras passes; bare/renamed
        # sets 403). Instructions pass through untouched — any
        # canonical lead steers behavior (title) or costs 9KB (agent).
        # Keyed operators keep exact fidelity. Genuine opencode
        # already carries the exact wire identity: passthrough.
        if not settings.zen_api_key and not genuine:
            _with_genuine_tools(outbound, {t.name for t in req.tools if t.name})
    elif egress == "chat" and not settings.zen_api_key and not genuine:
        _ensure_chat_system(outbound)
    bucket = bucket_for(
        affinity,
        req.model,
        settings.num_buckets,
        secret_key,
        # Tracked continuations only: a tracker hit means a real ongoing
        # conversation, so its turns pin to one bucket (cache stays warm)
        # while distinct conversations spread across slots. Fresh mints
        # stay on the stable hash — no identity exists yet to pin, and
        # this keeps untracked traffic (and its tests) deterministic.
        session_id if session_tracked else None,
    )
    table = request.app.state.bucket_table
    slot = table.slot_for(bucket)
    log_ingress(
        trace_id,
        request.url.path,
        {
            "model": req.model,
            "requested_model": requested,
            "ingress": ingress,
            "egress": egress,
            "affinity": affinity,
            "bucket": bucket,
            "slot": slot,
        },
    )
    headers = build_zen_headers(settings, session_id=session_id)
    url = settings.zen_base_url.rstrip("/") + ENDPOINT_PATH[egress]
    log_upstream(trace_id, url, headers, outbound)
    egress_provider = request.app.state.egress
    started = time.monotonic()
    provider_id: str | None = None
    kind: str = "noproxy"
    warp_egress = None
    via_warp: dict | None = None
    warp_idx: int | None = None
    registry = getattr(request.app.state, "providers", None)
    resolve = getattr(egress_provider, "resolve", None)
    if callable(resolve):
        provider_id, kind, warp_egress = resolve(req.model, bucket)
        if kind == "queued":
            # Main-queue hold: warp capacity exists for this model but
            # every candidate is an at-quota probe. Hold the connection
            # open and re-resolve until a probe releases (falls through
            # to the warp block below) or the budget lapses (429).
            # Keepalive is the open connection itself: wakeups every
            # queue_keepalive_s bound the silent gap. Waiters are never
            # in-flight tracked (track sits below), so no release is
            # owed on any exit from this loop. A disconnected client
            # aborts the hold at once (nothing tracked, just return) —
            # otherwise long holds forward upstream to a ghost behind
            # idle timeouts (Cloudflare 120s per spec §5).
            _waited = 0.0
            # No `or` defaults here: an explicit 0.0 budget must degrade
            # immediately (0.0 or 600.0 would read as 600.0 and hang the
            # waiter for the full budget). Floors only stop busy-spin
            # (step) and negative budgets.
            _step = max(0.01, float(getattr(settings, "queue_keepalive_s", 15.0)))
            _budget = max(0.0, float(getattr(settings, "queue_wait_s", 600.0)))
            while _waited < _budget and kind == "queued":
                _sleep = min(_step, _budget - _waited)
                await asyncio.sleep(_sleep)
                _waited += _sleep
                try:
                    if await request.is_disconnected():
                        return JSONResponse(
                            status_code=499,
                            content={"error": {"message": "client disconnected"}},
                        )
                except Exception:
                    pass
                provider_id, kind, warp_egress = resolve(req.model, bucket)
            if kind == "queued":
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "queue wait exceeded",
                            "type": "queue_timeout",
                        }
                    },
                    headers={"retry-after": str(int(_budget))},
                )
        if kind == "warp" and warp_egress is not None:
            provider = None
            if registry is not None:
                for p in registry.load():
                    if p.id == provider_id:
                        provider = p
                        break
            if registry is not None and provider is not None:
                _before = registry.runtime(provider.id).health.fetched_at
                await registry.refresh_health(provider)
                if registry.runtime(provider.id).health.fetched_at != _before:
                    # The refresh actually moved health (not a TTL no-op):
                    # push so admin viewers converge without polling or a
                    # debug click (stale red-until-debug regression).
                    await _publish_providers(request)
                # Re-resolve after the refresh: a slot that flipped ready
                # during this very poll must seed the egress's SOCKS ports
                # before client_for runs, or the request fails open despite
                # a connected tunnel (live finding: final status.json showed
                # ready=true while the request went direct).
                provider_id, kind, warp_egress = resolve(req.model, bucket)
                if kind != "warp" or warp_egress is None:
                    provider_id, via_warp = None, None
                sync = getattr(egress_provider, "sync_bucket_slots", None)
                if callable(sync):
                    sync(table)
            if kind != "warp" or warp_egress is None:
                client = egress_provider.client_for(bucket, slot)
            else:
                try:
                    client = warp_egress.client_for(bucket, slot)
                except RuntimeError:
                    # No ready SOCKS exits: fail open to direct.
                    logger.info(
                        "[%s] warp provider %s has no ready exits; failing open",
                        trace_id,
                        provider_id,
                    )
                    provider_id, via_warp = None, None
                    client = egress_provider.client_for(bucket, slot)
                else:
                    # Slot-spread egress: the request's slot pins to one
                    # ready exit (slot % ready), so warp_idx is the slot's
                    # position in the ready spread — not a pool pin.
                    ports = warp_egress.ready_ports()
                    warp_idx = (
                        ports.index(warp_egress.pick_port(slot)) if ports else None
                    )
                    via_warp = {
                        "provider_id": provider_id or "",
                        "warp_idx": warp_idx,
                        "socks_port": warp_egress.pick_port(slot),
                    }
        else:
            client = egress_provider.client_for(bucket, slot)
    else:
        client = egress_provider.client_for(bucket, slot)
    # In-flight accounting starts once resolve() assigns a provider, and
    # the count change pushes immediately: the per-request refresh below
    # is TTL-gated (usually a no-op), so without this the Busy tick would
    # only surface on the release publish after the response — or never,
    # if release coalesces into the cooldown window.
    # Cooldown-expiry observation: a leftover retry_until whose window
    # lapsed enters ready-probation here (concurrency 1) instead of
    # slipping straight to ready (derive stays pure, so this gate is the
    # only live observer of expiry).
    if registry is not None and provider_id:
        _arm_probation_on_expiry(registry, provider_id)
    _inflight_track(request, provider_id)
    # Post-track quota re-check: two requests can both resolve to the
    # same probation provider between resolve() and track (two awaits
    # sit between them: refresh_health, _publish_providers). The second
    # arrival must not join the armed probe flight at concurrency 2 —
    # release its slot and re-enter the queued hold instead.
    if registry is not None and provider_id:
        try:
            _rt = registry.runtime(provider_id)
            if bool(getattr(_rt, "probation", False)) and _rt.in_flight > 1:
                _inflight_release(request, provider_id)
                provider_id, kind, warp_egress = None, "queued", None
        except Exception:
            pass
    if kind == "queued" and provider_id is None:
        # Bounced by the post-track quota re-check: re-enter the hold
        # loop above is behind us, so hold inline with the same budget
        # semantics (disconnect abort, degrade with the actual budget).
        _waited = 0.0
        _step = max(0.01, float(getattr(settings, "queue_keepalive_s", 15.0)))
        _budget = max(0.0, float(getattr(settings, "queue_wait_s", 600.0)))
        _resolve = getattr(egress_provider, "resolve", None)
        while _waited < _budget and kind == "queued":
            _sleep = min(_step, _budget - _waited)
            await asyncio.sleep(_sleep)
            _waited += _sleep
            try:
                if await request.is_disconnected():
                    return JSONResponse(
                        status_code=499,
                        content={"error": {"message": "client disconnected"}},
                    )
            except Exception:
                pass
            if callable(_resolve):
                provider_id, kind, warp_egress = _resolve(req.model, bucket)
            if kind == "warp" and warp_egress is not None:
                break
        if kind == "queued":
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "queue wait exceeded",
                        "type": "queue_timeout",
                    }
                },
                headers={"retry-after": str(int(_budget))},
            )
        if kind != "warp" or warp_egress is None:
            client = egress_provider.client_for(bucket, slot)
        else:
            try:
                client = warp_egress.client_for(bucket, slot)
            except RuntimeError:
                logger.info(
                    "[%s] warp provider %s has no ready exits; failing open",
                    trace_id,
                    provider_id,
                )
                provider_id, via_warp = None, None
                client = egress_provider.client_for(bucket, slot)
            else:
                ports = warp_egress.ready_ports()
                warp_idx = ports.index(warp_egress.pick_port(slot)) if ports else None
                via_warp = {
                    "provider_id": provider_id or "",
                    "warp_idx": warp_idx,
                    "socks_port": warp_egress.pick_port(slot),
                }
        if registry is not None and provider_id:
            _arm_probation_on_expiry(registry, provider_id)
        _inflight_track(request, provider_id)
    await _publish_providers(request)
    if registry is not None:
        # Pool-dry shed: direct also ratelimited and a warp bounce in
        # flight. Otherwise fall through to normal routing — resolve()
        # already skipped cycling warps, so the next request fails over
        # (fail open to direct as usual).
        shed, shed_provider = _shed_if_pool_dry(registry, req.model, trace_id)
        if shed is not None:
            _inflight_release(request, provider_id)
            _record_usage(request, ingress, req.model, shed)
            await registry.runtime(shed_provider or "").record(
                RecentRequest(
                    ts=time.time(),
                    model=req.model,
                    status=429,
                    ms=(time.monotonic() - started) * 1000.0,
                    warp_idx=None,
                    error="pool dry: warp cycling, direct limited",
                )
            )
            return shed
    tracker = getattr(request.app.state, "usage", None)
    usage_secret = getattr(request.state, "secret_key", None)
    stream_usage_cb = None
    if (
        outbound.get("stream") is True
        and tracker is not None
        and usage_secret is not None
    ):

        def stream_usage_cb(
            done, _tracker=tracker, _key=usage_secret, _model=req.model
        ):
            from llms.proxy.ir import StreamDone as _StreamDone

            if isinstance(done, _StreamDone):
                _tracker.record(
                    _key,
                    _model,
                    done.input_tokens,
                    done.output_tokens,
                    done.cached_tokens,
                    done.reasoning_tokens,
                    count_request=False,
                )
                # Stow the parser outcome for stream settle: a fully
                # consumed StreamingResponse always carries HTTP 200, so
                # settle cannot tell truncation/failure from success by
                # status alone. The sink runs at stream end on the same
                # request, before the wrap's exhaustion callback.
                try:
                    request.state.stream_outcome = done.status
                except Exception:
                    pass
                # Streaming passthrough legs hand the client the upstream
                # response id, which comes back as previous_response_id:
                # chain it to this session now (stream end) so the
                # follow-up turn reuses the session and the prompt cache
                # stays warm. (JSON bodies record at response time in
                # _record_conversation; translate legs mint downstream ids
                # recorded synchronously there.)
                if done.response_id and session_tracker is not None:
                    try:
                        session_tracker.remember(
                            _key, "chain:" + done.response_id, session_id
                        )
                    except Exception as exc:
                        logger.debug(
                            "[%s] session chain remember failed: %r",
                            trace_id,
                            exc,
                        )

    synthesize = (
        _synthesize_for(ingress, egress, req.model)
        if outbound.get("stream") is not True and not settings.zen_api_key
        else None
    )
    synthesize = (
        _synthesize_for(ingress, egress, req.model)
        if outbound.get("stream") is not True and not settings.zen_api_key
        else None
    )

    async def _do_forward():
        _response = await forward(
            client,
            url,
            headers,
            outbound,
            trace_id,
            convert=_convert_for(ingress, egress, req.model),
            translate_dialects=_stream_for(ingress, egress, req.model),
            synthesize_json=synthesize,
            via_warp=via_warp,
            # Usage is sniffed from upstream (egress-dialect) bytes; identical
            # for passthrough (ingress == egress) and required for translate.
            stream_ingress=egress if outbound.get("stream") is True else None,
            stream_usage_sink=stream_usage_cb,
        )
        if synthesize is not None and egress == "responses":
            # Steer any non-client tool call (genuine or hallucinated) back:
            # with client tools list them, otherwise demand a direct answer.
            _client_names = {t.name for t in req.tools if t.name}
            _response, _ = await _steer_genuine_calls(
                _response,
                client=client,
                url=url,
                headers=headers,
                outbound=outbound,
                synthesize=synthesize,
                trace_id=trace_id,
                client_names=_client_names,
            )
        return _response

    # Slow-request dedup (non-streaming only): a request outrunning
    # MAX_TIMEOUT gets 429 + Retry-After while its upstream work
    # continues in the background; a retry with a matching hash claims
    # the held response (evicting it) or 429s again while running.
    dedup_table = getattr(request.app.state, "dedup", None)
    dedup_key = None
    # Any exception escaping the forward/dedup await region below
    # (CancelledError on client disconnect, upstream errors) must release
    # the track() slot: the tail release is then unreachable. The handler
    # covers only this region and re-raises — a blanket post-track
    # try/finally would release the streaming path immediately and defeat
    # the deferred _wrap_inflight release.
    try:
        if dedup_table is not None and not req.stream:
            dedup_key = dedup.request_hash(
                secret_key, affinity, ingress, req.model, body, session_id
            )
            _entry = dedup_table.lookup(dedup_key)
            if _entry is not None:
                if _entry.done:
                    response = dedup.replay_response(
                        _entry.status, _entry.body, _entry.headers
                    )
                    dedup_table.drop(dedup_key)
                else:
                    response = dedup.inflight_response()
                # No upstream leg ran for this waiter: release the track()
                # slot before returning (same rationale as the shed and
                # synthetic-dedup returns).
                _inflight_release(request, provider_id)
                elapsed_ms = (time.monotonic() - started) * 1000.0
                return response
            _timeout = float(settings.max_timeout_s)
            if _timeout > 0:
                _task = asyncio.create_task(_do_forward())
                _entry = dedup_table.track(dedup_key, _task)
                _task.add_done_callback(
                    lambda t: _dedup_background_finish(
                        t,
                        table=dedup_table,
                        key=dedup_key,
                        entry=_entry,
                        request=request,
                        ingress=ingress,
                        model=req.model,
                        session_tracker=session_tracker,
                        secret_key=secret_key,
                        session_id=session_id,
                        is_new_conversation=is_new_conversation,
                        egress=egress,
                        client=client,
                        url=url,
                        headers=headers,
                        trace_id=trace_id,
                    )
                )
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(_task), timeout=_timeout
                    )
                except TimeoutError:
                    _entry.timed_out = True
                    # The upstream leg keeps running detached (shielded);
                    # the background finish never owns the slot (it would
                    # pin Busy for the full upstream duration), so the
                    # foreground releases here.
                    _inflight_release(request, provider_id)
                    response = dedup.inflight_response()
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    return response
            else:
                response = await _do_forward()
        else:
            response = await _do_forward()
    except BaseException:
        _inflight_release(request, provider_id)
        raise
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if (
        isinstance(response, JSONResponse)
        and response.headers.get(dedup.DEDUP_HEADER) is not None
    ):
        # Synthetic dedup response (inflight 429): skip every tail
        # side-effect so it can never read as provider congestion — but
        # release the track() slot first (no upstream leg ran for it).
        _inflight_release(request, provider_id)
        return response
    _note_free_tier_error(response, settings, trace_id)
    stream_id = None
    if (
        isinstance(response, StreamingResponse)
        and _stream_for(ingress, egress, req.model) is not None
    ):
        stream_id = {
            "responses": new_resp_id,
            "chat": new_chat_id,
            "messages": new_msg_id,
        }[ingress](trace_id)
    _record_conversation(response, session_tracker, secret_key, session_id, stream_id)
    _warm_new_conversation(
        response,
        is_new_conversation,
        egress,
        client,
        url,
        headers,
        req.model,
        session_id,
        trace_id,
    )
    outcome, retry_after = _outcome_of(response)
    if outcome == "ratelimited":
        new_slot = table.note_ratelimited(bucket, retry_after)
        logger.info(
            "[%s] bucket %s ratelimited, moved slot %s -> %s",
            trace_id,
            bucket,
            slot,
            new_slot,
        )
        registry = getattr(request.app.state, "providers", None)
        if registry is not None:
            reason = ""
            if isinstance(response, JSONResponse):
                try:
                    reason = (
                        json.loads(response.body.decode())
                        .get("error", {})
                        .get("message", "")
                    )
                except Exception:
                    reason = ""
            if provider_id:
                registry.runtime(provider_id).note_ratelimited(retry_after, reason)
                _maybe_auto_cycle(request, provider_id, via_warp, trace_id)
            else:
                # Direct path (fail-open or noproxy-routed): record on the
                # noproxy runtime so the pool-dry gate can see direct's
                # ratelimit. Without this the gate would never fire.
                direct_id = next(
                    (
                        p.id
                        for p in _providers_serving(registry, req.model)
                        if p.kind == "noproxy"
                    ),
                    "noproxy",
                )
                registry.runtime(direct_id).note_ratelimited(retry_after, reason)
            # The 429 backoff feeds the providers SSE status (yellow dot):
            # push so admin viewers converge without polling.
            await get_hub(request).publish("providers")
            # Fast-failover hint: pool min-retry over providers serving this
            # model ("when the next request is ok" per the 429 memo). ~1s
            # floor so the client retries fast onto the failover provider.
            if isinstance(response, JSONResponse):
                providers = _providers_serving(registry, req.model)
                wait = math.ceil(_min_retry_in(registry, providers))
                hint = max(1, min(60, int(wait)))
                response.headers["retry-after"] = str(hint)
                logger.info(
                    "[%s] provider %s ratelimited, retry-after hint %s",
                    trace_id,
                    provider_id or "direct",
                    hint,
                )
    _record_usage(request, ingress, req.model, response)
    if isinstance(response, StreamingResponse):
        return _wrap_inflight(
            request,
            response,
            provider_id,
            model=req.model,
            warp_idx=warp_idx,
            started=started,
        )
    _inflight_release(request, provider_id)
    if registry is not None and provider_id:
        _settle_probation(request, registry, provider_id, response)
        status = response.status_code if hasattr(response, "status_code") else 0
        error = ""
        if isinstance(response, JSONResponse) and status >= 400:
            try:
                error = (
                    json.loads(response.body.decode())
                    .get("error", {})
                    .get("message", "")
                )
            except Exception:
                error = ""
        await registry.runtime(provider_id).record(
            RecentRequest(
                ts=time.time(),
                model=req.model,
                status=status,
                ms=elapsed_ms,
                warp_idx=warp_idx,
                error=str(error)[:200],
            )
        )
    return response


def _arm_probation_on_expiry(registry, provider_id: str | None) -> bool:
    """Enter ready-probation when a recorded cooldown just expired.

    The only live observer of retry expiry (derive_lifecycle stays pure,
    so direct `retry_until = 0.0` test manipulation keeps working): a
    leftover non-zero `retry_until` whose `retry_in()` reached 0 means the
    cooldown lapsed without a request noticing — zero it and arm
    probation so the next flight probes at concurrency 1 instead of
    slipping straight to ready. Active backoffs (retry_in > 0) and
    never-limited providers (retry_until == 0) are untouched, preserving
    the Task 4 discipline that only 429/reconnect/expiry mutates
    retry_until. Never raises; returns True when it armed.
    """
    try:
        if registry is None or not provider_id:
            return False
        rt = registry.runtime(provider_id)
        if float(getattr(rt, "retry_until", 0.0) or 0.0) == 0.0:
            return False
        if rt.retry_in() > 0:
            return False
        if bool(getattr(rt, "probation", False)):
            return False
        rt.retry_until = 0.0
        rt.retry_reason = ""
        rt.probation = True
        return True
    except Exception:
        return False


def _settle_probation(request, registry, provider_id: str, response) -> None:
    """Promote or demote a probation provider on its probe outcome.

    Success (2xx) promotes probation -> ready (unbounded). Probe 429
    returns to ratelimited: note_ratelimited already applied the 60s
    default / retry-after value in the outcome block above, so only the
    flag clears here. Non-429 errors leave probation armed: the single
    flight stays the test, and the next request re-probes.

    Streams: a fully-consumed StreamingResponse always carries HTTP 200,
    so settle consults the parser outcome stowed on request.state by
    stream_usage_cb (failed/incomplete stays armed; only completed
    promotes). Absent outcome (untracked stream legs) falls back to
    status, preserving the old behavior.
    """
    try:
        rt = registry.runtime(provider_id)
        if not bool(getattr(rt, "probation", False)):
            return
        status = response.status_code if hasattr(response, "status_code") else 0
        outcome = None
        try:
            outcome = getattr(getattr(request, "state", None), "stream_outcome", None)
        except Exception:
            outcome = None
        if isinstance(response, StreamingResponse) and outcome is not None:
            if outcome == "completed" and 200 <= status < 300:
                rt.probation = False
            return
        if 200 <= status < 300 or status == 429:
            rt.probation = False
    except Exception:
        pass


def _providers_serving(registry, model: str) -> list:
    """Enabled providers serving a model (best-effort; [] when unknown)."""
    try:
        return [p for p in registry.load() if p.enabled and p.serves(model)]
    except Exception:
        return []


def _shed_if_pool_dry(
    registry, model: str, trace_id: str
) -> tuple[JSONResponse | None, str | None]:
    """Escalating 429 only when the provider pool is dry — else (None, None).

    Dry = direct (noproxy) also ratelimited AND ≥1 warp provider serving
    the model is mid-restart, with no serving provider currently usable
    (nothing available, but a bounce in flight means capacity is coming
    back soon). Anything else falls through to normal routing: resolve()
    already skipped cycling warps, so the next request fails over (fail
    open to direct as usual).

    Returns the shed response plus the cycling provider id (for the
    recent-request ring) when shedding.
    """
    providers = _providers_serving(registry, model)
    warps = [p for p in providers if p.kind == "warp"]
    cycling = [p for p in warps if registry.runtime(p.id).cycling]
    if not cycling:
        return None, None
    for p in providers:
        rt = registry.runtime(p.id)
        if p.kind == "noproxy":
            if rt.retry_in() <= 0:
                return None, None
        elif not rt.cycling and any(w.ready for w in rt.health.exits):
            return None, None
    rt = registry.runtime(cycling[0].id)
    rt.cycle_hits += 1
    retry_after = min(60, 5 + (rt.cycle_hits - 1))
    logger.info(
        "[%s] pool dry (warp %s cycling, direct limited), shedding 429 "
        "(hit %s, retry-after %s)",
        trace_id,
        cycling[0].id,
        rt.cycle_hits,
        retry_after,
    )
    return (
        JSONResponse(
            status_code=429,
            content={"error": {"message": f"warp provider {cycling[0].id} cycling"}},
            headers={"retry-after": str(retry_after)},
        ),
        cycling[0].id,
    )


def _min_retry_in(registry, providers: list) -> float:
    """Seconds until the next serving provider is expected usable (≥0)."""
    waits: list[float] = []
    for p in providers:
        rt = registry.runtime(p.id)
        if p.kind == "noproxy":
            waits.append(rt.retry_in())
            continue
        ready = any(w.ready for w in rt.health.exits)
        if ready or not rt.cycling:
            waits.append(0.0)
        else:
            # Cycling with no ready exit: the bounce (or its 45s bring-up)
            # is the soonest this provider recovers; cap the estimate so a
            # wedged bounce doesn't pin the hint.
            waits.append(45.0)
    return min(waits) if waits else 0.0


def _auto_cycle_cooldown_s(request: Request) -> float:
    settings = getattr(request.app.state, "settings", None)
    return float(getattr(settings, "warp_auto_cycle_cooldown_s", 300) or 300)


def _maybe_auto_cycle(
    request: Request, provider_id: str, via_warp: dict | None, trace_id: str
) -> None:
    """Bounce a ratelimited warp exit in the background (best-effort).

    Only fires for requests that actually rode warp (via_warp carries the
    exit's SOCKS port); guarded by per-provider cooldown + in-flight dedup.
    Never raises — the request path must not fail because the bounce did.
    """
    import asyncio
    import time

    try:
        if not via_warp or via_warp.get("socks_port") is None:
            return
        registry = getattr(request.app.state, "providers", None)
        if registry is None:
            return
        provider = next((p for p in registry.load() if p.id == provider_id), None)
        if provider is None or provider.kind != "warp":
            return
        rt = registry.runtime(provider_id)
        now = time.monotonic()
        if rt.cycling:
            logger.info(
                "[%s] warp provider %s already cycling, skipping auto-cycle",
                trace_id,
                provider_id,
            )
            return
        cooldown = _auto_cycle_cooldown_s(request)
        if now - rt.last_auto_cycle < cooldown:
            logger.info(
                "[%s] warp provider %s auto-cycle on cooldown (%.0fs left)",
                trace_id,
                provider_id,
                cooldown - (now - rt.last_auto_cycle),
            )
            return
        pool = registry.pool_for(provider)
        if pool is None:
            return
        port = via_warp["socks_port"]
        inst = next((w for w in pool.instances if w.socks_port == port), None)
        if inst is None:
            logger.warning(
                "[%s] warp provider %s auto-cycle: no slot on port %s",
                trace_id,
                provider_id,
                port,
            )
            return
        rt.last_auto_cycle = now
        rt.cycling = True
        rt.cycle_hits = 0
        bounce_epoch = rt.retry_epoch
        bounce_gen = rt.gen
        logger.info(
            "[%s] warp provider %s exit %s (port %s) ratelimited, cycling",
            trace_id,
            provider_id,
            inst.idx,
            port,
        )

        async def _bounce_and_clear() -> None:
            bounced_ok = False
            try:
                result = await pool.bounce_exit(inst.idx)
                bounced_ok = isinstance(result, dict) and bool(result.get("ok", False))
                logger.info(
                    "[%s] warp provider %s exit %s bounce done: %s",
                    trace_id,
                    provider_id,
                    inst.idx,
                    result,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] warp provider %s exit %s bounce failed: %r",
                    trace_id,
                    provider_id,
                    inst.idx,
                    exc,
                )
            rt.cycling = False
            rt.cycle_hits = 0
            if rt.gen != bounce_gen:
                # Superseded mid-bounce (disable/delete): the pool is
                # gone on purpose. refresh_health would ensure_pool a
                # brand-new pool for a dead provider and resurrect
                # daemons for it — skip the refresh, just converge SSE.
                # (Unconditional clears, not finally: a return in
                # finally would swallow bounce exceptions.)
                logger.info(
                    "[%s] warp provider %s auto-cycle superseded "
                    "(gen %s -> %s), skipping post-cycle refresh",
                    trace_id,
                    provider_id,
                    bounce_gen,
                    rt.gen,
                )
                try:
                    await get_hub(request).publish("providers")
                except Exception:
                    pass
                return
            try:
                await registry.refresh_health(provider, force=True)
            except Exception as exc:
                logger.warning(
                    "[%s] warp provider %s post-cycle refresh failed: %r",
                    trace_id,
                    provider_id,
                    exc,
                )
            if bounced_ok and rt.retry_epoch == bounce_epoch:
                # Fresh circuits: drop the 429 backoff so the provider
                # rejoins immediately instead of sitting out max(60s,
                # retry-after). Guarded by epoch: a sibling 429 that
                # landed mid-bounce bumped the epoch and must survive.
                # A still-bad exit keeps its backoff and the cooldown
                # gates the next bounce.
                rt.retry_until = 0.0
                rt.retry_reason = ""
                # Bounce-ok is a becoming-ready path: arm probation so
                # the next request probes at concurrency 1 (promotion
                # happens on first success).
                rt.probation = True
                logger.info(
                    "[%s] warp provider %s exit %s backoff cleared",
                    trace_id,
                    provider_id,
                    inst.idx,
                )
                # Backoff cleared: push the fresh (green) status to SSE.
                await get_hub(request).publish("providers")

        rt.cycle_task = asyncio.create_task(_bounce_and_clear())
    except Exception as exc:
        logger.warning(
            "[%s] warp provider %s auto-cycle trigger failed: %r",
            trace_id,
            provider_id,
            exc,
        )


def _record_usage(
    request: Request, ingress: str, model: str, response: Response
) -> None:
    from llms.proxy.ir import StreamDone
    from llms.proxy.usage import extract_usage

    tracker = getattr(request.app.state, "usage", None)
    secret_key = getattr(request.state, "secret_key", None)
    if tracker is None or secret_key is None:
        return
    if isinstance(response, StreamingResponse):
        # Translate path: usage arrives via stream_usage_sink when the
        # heartbeat generator finishes (see forward.translate_streaming).
        done = getattr(response, "stream_usage", None)
        if isinstance(done, StreamDone):
            tracker.record(
                secret_key,
                model,
                done.input_tokens,
                done.output_tokens,
                done.cached_tokens,
                done.reasoning_tokens,
            )
            return
        # Passthrough: the tap in forward.py records tokens synchronously
        # when the stream exhausts (request counted here, tokens merged on
        # completion without double-counting). Nothing more to do at
        # response-build time; fall through to the request-only record.
        tracker.record(secret_key, model, None, None, None, None)
        return
    if not isinstance(response, JSONResponse) or response.status_code >= 400:
        return
    try:
        payload = json.loads(response.body.decode())
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return
    in_tokens, out_tokens, cached_tokens, reasoning_tokens = extract_usage(
        ingress, payload
    )
    tracker.record(
        secret_key, model, in_tokens, out_tokens, cached_tokens, reasoning_tokens
    )
