from __future__ import annotations

import json
import math
import time

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llms.proxy.affinity import bucket_for
from llms.proxy.config import Settings
from llms.proxy.forward import forward, parse_body
from llms.proxy.ir import RequestIR
from llms.proxy.logging import log_ingress, log_upstream, new_trace_id, setup_logging
from llms.proxy.providers import RecentRequest
from llms.proxy.rate_limit import classify
from llms.proxy.router import ENDPOINT_PATH, pick, resolve_alias
from llms.proxy.stream_translate import (
    PARSERS,
)
from llms.proxy.stream_translate import (
    chat_to_messages as stream_chat_to_messages,
)
from llms.proxy.stream_translate import (
    chat_to_responses as stream_chat_to_responses,
)
from llms.proxy.stream_translate import (
    messages_to_chat as stream_messages_to_chat,
)
from llms.proxy.stream_translate import (
    messages_to_responses as stream_messages_to_responses,
)
from llms.proxy.stream_translate import (
    responses_to_chat as stream_responses_to_chat,
)
from llms.proxy.stream_translate import (
    responses_to_messages as stream_responses_to_messages,
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
    translators = {
        ("chat", "responses"): stream_responses_to_chat,
        ("responses", "chat"): stream_chat_to_responses,
        ("messages", "responses"): stream_responses_to_messages,
        ("responses", "messages"): stream_messages_to_responses,
        ("chat", "messages"): stream_messages_to_chat,
        ("messages", "chat"): stream_chat_to_messages,
    }
    translate = translators.get((ingress, egress))
    if translate is None:
        return None
    return lambda lines, trace_id: translate(lines, trace_id, model)


def _synthesize_for(ingress: str, egress: str, model: str):
    """Fold upstream SSE into one downstream JSON body (responses leg only).

    Anonymous Zen requires stream:true even for single-shot callers; the
    deltas accumulate into a ResponseIR that the normal response emitters
    render in the ingress dialect.
    """
    if egress != "responses":
        return None

    def run(lines, trace_id):
        return EMITTERS[ingress](
            deltas_to_response_ir(PARSERS[egress](lines), model), model
        )

    return run


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
    # Zen's free-tier gate requires body prompt_cache_key to equal the
    # x-opencode-session header (both well-formed ses_ IDs) on the
    # responses leg; mint one id here and share it with the headers.
    # Chat has no such field/gate; messages needs a real API key instead.
    session_id = stable_session_id(settings.zen_api_key)
    if egress == "responses":
        outbound["prompt_cache_key"] = session_id
    affinity = getattr(request.state, "affinity", None)
    secret_key = getattr(request.state, "secret_key", None)
    bucket = bucket_for(affinity, req.model, settings.num_buckets, secret_key)
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
    via_warp: dict | None = None
    warp_idx: int | None = None
    registry = getattr(request.app.state, "providers", None)
    resolve = getattr(egress_provider, "resolve", None)
    if callable(resolve):
        provider_id, kind, warp_egress = resolve(req.model)
        if kind == "warp" and warp_egress is not None:
            provider = None
            if registry is not None:
                for p in registry.load():
                    if p.id == provider_id:
                        provider = p
                        break
            if registry is not None and provider is not None:
                await registry.refresh_health(provider)
                # Re-resolve after the refresh: a slot that flipped ready
                # during this very poll must seed the egress's SOCKS ports
                # before client_for runs, or the request fails open despite
                # a connected tunnel (live finding: final status.json showed
                # ready=true while the request went direct).
                provider_id, kind, warp_egress = resolve(req.model)
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
    if registry is not None:
        # Pool-dry shed: direct also ratelimited and a warp bounce in
        # flight. Otherwise fall through to normal routing — resolve()
        # already skipped cycling warps, so the next request fails over
        # (fail open to direct as usual).
        shed, shed_provider = _shed_if_pool_dry(registry, req.model, trace_id)
        if shed is not None:
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

    response = await forward(
        client,
        url,
        headers,
        outbound,
        trace_id,
        convert=_convert_for(ingress, egress, req.model),
        translate_stream=_stream_for(ingress, egress, req.model),
        synthesize_json=(
            _synthesize_for(ingress, egress, req.model)
            if outbound.get("stream") is not True and not settings.zen_api_key
            else None
        ),
        via_warp=via_warp,
        # Usage is sniffed from upstream (egress-dialect) bytes; identical
        # for passthrough (ingress == egress) and required for translate.
        stream_ingress=egress if outbound.get("stream") is True else None,
        stream_usage_sink=stream_usage_cb,
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    _note_free_tier_error(response, settings, trace_id)
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
    if registry is not None and provider_id:
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
        logger.info(
            "[%s] warp provider %s exit %s (port %s) ratelimited, cycling",
            trace_id,
            provider_id,
            inst.idx,
            port,
        )

        async def _bounce_and_clear() -> None:
            try:
                result = await pool.bounce_exit(inst.idx)
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
            finally:
                rt.cycling = False
                rt.cycle_hits = 0
                try:
                    await registry.refresh_health(provider, force=True)
                except Exception as exc:
                    logger.warning(
                        "[%s] warp provider %s post-cycle refresh failed: %r",
                        trace_id,
                        provider_id,
                        exc,
                    )

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
        # heartbeat generator finishes (see forward.translate_with_heartbeat).
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
