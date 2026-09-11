from __future__ import annotations

import json
import time

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llms.proxy.affinity import bucket_for
from llms.proxy.config import Settings
from llms.proxy.forward import forward, parse_body
from llms.proxy.ir import RequestIR
from llms.proxy.logging import log_ingress, log_upstream, new_trace_id, setup_logging
from llms.proxy.providers import RecentRequest, warp_exit_for
from llms.proxy.rate_limit import classify
from llms.proxy.router import ENDPOINT_PATH, pick, resolve_alias
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
from llms.proxy.translate_response import convert_response
from llms.proxy.zen_headers import build_zen_headers

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
    headers = build_zen_headers(settings)
    url = settings.zen_base_url.rstrip("/") + ENDPOINT_PATH[egress]
    log_upstream(trace_id, url, headers, outbound)
    egress_provider = request.app.state.egress
    started = time.monotonic()
    provider_id: str | None = None
    via_pool: dict | None = None
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
            pool_base = provider.base_url if provider else warp_egress.pool_base_url
            pool_token = provider.token if provider else warp_egress.token
            if registry is not None and provider is not None:
                health = await registry.refresh_health(provider)
                warp_idx = warp_exit_for(health, bucket, slot)
            via_pool = {
                "base_url": pool_base,
                "token": pool_token,
                "provider_id": provider_id or "",
                "warp_idx": warp_idx,
            }
            if outbound.get("stream") is True:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": {
                            "message": "streaming is not supported through warp providers"
                        }
                    },
                )
            client = warp_egress.client_for(bucket, slot)
        else:
            client = egress_provider.client_for(bucket, slot)
    else:
        client = egress_provider.client_for(bucket, slot)
    response = await forward(
        client,
        url,
        headers,
        outbound,
        trace_id,
        convert=_convert_for(ingress, egress, req.model),
        translate_stream=_stream_for(ingress, egress, req.model),
        via_pool=via_pool,
    )
    elapsed_ms = (__import__("time").monotonic() - started) * 1000.0
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
        if registry is not None and provider_id:
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
            registry.runtime(provider_id).note_ratelimited(retry_after, reason)
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


def _record_usage(
    request: Request, ingress: str, model: str, response: Response
) -> None:
    from llms.proxy.usage import extract_usage

    tracker = getattr(request.app.state, "usage", None)
    secret_key = getattr(request.state, "secret_key", None)
    if tracker is None or secret_key is None:
        return
    if isinstance(response, StreamingResponse):
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
