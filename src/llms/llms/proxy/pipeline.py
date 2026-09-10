from __future__ import annotations

import json

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from llms.proxy.affinity import bucket_for
from llms.proxy.config import Settings
from llms.proxy.forward import forward, parse_body
from llms.proxy.ir import LlmRequest
from llms.proxy.logging import log_ingress, log_upstream, new_trace_id, setup_logging
from llms.proxy.rate_limit import classify
from llms.proxy.router import ENDPOINT_PATH, pick
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
    chat_to_responses,
    messages_to_responses,
    responses_to_chat,
    responses_to_messages,
)
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
    converters = {
        ("chat", "responses"): responses_to_chat,
        ("responses", "chat"): chat_to_responses,
        ("messages", "responses"): responses_to_messages,
        ("responses", "messages"): messages_to_responses,
    }
    convert = converters.get((ingress, egress))
    if convert is None:
        return None
    return lambda payload: convert(payload, model)


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
        req: LlmRequest = FROM[ingress](body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})
    if not req.model:
        req = with_model(req, getattr(settings, DEFAULT_MODEL_ATTR[ingress]))
    egress = pick(req.model, ingress)
    outbound = TO[egress](req)
    affinity = getattr(request.state, "affinity", None)
    bucket = bucket_for(affinity, req.model, settings.num_buckets)
    table = request.app.state.bucket_table
    slot = table.slot_for(bucket)
    log_ingress(
        trace_id,
        request.url.path,
        {
            "model": req.model,
            "ingress": ingress,
            "egress": egress,
            "affinity": affinity,
            "bucket": bucket,
            "slot": slot,
        },
    )
    headers = build_zen_headers(settings, request.headers.get("authorization"))
    url = settings.zen_base_url.rstrip("/") + ENDPOINT_PATH[egress]
    log_upstream(trace_id, url, headers, outbound)
    egress_provider = request.app.state.egress
    client = egress_provider.client_for(bucket, slot)
    response = await forward(
        client,
        url,
        headers,
        outbound,
        trace_id,
        convert=_convert_for(ingress, egress, req.model),
        translate_stream=_stream_for(ingress, egress, req.model),
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
    return response
