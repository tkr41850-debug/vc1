from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from llms.proxy.config import Settings
from llms.proxy.forward import forward, parse_body
from llms.proxy.ir import LlmRequest
from llms.proxy.logging import log_ingress, log_upstream, new_trace_id
from llms.proxy.router import ENDPOINT_PATH, pick
from llms.proxy.stream_translate import (
    chat_to_responses as stream_chat_to_responses,
)
from llms.proxy.stream_translate import (
    responses_to_chat as stream_responses_to_chat,
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
from llms.proxy.translate_response import chat_to_responses, responses_to_chat
from llms.proxy.zen_headers import build_zen_headers

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
    if ingress == "chat" and egress == "responses":
        return lambda payload: responses_to_chat(payload, model)
    if ingress == "responses" and egress == "chat":
        return lambda payload: chat_to_responses(payload, model)
    return None


def _stream_for(ingress: str, egress: str, model: str):
    if ingress == egress:
        return None
    if ingress == "chat" and egress == "responses":
        return lambda lines, trace_id: stream_responses_to_chat(lines, trace_id, model)
    if ingress == "responses" and egress == "chat":
        return lambda lines, trace_id: stream_chat_to_responses(lines, trace_id, model)
    return None
    if ingress == egress:
        return None
    if ingress == "chat" and egress == "responses":
        return lambda payload: responses_to_chat(payload, model)
    if ingress == "responses" and egress == "chat":
        return lambda payload: chat_to_responses(payload, model)
    return None


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
    log_ingress(
        trace_id,
        request.url.path,
        {"model": req.model, "ingress": ingress, "egress": egress},
    )
    headers = build_zen_headers(settings, request.headers.get("authorization"))
    url = settings.zen_base_url.rstrip("/") + ENDPOINT_PATH[egress]
    log_upstream(trace_id, url, headers, outbound)
    client = request.app.state.upstream_client
    return await forward(
        client,
        url,
        headers,
        outbound,
        trace_id,
        convert=_convert_for(ingress, egress, req.model),
        translate_stream=_stream_for(ingress, egress, req.model),
    )
