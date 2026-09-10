from __future__ import annotations

import time
import uuid

from llms.proxy.ir import (
    ROLE_ASSISTANT,
    LlmMessage,
    ResponseIR,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
)
from llms.proxy.translate import (
    ir_messages_to_messages_content,
    ir_messages_to_responses_output,
    messages_content_to_ir_blocks,
    responses_output_to_ir_messages,
)


def _chat_message_to_ir(message: dict) -> LlmMessage:
    blocks: list = []
    content = message.get("content")
    if (
        isinstance(message.get("reasoning_content"), str)
        and message["reasoning_content"]
    ):
        blocks.append(ThinkingBlock(message["reasoning_content"]))
    if isinstance(content, str) and content:
        blocks.append(TextBlock(content))
    for call in message.get("tool_calls", []) or []:
        fn = call.get("function", {})
        blocks.append(
            ToolCallBlock(
                str(call.get("id", "")),
                str(fn.get("name", "")),
                str(fn.get("arguments", "")),
            )
        )
    return LlmMessage(role=ROLE_ASSISTANT, blocks=tuple(blocks))


def _usage_details(usage: dict) -> tuple[int, int, int, int]:
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    prompt_details = usage.get(
        "prompt_tokens_details", usage.get("input_tokens_details", {})
    )
    completion_details = usage.get(
        "completion_tokens_details", usage.get("output_tokens_details", {})
    )
    cached = (prompt_details or {}).get("cached_tokens", 0)
    reasoning = (completion_details or {}).get("reasoning_tokens", 0)
    return int(prompt or 0), int(completion or 0), int(cached or 0), int(reasoning or 0)


def parse_chat_response(payload: dict) -> ResponseIR:
    choice = (payload.get("choices", []) or [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")
    usage = payload.get("usage", {})
    status = (
        "completed"
        if finish in ("stop", "tool_calls")
        else "incomplete"
        if finish == "length"
        else "failed"
    )
    prompt, completion, cached, reasoning = _usage_details(usage)
    return ResponseIR(
        model=str(payload.get("model", "")),
        status=status,
        messages=(_chat_message_to_ir(message),),
        input_tokens=prompt,
        output_tokens=completion,
        raw_id=str(payload.get("id", uuid.uuid4().hex[:12])).removeprefix("chatcmpl-"),
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        incomplete_reason="max_output_tokens" if finish == "length" else None,
    )


def parse_responses_response(payload: dict) -> ResponseIR:
    usage = payload.get("usage", {})
    prompt, completion, cached, reasoning = _usage_details(usage)
    details = payload.get("incomplete_details") or {}
    return ResponseIR(
        model=str(payload.get("model", "")),
        status=str(payload.get("status", "completed")),
        messages=responses_output_to_ir_messages(payload.get("output", [])),
        input_tokens=prompt,
        output_tokens=completion,
        raw_id=str(payload.get("id", uuid.uuid4().hex[:12])).removeprefix("resp_"),
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        incomplete_reason=details.get("reason"),
    )


def parse_messages_response(payload: dict) -> ResponseIR:
    stop = payload.get("stop_reason", "end_turn")
    if stop in ("end_turn", "tool_use", "stop_sequence"):
        status = "completed"
    elif stop == "max_tokens":
        status = "incomplete"
    else:
        status = "failed"
    usage = payload.get("usage", {})
    cached = int(usage.get("cache_read_input_tokens") or 0)
    return ResponseIR(
        model=str(payload.get("model", "")),
        status=status,
        messages=(
            LlmMessage(
                role=ROLE_ASSISTANT,
                blocks=messages_content_to_ir_blocks(payload.get("content", [])),
            ),
        ),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        raw_id=str(payload.get("id", uuid.uuid4().hex[:12])).removeprefix("msg_"),
        cached_tokens=cached,
        incomplete_reason="max_output_tokens" if stop == "max_tokens" else None,
    )


PARSERS = {
    "chat": parse_chat_response,
    "responses": parse_responses_response,
    "messages": parse_messages_response,
}


def _total(rir: ResponseIR) -> int:
    return rir.input_tokens + rir.output_tokens


def _has_calls(rir: ResponseIR) -> bool:
    return any(isinstance(b, ToolCallBlock) for m in rir.messages for b in m.blocks)


def emit_chat_response(rir: ResponseIR, model: str) -> dict:
    texts = "".join(
        b.text for m in rir.messages for b in m.blocks if isinstance(b, TextBlock)
    )
    thinking = "\n".join(
        b.text for m in rir.messages for b in m.blocks if isinstance(b, ThinkingBlock)
    )
    calls = [
        {
            "id": b.call_id,
            "type": "function",
            "function": {"name": b.name, "arguments": b.arguments},
        }
        for m in rir.messages
        for b in m.blocks
        if isinstance(b, ToolCallBlock)
    ]
    if rir.status == "completed" and calls:
        finish = "tool_calls"
    elif rir.status == "completed":
        finish = "stop"
    elif rir.status == "incomplete":
        finish = "length"
    else:
        finish = "stop"
    message: dict = {
        "role": "assistant",
        "content": texts or None,
        "tool_calls": calls or None,
    }
    if thinking:
        message["reasoning_content"] = thinking
    return {
        "id": f"chatcmpl-{rir.raw_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": rir.input_tokens,
            "completion_tokens": rir.output_tokens,
            "total_tokens": _total(rir),
            "prompt_tokens_details": {"cached_tokens": rir.cached_tokens},
            "completion_tokens_details": {"reasoning_tokens": rir.reasoning_tokens},
        },
    }


def emit_responses_response(rir: ResponseIR, model: str) -> dict:
    body: dict = {
        "id": f"resp_{rir.raw_id}",
        "object": "response",
        "created_at": int(time.time()),
        "status": rir.status,
        "model": model,
        "error": None,
        "output": ir_messages_to_responses_output(rir.messages),
        "usage": {
            "input_tokens": rir.input_tokens,
            "output_tokens": rir.output_tokens,
            "total_tokens": _total(rir),
            "input_tokens_details": {"cached_tokens": rir.cached_tokens},
            "output_tokens_details": {"reasoning_tokens": rir.reasoning_tokens},
        },
    }
    if rir.status == "incomplete":
        body["incomplete_details"] = {
            "reason": rir.incomplete_reason or "max_output_tokens"
        }
    return body


def emit_messages_response(rir: ResponseIR, model: str) -> dict:
    if rir.status == "completed" and _has_calls(rir):
        stop = "tool_use"
    elif rir.status == "completed":
        stop = "end_turn"
    elif rir.status == "incomplete":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    return {
        "id": f"msg_{rir.raw_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": ir_messages_to_messages_content(rir.messages),
        "stop_reason": stop,
        "usage": {
            "input_tokens": rir.input_tokens,
            "output_tokens": rir.output_tokens,
            "cache_read_input_tokens": rir.cached_tokens,
        },
    }


EMITTERS = {
    "chat": emit_chat_response,
    "responses": emit_responses_response,
    "messages": emit_messages_response,
}


def convert_response(ingress: str, egress: str, payload: dict, model: str) -> dict:
    return EMITTERS[egress](PARSERS[ingress](payload), model)


def responses_to_chat(payload: dict, model: str) -> dict:
    return convert_response("responses", "chat", payload, model)


def chat_to_responses(payload: dict, model: str) -> dict:
    return convert_response("chat", "responses", payload, model)


def responses_to_messages(payload: dict, model: str) -> dict:
    return convert_response("responses", "messages", payload, model)


def messages_to_responses(payload: dict, model: str) -> dict:
    return convert_response("messages", "responses", payload, model)


def chat_to_messages(payload: dict, model: str) -> dict:
    return convert_response("chat", "messages", payload, model)


def messages_to_chat(payload: dict, model: str) -> dict:
    return convert_response("messages", "chat", payload, model)
