from __future__ import annotations

import time
import uuid

from llms.proxy.ir import ROLE_ASSISTANT, LlmMessage, ToolCallBlock


def _resp_text_of(output: list) -> str:
    texts = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
    return "".join(texts)


def _resp_calls_of(output: list) -> list:
    return [
        {
            "id": str(item.get("call_id", item.get("id", ""))),
            "type": "function",
            "function": {
                "name": str(item.get("name", "")),
                "arguments": str(item.get("arguments", "")),
            },
        }
        for item in output
        if item.get("type") == "function_call"
    ]


def responses_to_chat(payload: dict, model: str) -> dict:
    raw_id = str(payload.get("id", uuid.uuid4().hex[:12])).removeprefix("resp_")
    output = payload.get("output", [])
    calls = _resp_calls_of(output)
    status = payload.get("status", "completed")
    if calls and status == "completed":
        finish = "tool_calls"
    elif status == "completed":
        finish = "stop"
    elif status == "incomplete":
        finish = "length"
    else:
        finish = "stop"
    usage = payload.get("usage", {})
    return {
        "id": f"chatcmpl-{raw_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _resp_text_of(output) or None,
                    "tool_calls": calls or None,
                },
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def chat_to_responses(payload: dict, model: str) -> dict:
    raw_id = str(payload.get("id", uuid.uuid4().hex[:12])).removeprefix("chatcmpl-")
    choice = (payload.get("choices", []) or [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")
    output: list = []
    content = message.get("content")
    if content:
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": content, "annotations": []}
                ],
            }
        )
    for call in message.get("tool_calls", []) or []:
        fn = call.get("function", {})
        output.append(
            {
                "type": "function_call",
                "call_id": str(call.get("id", "")),
                "name": str(fn.get("name", "")),
                "arguments": str(fn.get("arguments", "")),
            }
        )
    status = "completed" if finish in ("stop", "tool_calls") else "incomplete"
    usage = payload.get("usage", {})
    return {
        "id": f"resp_{raw_id}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "error": None,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def responses_to_messages(payload: dict, model: str) -> dict:
    from llms.proxy.translate import (
        ir_messages_to_messages_content,
        responses_output_to_ir_messages,
    )

    raw_id = str(payload.get("id", uuid.uuid4().hex[:12])).removeprefix("resp_")
    output = payload.get("output", [])
    messages = responses_output_to_ir_messages(output)
    content = ir_messages_to_messages_content(messages)
    status = payload.get("status", "completed")
    has_calls = any(isinstance(b, ToolCallBlock) for m in messages for b in m.blocks)
    if has_calls and status == "completed":
        stop = "tool_use"
    elif status == "completed":
        stop = "end_turn"
    elif status == "incomplete":
        stop = "max_tokens"
    else:
        stop = "end_turn"
    usage = payload.get("usage", {})
    return {
        "id": f"msg_{raw_id}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
    }


def messages_to_responses(payload: dict, model: str) -> dict:
    from llms.proxy.translate import (
        ir_messages_to_responses_output,
        messages_content_to_ir_blocks,
    )

    raw_id = str(payload.get("id", uuid.uuid4().hex[:12])).removeprefix("msg_")
    messages = (
        LlmMessage(
            role=ROLE_ASSISTANT,
            blocks=messages_content_to_ir_blocks(payload.get("content", [])),
        ),
    )
    output = ir_messages_to_responses_output(messages)
    stop = payload.get("stop_reason", "end_turn")
    status = (
        "completed"
        if stop in ("end_turn", "tool_use", "stop_sequence")
        else "incomplete"
    )
    usage = payload.get("usage", {})
    return {
        "id": f"resp_{raw_id}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "error": None,
        "output": output,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        },
    }
