from __future__ import annotations

import json
import time
from collections.abc import Iterable

from llms.proxy.logging import setup_logging

logger = setup_logging()


def split_events(lines: Iterable[str]) -> Iterable[str]:
    data: list[str] = []
    for line in lines:
        if line.strip() == "":
            if data:
                yield "\n".join(data)
                data = []
            continue
        if line.startswith("data:"):
            data.append(line[len("data:") :].strip())
    if data:
        yield "\n".join(data)


def skip_payload(payload: str) -> bool:
    return payload == "[DONE]" or "inference-cost" in payload


def chat_chunk(resp_id: str, model: str, delta: dict, finish: str | None) -> bytes:
    return (
        json.dumps(
            {
                "id": resp_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        ).encode()
        + b"\n\n"
    )


def responses_event(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def new_resp_id(trace_id: str) -> str:
    return f"resp_{trace_id}"


def new_chat_id(trace_id: str) -> str:
    return f"chatcmpl-{trace_id}"


def responses_to_chat(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    chat_id = new_chat_id(trace_id)
    names: dict[str, str] = {}
    indices: dict[str, int] = {}
    pending_item: str | None = None
    saw_item = False
    for payload in split_events(lines):
        if skip_payload(payload):
            continue
        try:
            event = json.loads(payload)
        except Exception as exc:
            logger.debug("skipping malformed SSE payload: %s", exc)
            continue
        kind = event.get("type", "")
        if kind == "response.output_item.added" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "function_call":
                names[item.get("id", "")] = item.get("name", "")
            continue
        if kind == "response.output_text.delta":
            saw_item = True
            yield b"data: " + chat_chunk(
                chat_id, model, {"content": event.get("delta", "")}, None
            )
            continue
        if kind == "response.function_call_arguments.delta":
            saw_item = True
            item_id = event.get("item_id", "")
            if item_id not in indices:
                indices[item_id] = len(indices)
            if pending_item != item_id:
                pending_item = item_id
                yield b"data: " + chat_chunk(
                    chat_id,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": indices[item_id],
                                "id": item_id,
                                "type": "function",
                                "function": {
                                    "name": names.get(item_id, ""),
                                    "arguments": "",
                                },
                            }
                        ]
                    },
                    None,
                )
            yield b"data: " + chat_chunk(
                chat_id,
                model,
                {
                    "tool_calls": [
                        {
                            "index": indices[item_id],
                            "function": {"arguments": event.get("delta", "")},
                        }
                    ]
                },
                None,
            )
            continue
        if kind in ("response.completed", "response.failed", "response.incomplete"):
            finish = "tool_calls" if indices else "stop"
            yield b"data: " + chat_chunk(chat_id, model, {}, finish)
            yield b"data: [DONE]\n\n"
            return
    if saw_item:
        finish = "tool_calls" if indices else "stop"
        yield b"data: " + chat_chunk(chat_id, model, {}, finish)
    yield b"data: [DONE]\n\n"


def chat_to_responses(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    resp_id = new_resp_id(trace_id)
    started = False
    tool_indices: dict[int, str] = {}
    tool_names: dict[int, str] = {}
    tool_args: dict[int, str] = {}
    text_started = False

    def begin_text():
        nonlocal started, text_started
        if not started:
            started = True
            yield responses_event(
                {
                    "type": "response.created",
                    "response": {
                        "id": resp_id,
                        "status": "in_progress",
                        "model": model,
                    },
                }
            )
            yield responses_event(
                {"type": "response.in_progress", "response": {"id": resp_id}}
            )
        if not text_started:
            text_started = True
            yield responses_event(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": f"{resp_id}-msg",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    },
                }
            )
            yield responses_event(
                {
                    "type": "response.content_part.added",
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }
            )

    def begin_tool(index: int, call_id: str, name: str):
        nonlocal started
        if not started:
            started = True
            yield responses_event(
                {
                    "type": "response.created",
                    "response": {
                        "id": resp_id,
                        "status": "in_progress",
                        "model": model,
                    },
                }
            )
            yield responses_event(
                {"type": "response.in_progress", "response": {"id": resp_id}}
            )
        yield responses_event(
            {
                "type": "response.output_item.added",
                "output_index": index + 1,
                "item": {
                    "id": call_id,
                    "type": "function_call",
                    "name": name,
                    "arguments": "",
                },
            }
        )

    for payload in split_events(lines):
        if skip_payload(payload):
            if payload == "[DONE]":
                break
            continue
        try:
            event = json.loads(payload)
        except Exception as exc:
            logger.debug("skipping malformed SSE payload: %s", exc)
            continue
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                yield from begin_text()
                yield responses_event(
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta["content"],
                    }
                )
            for call in delta.get("tool_calls", []):
                index = int(call.get("index", 0))
                fn = call.get("function", {})
                if index not in tool_indices:
                    call_id = call.get("id", f"{resp_id}-call-{index}")
                    tool_indices[index] = call_id
                    tool_names[index] = fn.get("name", "")
                    tool_args[index] = ""
                    yield from begin_tool(index, call_id, tool_names[index])
                if fn.get("arguments"):
                    tool_args[index] += fn["arguments"]
                    yield responses_event(
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": index + 1,
                            "item_id": tool_indices[index],
                            "delta": fn["arguments"],
                        }
                    )
            finish = choice.get("finish_reason")
            if finish:
                if text_started:
                    yield responses_event(
                        {
                            "type": "response.output_text.done",
                            "output_index": 0,
                            "content_index": 0,
                            "text": "",
                        }
                    )
                    yield responses_event(
                        {
                            "type": "response.content_part.done",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        }
                    )
                    yield responses_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "id": f"{resp_id}-msg",
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                            },
                        }
                    )
                for index, call_id in tool_indices.items():
                    yield responses_event(
                        {
                            "type": "response.function_call_arguments.done",
                            "output_index": index + 1,
                            "item_id": call_id,
                            "arguments": tool_args[index],
                        }
                    )
                    yield responses_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": index + 1,
                            "item": {
                                "id": call_id,
                                "type": "function_call",
                                "name": tool_names[index],
                                "arguments": tool_args[index],
                            },
                        }
                    )
                yield responses_event(
                    {
                        "type": "response.completed",
                        "response": {
                            "id": resp_id,
                            "status": "completed",
                            "model": model,
                        },
                    }
                )
                return
    yield responses_event(
        {
            "type": "response.completed",
            "response": {"id": resp_id, "status": "completed", "model": model},
        }
    )
