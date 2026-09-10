from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator

from llms.proxy.ir import StreamDone, TextDelta, ToolArgsDelta
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


def new_resp_id(trace_id: str) -> str:
    return f"resp_{trace_id}"


def new_chat_id(trace_id: str) -> str:
    return f"chatcmpl-{trace_id}"


def new_msg_id(trace_id: str) -> str:
    return f"msg_{trace_id}"


def _load(payload: str) -> dict | None:
    try:
        event = json.loads(payload)
    except Exception as exc:
        logger.debug("skipping malformed SSE payload: %s", exc)
        return None
    return event if isinstance(event, dict) else None


def parse_chat_sse(
    lines: Iterable[str],
) -> Iterator[TextDelta | ToolArgsDelta | StreamDone]:
    index_to_id: dict[int, str] = {}
    names: dict[str, str] = {}
    saw_calls = False
    for payload in split_events(lines):
        if payload == "[DONE]":
            break
        if "inference-cost" in payload:
            continue
        event = _load(payload)
        if event is None:
            continue
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if delta.get("content"):
                yield TextDelta(delta["content"])
            for call in delta.get("tool_calls", []):
                index = int(call.get("index", 0))
                fn = call.get("function", {})
                if index not in index_to_id:
                    index_to_id[index] = call.get("id", f"call-{index}")
                    names[index_to_id[index]] = fn.get("name", "")
                call_id = index_to_id[index]
                if fn.get("name") and not names[call_id]:
                    names[call_id] = fn["name"]
                yield ToolArgsDelta(call_id, names[call_id], fn.get("arguments", ""))
                saw_calls = True
            finish = choice.get("finish_reason")
            if finish:
                if finish in ("stop", "tool_calls"):
                    yield StreamDone(
                        "completed", has_tool_calls=saw_calls or finish == "tool_calls"
                    )
                elif finish == "length":
                    yield StreamDone("incomplete", has_tool_calls=saw_calls)
                else:
                    yield StreamDone("failed", has_tool_calls=saw_calls)
                return
    yield StreamDone("completed", has_tool_calls=saw_calls)


def parse_responses_sse(
    lines: Iterable[str],
) -> Iterator[TextDelta | ToolArgsDelta | StreamDone]:
    names: dict[str, str] = {}
    saw_calls = False
    for payload in split_events(lines):
        if skip_payload(payload):
            continue
        event = _load(payload)
        if event is None:
            continue
        kind = event.get("type", "")
        if kind == "response.output_item.added" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "function_call":
                names[item.get("id", "")] = item.get("name", "")
            continue
        if kind == "response.output_text.delta":
            yield TextDelta(event.get("delta", ""))
            continue
        if kind == "response.function_call_arguments.delta":
            item_id = event.get("item_id", "")
            saw_calls = True
            yield ToolArgsDelta(item_id, names.get(item_id, ""), event.get("delta", ""))
            continue
        if kind in ("response.completed", "response.failed", "response.incomplete"):
            status = {
                "response.completed": "completed",
                "response.failed": "failed",
            }.get(kind, "incomplete")
            yield StreamDone(status, has_tool_calls=saw_calls or bool(names))
            return
    yield StreamDone("completed", has_tool_calls=saw_calls or bool(names))


def parse_messages_sse(
    lines: Iterable[str],
) -> Iterator[TextDelta | ToolArgsDelta | StreamDone]:
    ids: dict[int, str] = {}
    names: dict[int, str] = {}
    stop = "end_turn"
    for payload in split_events(lines):
        if skip_payload(payload):
            continue
        event = _load(payload)
        if event is None:
            continue
        kind = event.get("type", "")
        if kind == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                ids[event.get("index", 0)] = block.get("id", "")
                names[event.get("index", 0)] = block.get("name", "")
            continue
        if kind == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                yield TextDelta(delta.get("text", ""))
            elif delta.get("type") == "input_json_delta":
                index = event.get("index", 0)
                yield ToolArgsDelta(
                    ids.get(index, ""),
                    names.get(index, ""),
                    delta.get("partial_json", ""),
                )
            continue
        if kind == "message_delta":
            stop = event.get("delta", {}).get("stop_reason", stop)
            continue
        if kind == "message_stop":
            yield StreamDone(
                "completed"
                if stop in ("end_turn", "tool_use", "stop_sequence")
                else "incomplete"
                if stop == "max_tokens"
                else "failed"
            )
            return
    yield StreamDone("completed")


def _chat_chunk(chat_id: str, model: str, delta: dict, finish: str | None) -> bytes:
    return (
        b"data: "
        + json.dumps(
            {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        ).encode()
        + b"\n\n"
    )


def _resp_event(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def _msg_event(event_type: str, payload: dict) -> bytes:
    body = {"type": event_type}
    body.update(payload)
    return (
        b"event: "
        + event_type.encode()
        + b"\ndata: "
        + json.dumps(body).encode()
        + b"\n\n"
    )


def emit_chat_sse(
    deltas: Iterable[TextDelta | ToolArgsDelta | StreamDone], trace_id: str, model: str
) -> Iterable[bytes]:
    chat_id = new_chat_id(trace_id)
    indices: dict[str, int] = {}
    announced: set[str] = set()
    for delta in deltas:
        if isinstance(delta, TextDelta):
            yield _chat_chunk(chat_id, model, {"content": delta.text}, None)
        elif isinstance(delta, ToolArgsDelta):
            if delta.call_id not in indices:
                indices[delta.call_id] = len(indices)
            if delta.call_id not in announced:
                announced.add(delta.call_id)
                yield _chat_chunk(
                    chat_id,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": indices[delta.call_id],
                                "id": delta.call_id,
                                "type": "function",
                                "function": {"name": delta.name, "arguments": ""},
                            }
                        ]
                    },
                    None,
                )
            if delta.args_chunk:
                yield _chat_chunk(
                    chat_id,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": indices[delta.call_id],
                                "function": {"arguments": delta.args_chunk},
                            }
                        ]
                    },
                    None,
                )
        elif isinstance(delta, StreamDone):
            if delta.status == "completed" and delta.has_tool_calls:
                finish = "tool_calls"
            elif delta.status == "completed":
                finish = "stop"
            elif delta.status == "incomplete":
                finish = "length"
            else:
                finish = "stop"
            yield _chat_chunk(chat_id, model, {}, finish)
            yield b"data: [DONE]\n\n"
            return


def emit_responses_sse(
    deltas: Iterable[TextDelta | ToolArgsDelta | StreamDone], trace_id: str, model: str
) -> Iterable[bytes]:
    resp_id = new_resp_id(trace_id)
    started = False
    text_open = False
    text_accum = ""
    tool_items: dict[str, int] = {}
    tool_names: dict[str, str] = {}
    tool_args: dict[str, str] = {}

    def begin():
        nonlocal started
        if started:
            return
        started = True
        yield _resp_event(
            {
                "type": "response.created",
                "response": {"id": resp_id, "status": "in_progress", "model": model},
            }
        )
        yield _resp_event({"type": "response.in_progress", "response": {"id": resp_id}})

    for delta in deltas:
        if isinstance(delta, TextDelta):
            yield from begin()
            text_accum += delta.text
            if not text_open:
                text_open = True
                yield _resp_event(
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
                yield _resp_event(
                    {
                        "type": "response.content_part.added",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    }
                )
            yield _resp_event(
                {
                    "type": "response.output_text.delta",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta.text,
                }
            )
        elif isinstance(delta, ToolArgsDelta):
            yield from begin()
            if delta.call_id not in tool_items:
                tool_items[delta.call_id] = len(tool_items) + 1
                tool_names[delta.call_id] = delta.name
                tool_args[delta.call_id] = ""
                yield _resp_event(
                    {
                        "type": "response.output_item.added",
                        "output_index": tool_items[delta.call_id],
                        "item": {
                            "id": delta.call_id,
                            "type": "function_call",
                            "name": delta.name,
                            "arguments": "",
                        },
                    }
                )
            if delta.args_chunk:
                tool_args[delta.call_id] += delta.args_chunk
                yield _resp_event(
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": tool_items[delta.call_id],
                        "item_id": delta.call_id,
                        "delta": delta.args_chunk,
                    }
                )
        elif isinstance(delta, StreamDone):
            yield from begin()
            if text_open:
                yield _resp_event(
                    {
                        "type": "response.output_text.done",
                        "output_index": 0,
                        "content_index": 0,
                        "text": text_accum,
                    }
                )
                yield _resp_event(
                    {
                        "type": "response.content_part.done",
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": text_accum,
                            "annotations": [],
                        },
                    }
                )
                yield _resp_event(
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
            for call_id, index in tool_items.items():
                yield _resp_event(
                    {
                        "type": "response.function_call_arguments.done",
                        "output_index": index,
                        "item_id": call_id,
                        "arguments": tool_args[call_id],
                    }
                )
                yield _resp_event(
                    {
                        "type": "response.output_item.done",
                        "output_index": index,
                        "item": {
                            "id": call_id,
                            "type": "function_call",
                            "name": tool_names[call_id],
                            "arguments": tool_args[call_id],
                        },
                    }
                )
            yield _resp_event(
                {
                    "type": f"response.{delta.status}",
                    "response": {"id": resp_id, "status": delta.status, "model": model},
                }
            )
            return


def emit_messages_sse(
    deltas: Iterable[TextDelta | ToolArgsDelta | StreamDone], trace_id: str, model: str
) -> Iterable[bytes]:
    msg_id = new_msg_id(trace_id)
    started = False
    text_index: int | None = None
    tool_indices: dict[str, int] = {}
    next_index = 0

    def begin():
        nonlocal started
        if started:
            return
        started = True
        yield _msg_event(
            "message_start",
            {
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            },
        )

    for delta in deltas:
        if isinstance(delta, TextDelta):
            yield from begin()
            if text_index is None:
                text_index = next_index
                next_index += 1
                yield _msg_event(
                    "content_block_start",
                    {
                        "index": text_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            yield _msg_event(
                "content_block_delta",
                {
                    "index": text_index,
                    "delta": {"type": "text_delta", "text": delta.text},
                },
            )
        elif isinstance(delta, ToolArgsDelta):
            yield from begin()
            if delta.call_id not in tool_indices:
                tool_indices[delta.call_id] = next_index
                next_index += 1
                yield _msg_event(
                    "content_block_start",
                    {
                        "index": tool_indices[delta.call_id],
                        "content_block": {
                            "type": "tool_use",
                            "id": delta.call_id,
                            "name": delta.name,
                            "input": {},
                        },
                    },
                )
            if delta.args_chunk:
                yield _msg_event(
                    "content_block_delta",
                    {
                        "index": tool_indices[delta.call_id],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": delta.args_chunk,
                        },
                    },
                )
        elif isinstance(delta, StreamDone):
            yield from begin()
            if text_index is not None:
                yield _msg_event("content_block_stop", {"index": text_index})
            for index in tool_indices.values():
                yield _msg_event("content_block_stop", {"index": index})
            if delta.status == "completed" and tool_indices:
                stop = "tool_use"
            elif delta.status == "completed":
                stop = "end_turn"
            elif delta.status == "incomplete":
                stop = "max_tokens"
            else:
                stop = "end_turn"
            yield _msg_event(
                "message_delta",
                {"delta": {"stop_reason": stop}, "usage": {"output_tokens": 0}},
            )
            yield _msg_event("message_stop", {})
            return


PARSERS = {
    "chat": parse_chat_sse,
    "responses": parse_responses_sse,
    "messages": parse_messages_sse,
}

EMITTERS = {
    "chat": emit_chat_sse,
    "responses": emit_responses_sse,
    "messages": emit_messages_sse,
}


def translate_lines(
    ingress: str, egress: str, lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    return EMITTERS[egress](PARSERS[ingress](lines), trace_id, model)


def responses_to_chat(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    return translate_lines("responses", "chat", lines, trace_id, model)


def chat_to_responses(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    return translate_lines("chat", "responses", lines, trace_id, model)


def responses_to_messages(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    return translate_lines("responses", "messages", lines, trace_id, model)


def messages_to_responses(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    return translate_lines("messages", "responses", lines, trace_id, model)


def chat_to_messages(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    return translate_lines("chat", "messages", lines, trace_id, model)


def messages_to_chat(
    lines: Iterable[str], trace_id: str, model: str
) -> Iterable[bytes]:
    return translate_lines("messages", "chat", lines, trace_id, model)
