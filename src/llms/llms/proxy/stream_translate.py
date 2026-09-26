from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator

from llms.proxy.ir import ReasoningDelta, StreamDone, TextDelta, ToolArgsDelta
from llms.proxy.logging import setup_logging

logger = setup_logging()


def split_events(lines: Iterable[str]) -> Iterable[str]:
    data: list[str] = []
    for line in lines:
        if line.strip() == "":
            if data:
                # SSE joins multi-line data payloads with newlines; each
                # JSON event here is single-line, so yield them one by one.
                yield from data
                data = []
            continue
        if line.startswith("data:"):
            data.append(line[len("data:") :].strip())
        elif line.startswith("event:"):
            if data:
                yield from data
                data = []
            # SSE event-type lines carry no payload; the parser keys on the
            # data payload's own "type" field instead.
            continue
        else:
            # Continuation of a multi-line data payload.
            data.append(line)
    if data:
        yield from data


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


class SseFramer:
    """Incremental split_events: feed raw lines, pull complete payloads."""

    def __init__(self) -> None:
        self._data: list[str] = []

    def feed(self, line: str) -> list[str]:
        out: list[str] = []
        if line.strip() == "":
            if self._data:
                out.extend(self._data)
                self._data = []
            return out
        if line.startswith("data:"):
            self._data.append(line[len("data:") :].strip())
        elif line.startswith("event:"):
            if self._data:
                out.extend(self._data)
                self._data = []
        else:
            self._data.append(line)
        return out

    def finish(self) -> list[str]:
        out, self._data = self._data, []
        return out


def _stream_tokens(
    usage: dict, *key_pairs: tuple[str, ...] | str
) -> tuple[int | None, int | None]:
    """Best-effort token ints from an SSE usage frame; absent stays None.

    Each position accepts one key or a tuple of aliases (first present wins),
    so tests can use short field names without changing parser behavior.
    """

    def _int(keys) -> int | None:
        names = (keys,) if isinstance(keys, str) else keys
        for key in names:
            if key in usage:
                try:
                    return int(usage[key])
                except (TypeError, ValueError):
                    return None
        return None

    in_keys, out_keys = key_pairs[0], key_pairs[1]
    return _int(in_keys), _int(out_keys)


class ChatParser:
    """Incremental parse_chat_sse: feed data payloads, take deltas, finish."""

    def __init__(self) -> None:
        self.index_to_id: dict[int, str] = {}
        self.names: dict[str, str] = {}
        self.saw_calls = False
        self.usage: dict | None = None
        self.pending_finish: str | None = None
        self._stopped = False
        self._done = False

    def feed_payload(
        self, payload: str
    ) -> list[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone]:
        deltas: list = []
        if self._stopped:
            return deltas
        if payload == "[DONE]":
            self._stopped = True
            return deltas
        if "inference-cost" in payload:
            return deltas
        event = _load(payload)
        if event is None:
            return deltas
        # Some chat upstreams emit a terminal usage-only chunk alongside (or
        # instead of) a finish_reason chunk; remember the latest for StreamDone.
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        for choice in event.get("choices", []) or []:
            delta = choice.get("delta", {})
            if delta.get("content"):
                deltas.append(TextDelta(delta["content"]))
            if delta.get("reasoning_content"):
                deltas.append(ReasoningDelta(delta["reasoning_content"]))
            for call in delta.get("tool_calls", []):
                index = int(call.get("index", 0))
                fn = call.get("function", {})
                if index not in self.index_to_id:
                    self.index_to_id[index] = call.get("id", f"call-{index}")
                    self.names[self.index_to_id[index]] = fn.get("name", "")
                call_id = self.index_to_id[index]
                if fn.get("name") and not self.names[call_id]:
                    self.names[call_id] = fn["name"]
                deltas.append(
                    ToolArgsDelta(call_id, self.names[call_id], fn.get("arguments", ""))
                )
                self.saw_calls = True
            finish = choice.get("finish_reason")
            if finish and self.pending_finish is None:
                # Don't emit yet: a usage-only chunk may follow the finish
                # chunk (OpenAI sends both). Keep consuming so the trailing
                # StreamDone below sees the final usage frame.
                self.pending_finish = finish
                if finish == "tool_calls":
                    self.saw_calls = True
                continue
        return deltas

    def finish(self) -> StreamDone | None:
        if self._done:
            return None
        self._done = True
        self._stopped = True
        in_tok, out_tok = (None, None)
        cached, reasoning = (None, None)
        if self.usage is not None:
            in_tok, out_tok = _stream_tokens(
                self.usage, ("prompt_tokens", "p"), ("completion_tokens", "c")
            )
            in_det = self.usage.get(
                "prompt_tokens_details", self.usage.get("p_det", {})
            )
            out_det = self.usage.get(
                "completion_tokens_details", self.usage.get("c_det", {})
            )
            if isinstance(in_det, dict):
                cached, _ = _stream_tokens(
                    in_det, ("cached_tokens", "cached", "c"), "__absent__"
                )
            if isinstance(out_det, dict):
                _, reasoning = _stream_tokens(
                    out_det, "__absent__", ("reasoning_tokens", "reasoning", "r")
                )
        if self.pending_finish in ("stop", "tool_calls"):
            status, calls = "completed", self.saw_calls
        elif self.pending_finish == "length":
            status, calls = "incomplete", self.saw_calls
        elif self.pending_finish is not None:
            status, calls = "failed", self.saw_calls
        else:
            status, calls = "completed", self.saw_calls
        return StreamDone(
            status,
            has_tool_calls=calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
        )


def parse_chat_sse(
    lines: Iterable[str],
) -> Iterator[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone]:
    framer = SseFramer()
    parser = ChatParser()
    for line in lines:
        for payload in framer.feed(line):
            yield from parser.feed_payload(payload)
    for payload in framer.finish():
        yield from parser.feed_payload(payload)
    done = parser.finish()
    if done is not None:
        yield done


class ResponsesParser:
    """Incremental parse_responses_sse: feed data payloads, take deltas, finish."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.saw_calls = False
        self._done = False

    def feed_payload(
        self, payload: str
    ) -> list[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone]:
        deltas: list = []
        if self._done:
            return deltas
        if skip_payload(payload):
            return deltas
        event = _load(payload)
        if event is None:
            return deltas
        kind = event.get("type", "")
        if kind == "response.output_item.added" and isinstance(event.get("item"), dict):
            item = event["item"]
            if item.get("type") == "function_call":
                self.names[item.get("id", "")] = item.get("name", "")
            return deltas
        if kind == "response.output_text.delta":
            deltas.append(TextDelta(event.get("delta", "")))
            return deltas
        if kind == "response.reasoning_text.delta":
            deltas.append(ReasoningDelta(event.get("delta", "")))
            return deltas
        if kind == "response.function_call_arguments.delta":
            item_id = event.get("item_id", "")
            self.saw_calls = True
            deltas.append(
                ToolArgsDelta(
                    item_id, self.names.get(item_id, ""), event.get("delta", "")
                )
            )
            return deltas
        if kind in ("response.completed", "response.failed", "response.incomplete"):
            status = {
                "response.completed": "completed",
                "response.failed": "failed",
            }.get(kind, "incomplete")
            usage = event.get("response", {}).get("usage", {})
            in_tok, out_tok = _stream_tokens(
                usage, ("input_tokens", "in"), ("output_tokens", "o")
            )
            in_det = usage.get("input_tokens_details", usage.get("in_d", {}))
            out_det = usage.get("output_tokens_details", usage.get("o_d", {}))
            cached, reasoning = None, None
            if isinstance(in_det, dict):
                cached, _ = _stream_tokens(
                    in_det, ("cached_tokens", "cached", "c"), "__absent__"
                )
            if isinstance(out_det, dict):
                _, reasoning = _stream_tokens(
                    out_det, "__absent__", ("reasoning_tokens", "reasoning", "r")
                )
            deltas.append(
                StreamDone(
                    status,
                    has_tool_calls=self.saw_calls or bool(self.names),
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cached_tokens=cached,
                    reasoning_tokens=reasoning,
                )
            )
            self._done = True
            return deltas
        return deltas

    def finish(self) -> StreamDone | None:
        if self._done:
            return None
        self._done = True
        return StreamDone(
            "completed", has_tool_calls=self.saw_calls or bool(self.names)
        )


def parse_responses_sse(
    lines: Iterable[str],
) -> Iterator[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone]:
    framer = SseFramer()
    parser = ResponsesParser()
    for line in lines:
        for payload in framer.feed(line):
            yield from parser.feed_payload(payload)
    for payload in framer.finish():
        yield from parser.feed_payload(payload)
    done = parser.finish()
    if done is not None:
        yield done


class MessagesParser:
    """Incremental parse_messages_sse: feed data payloads, take deltas, finish."""

    def __init__(self) -> None:
        self.ids: dict[int, str] = {}
        self.names: dict[int, str] = {}
        self.stop = "end_turn"
        self.in_tok: int | None = None
        self.out_tok: int | None = None
        self.cached_tok: int | None = None
        self._done = False

    def feed_payload(
        self, payload: str
    ) -> list[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone]:
        deltas: list = []
        if self._done:
            return deltas
        if skip_payload(payload):
            return deltas
        event = _load(payload)
        if event is None:
            return deltas
        kind = event.get("type", "")
        if kind == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                self.ids[event.get("index", 0)] = block.get("id", "")
                self.names[event.get("index", 0)] = block.get("name", "")
            return deltas
        if kind == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                deltas.append(TextDelta(delta.get("text", "")))
            elif delta.get("type") == "input_json_delta":
                index = event.get("index", 0)
                deltas.append(
                    ToolArgsDelta(
                        self.ids.get(index, ""),
                        self.names.get(index, ""),
                        delta.get("partial_json", ""),
                    )
                )
            elif delta.get("type") == "thinking_delta":
                deltas.append(ReasoningDelta(delta.get("thinking", "")))
            return deltas
        if kind == "message_delta":
            self.stop = event.get("delta", {}).get("stop_reason", self.stop)
            usage = event.get("usage", {})
            frame_in, frame_out = _stream_tokens(
                usage, ("input_tokens", "in"), ("output_tokens", "o")
            )
            if frame_in is not None:
                self.in_tok = frame_in
            if frame_out is not None:
                self.out_tok = frame_out
            frame_cached, _ = _stream_tokens(
                usage, ("cache_read_input_tokens", "cache_read", "cr"), "__absent__"
            )
            if frame_cached is not None:
                self.cached_tok = frame_cached
            return deltas
        if kind == "message_stop":
            deltas.append(
                StreamDone(
                    "completed"
                    if self.stop in ("end_turn", "tool_use", "stop_sequence")
                    else "incomplete"
                    if self.stop == "max_tokens"
                    else "failed",
                    input_tokens=self.in_tok,
                    output_tokens=self.out_tok,
                    cached_tokens=self.cached_tok,
                )
            )
            self._done = True
            return deltas
        return deltas

    def finish(self) -> StreamDone | None:
        if self._done:
            return None
        self._done = True
        return StreamDone("completed")


def parse_messages_sse(
    lines: Iterable[str],
) -> Iterator[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone]:
    framer = SseFramer()
    parser = MessagesParser()
    for line in lines:
        for payload in framer.feed(line):
            yield from parser.feed_payload(payload)
    for payload in framer.finish():
        yield from parser.feed_payload(payload)
    done = parser.finish()
    if done is not None:
        yield done


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


class ChatEmitter:
    """Incremental emit_chat_sse: feed deltas, take event bytes."""

    def __init__(self, trace_id: str, model: str) -> None:
        self.chat_id = new_chat_id(trace_id)
        self.model = model
        self.indices: dict[str, int] = {}
        self.announced: set[str] = set()

    def feed_delta(
        self, delta: TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone
    ) -> list[bytes]:
        if isinstance(delta, TextDelta):
            return [
                _chat_chunk(self.chat_id, self.model, {"content": delta.text}, None)
            ]
        if isinstance(delta, ToolArgsDelta):
            out: list[bytes] = []
            if delta.call_id not in self.indices:
                self.indices[delta.call_id] = len(self.indices)
            if delta.call_id not in self.announced:
                self.announced.add(delta.call_id)
                out.append(
                    _chat_chunk(
                        self.chat_id,
                        self.model,
                        {
                            "tool_calls": [
                                {
                                    "index": self.indices[delta.call_id],
                                    "id": delta.call_id,
                                    "type": "function",
                                    "function": {"name": delta.name, "arguments": ""},
                                }
                            ]
                        },
                        None,
                    )
                )
            if delta.args_chunk:
                out.append(
                    _chat_chunk(
                        self.chat_id,
                        self.model,
                        {
                            "tool_calls": [
                                {
                                    "index": self.indices[delta.call_id],
                                    "function": {"arguments": delta.args_chunk},
                                }
                            ]
                        },
                        None,
                    )
                )
            return out
        if isinstance(delta, StreamDone):
            if delta.status == "completed" and delta.has_tool_calls:
                finish = "tool_calls"
            elif delta.status == "completed":
                finish = "stop"
            elif delta.status == "incomplete":
                finish = "length"
            else:
                finish = "stop"
            return [
                _chat_chunk(self.chat_id, self.model, {}, finish),
                b"data: [DONE]\n\n",
            ]
        return []


def emit_chat_sse(
    deltas: Iterable[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone],
    trace_id: str,
    model: str,
) -> Iterable[bytes]:
    emitter = ChatEmitter(trace_id, model)
    for delta in deltas:
        yield from emitter.feed_delta(delta)


class ResponsesEmitter:
    """Incremental emit_responses_sse: feed deltas, take event bytes."""

    def __init__(self, trace_id: str, model: str) -> None:
        self.resp_id = new_resp_id(trace_id)
        self.model = model
        self.started = False
        self.text_open = False
        self.text_accum = ""
        self.reasoning_open = False
        self.reasoning_accum = ""
        self.tool_items: dict[str, int] = {}
        self.tool_names: dict[str, str] = {}
        self.tool_args: dict[str, str] = {}

    def _begin_chunks(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        return [
            _resp_event(
                {
                    "type": "response.created",
                    "response": {
                        "id": self.resp_id,
                        "status": "in_progress",
                        "model": self.model,
                    },
                }
            ),
            _resp_event(
                {"type": "response.in_progress", "response": {"id": self.resp_id}}
            ),
        ]

    def feed_delta(
        self, delta: TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone
    ) -> list[bytes]:
        out = self._begin_chunks()
        if isinstance(delta, TextDelta):
            self.text_accum += delta.text
            if not self.text_open:
                self.text_open = True
                out.append(
                    _resp_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": f"{self.resp_id}-msg",
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                            },
                        }
                    )
                )
                out.append(
                    _resp_event(
                        {
                            "type": "response.content_part.added",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                                "annotations": [],
                            },
                        }
                    )
                )
            out.append(
                _resp_event(
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": delta.text,
                    }
                )
            )
            return out
        if isinstance(delta, ToolArgsDelta):
            if delta.call_id not in self.tool_items:
                self.tool_items[delta.call_id] = len(self.tool_items) + 1
                self.tool_names[delta.call_id] = delta.name
                self.tool_args[delta.call_id] = ""
                out.append(
                    _resp_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": self.tool_items[delta.call_id],
                            "item": {
                                "id": delta.call_id,
                                "type": "function_call",
                                "name": delta.name,
                                "arguments": "",
                            },
                        }
                    )
                )
            if delta.args_chunk:
                self.tool_args[delta.call_id] += delta.args_chunk
                out.append(
                    _resp_event(
                        {
                            "type": "response.function_call_arguments.delta",
                            "output_index": self.tool_items[delta.call_id],
                            "item_id": delta.call_id,
                            "delta": delta.args_chunk,
                        }
                    )
                )
            return out
        if isinstance(delta, ReasoningDelta):
            if not self.reasoning_open:
                self.reasoning_open = True
                out.append(
                    _resp_event(
                        {
                            "type": "response.output_item.added",
                            "output_index": 0,
                            "item": {
                                "id": f"{self.resp_id}-reasoning",
                                "type": "reasoning",
                                "summary": [],
                            },
                        }
                    )
                )
            if delta.text:
                self.reasoning_accum += delta.text
                out.append(
                    _resp_event(
                        {
                            "type": "response.reasoning_text.delta",
                            "output_index": 0,
                            "delta": delta.text,
                        }
                    )
                )
            return out
        if isinstance(delta, StreamDone):
            if self.reasoning_open:
                out.append(
                    _resp_event(
                        {
                            "type": "response.reasoning_text.done",
                            "output_index": 0,
                            "text": self.reasoning_accum,
                        }
                    )
                )
                out.append(
                    _resp_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "id": f"{self.resp_id}-reasoning",
                                "type": "reasoning",
                                "summary": [],
                            },
                        }
                    )
                )
            if self.text_open:
                out.append(
                    _resp_event(
                        {
                            "type": "response.output_text.done",
                            "output_index": 0,
                            "content_index": 0,
                            "text": self.text_accum,
                        }
                    )
                )
                out.append(
                    _resp_event(
                        {
                            "type": "response.content_part.done",
                            "output_index": 0,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": self.text_accum,
                                "annotations": [],
                            },
                        }
                    )
                )
                out.append(
                    _resp_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": 0,
                            "item": {
                                "id": f"{self.resp_id}-msg",
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                            },
                        }
                    )
                )
            for call_id, index in self.tool_items.items():
                out.append(
                    _resp_event(
                        {
                            "type": "response.function_call_arguments.done",
                            "output_index": index,
                            "item_id": call_id,
                            "arguments": self.tool_args[call_id],
                        }
                    )
                )
                out.append(
                    _resp_event(
                        {
                            "type": "response.output_item.done",
                            "output_index": index,
                            "item": {
                                "id": call_id,
                                "type": "function_call",
                                "name": self.tool_names[call_id],
                                "arguments": self.tool_args[call_id],
                            },
                        }
                    )
                )
            in_tok = delta.input_tokens or 0
            out_tok = delta.output_tokens or 0
            out.append(
                _resp_event(
                    {
                        "type": f"response.{delta.status}",
                        "response": {
                            "id": self.resp_id,
                            "status": delta.status,
                            "model": self.model,
                            "usage": {
                                "input_tokens": in_tok,
                                "output_tokens": out_tok,
                                "total_tokens": in_tok + out_tok,
                            },
                        },
                    }
                )
            )
            return out
        return out


def emit_responses_sse(
    deltas: Iterable[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone],
    trace_id: str,
    model: str,
) -> Iterable[bytes]:
    emitter = ResponsesEmitter(trace_id, model)
    for delta in deltas:
        yield from emitter.feed_delta(delta)


class MessagesEmitter:
    """Incremental emit_messages_sse: feed deltas, take event bytes."""

    def __init__(self, trace_id: str, model: str) -> None:
        self.msg_id = new_msg_id(trace_id)
        self.model = model
        self.started = False
        self.text_index: int | None = None
        self.thinking_index: int | None = None
        self.tool_indices: dict[str, int] = {}
        self.next_index = 0

    def _begin_chunks(self) -> list[bytes]:
        if self.started:
            return []
        self.started = True
        return [
            _msg_event(
                "message_start",
                {
                    "message": {
                        "id": self.msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                },
            )
        ]

    def feed_delta(
        self, delta: TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone
    ) -> list[bytes]:
        out = self._begin_chunks()
        if isinstance(delta, TextDelta):
            if self.text_index is None:
                self.text_index = self.next_index
                self.next_index += 1
                out.append(
                    _msg_event(
                        "content_block_start",
                        {
                            "index": self.text_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                )
            out.append(
                _msg_event(
                    "content_block_delta",
                    {
                        "index": self.text_index,
                        "delta": {"type": "text_delta", "text": delta.text},
                    },
                )
            )
            return out
        if isinstance(delta, ToolArgsDelta):
            if delta.call_id not in self.tool_indices:
                self.tool_indices[delta.call_id] = self.next_index
                self.next_index += 1
                out.append(
                    _msg_event(
                        "content_block_start",
                        {
                            "index": self.tool_indices[delta.call_id],
                            "content_block": {
                                "type": "tool_use",
                                "id": delta.call_id,
                                "name": delta.name,
                                "input": {},
                            },
                        },
                    )
                )
            if delta.args_chunk:
                out.append(
                    _msg_event(
                        "content_block_delta",
                        {
                            "index": self.tool_indices[delta.call_id],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": delta.args_chunk,
                            },
                        },
                    )
                )
            return out
        if isinstance(delta, ReasoningDelta):
            if self.thinking_index is None:
                self.thinking_index = self.next_index
                self.next_index += 1
                out.append(
                    _msg_event(
                        "content_block_start",
                        {
                            "index": self.thinking_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        },
                    )
                )
            if delta.text:
                out.append(
                    _msg_event(
                        "content_block_delta",
                        {
                            "index": self.thinking_index,
                            "delta": {"type": "thinking_delta", "thinking": delta.text},
                        },
                    )
                )
            return out
        if isinstance(delta, StreamDone):
            if self.text_index is not None:
                out.append(_msg_event("content_block_stop", {"index": self.text_index}))
            if self.thinking_index is not None:
                out.append(
                    _msg_event("content_block_stop", {"index": self.thinking_index})
                )
            for index in self.tool_indices.values():
                out.append(_msg_event("content_block_stop", {"index": index}))
            if delta.status == "completed" and self.tool_indices:
                stop = "tool_use"
            elif delta.status == "completed":
                stop = "end_turn"
            elif delta.status == "incomplete":
                stop = "max_tokens"
            else:
                stop = "end_turn"
            out.append(
                _msg_event(
                    "message_delta",
                    {
                        "delta": {"stop_reason": stop},
                        "usage": {
                            "input_tokens": delta.input_tokens or 0,
                            "output_tokens": delta.output_tokens or 0,
                        },
                    },
                )
            )
            out.append(_msg_event("message_stop", {}))
            return out
        return out


def emit_messages_sse(
    deltas: Iterable[TextDelta | ToolArgsDelta | ReasoningDelta | StreamDone],
    trace_id: str,
    model: str,
) -> Iterable[bytes]:
    emitter = MessagesEmitter(trace_id, model)
    for delta in deltas:
        yield from emitter.feed_delta(delta)


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


STREAM_PARSERS = {
    "chat": ChatParser,
    "responses": ResponsesParser,
    "messages": MessagesParser,
}

STREAM_EMITTERS = {
    "chat": ChatEmitter,
    "responses": ResponsesEmitter,
    "messages": MessagesEmitter,
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
