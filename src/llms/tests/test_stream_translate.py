from __future__ import annotations

import json

from llms.proxy.stream_translate import (
    chat_to_responses,
    messages_to_responses,
    responses_to_chat,
    responses_to_messages,
)


def collect(chunks) -> list[dict]:
    out = []
    for raw in chunks:
        text = raw.decode()
        for part in text.split("data:"):
            part = part.strip()
            if not part or part == "[DONE]":
                continue
            try:
                out.append(json.loads(part))
            except Exception:
                continue
    return out


RESP_TEXT_STREAM = [
    'data: {"type":"response.created"}',
    "",
    'data: {"type":"response.output_text.delta","delta":"hel"}',
    "",
    'data: {"type":"response.output_text.delta","delta":"lo"}',
    "",
    'data: {"inference-cost":42}',
    "",
    'data: {"type":"response.completed"}',
    "",
]

RESP_TOOL_STREAM = [
    'data: {"type":"response.output_item.added","item":{"id":"c1","type":"function_call","name":"bash"}}',
    "",
    'data: {"type":"response.function_call_arguments.delta","item_id":"c1","delta":"{\\"a\\""}',
    "",
    'data: {"type":"response.function_call_arguments.delta","item_id":"c1","delta":"}"}',
    "",
    'data: {"type":"response.completed"}',
    "",
]

CHAT_TEXT_STREAM = [
    'data: {"choices":[{"delta":{"role":"assistant","content":"hel"}}]}',
    "",
    'data: {"choices":[{"delta":{"content":"lo"}}]}',
    "",
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "",
    "data: [DONE]",
    "",
]

CHAT_TOOL_STREAM = [
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c9","type":"function","function":{"name":"bash","arguments":""}}]}}]}',
    "",
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"x\\""}}]}}]}',
    "",
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
    "",
    "data: [DONE]",
    "",
]


def test_responses_text_to_chat_chunks_and_done():
    events = collect(responses_to_chat(RESP_TEXT_STREAM, "t1", "m"))
    contents = [
        c["choices"][0]["delta"].get("content", "")
        for c in events
        if c.get("object") == "chat.completion.chunk"
    ]
    assert "".join(contents) == "hello"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    raw = b"".join(responses_to_chat(RESP_TEXT_STREAM, "t1", "m")).decode()
    assert "data: [DONE]" in raw
    assert "inference-cost" not in raw


def test_responses_tool_deltas_become_tool_calls():
    events = collect(responses_to_chat(RESP_TOOL_STREAM, "t1", "m"))
    calls = [
        c["choices"][0]["delta"]["tool_calls"]
        for c in events
        if "tool_calls" in c["choices"][0]["delta"]
    ]
    assert calls[0][0]["function"]["name"] == "bash"
    assert "".join(c[0]["function"].get("arguments", "") for c in calls[1:]) == '{"a"}'
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_responses_failed_still_terminates():
    raw = b"".join(
        responses_to_chat(['data: {"type":"response.failed"}', ""], "t1", "m")
    ).decode()
    # Upstream failure must surface as an error, never finish_reason stop
    # (which an OpenAI SDK client reads as success). The stream still
    # terminates with [DONE] so clients don't hang on a failed leg.
    assert '"object": "error"' in raw
    assert '"type": "upstream_failed"' in raw
    assert "data: [DONE]" in raw
    events = collect(
        responses_to_chat(['data: {"type":"response.failed"}', ""], "t1", "m")
    )
    assert all("choices" not in e for e in events)


def test_chat_text_to_responses_events():
    events = collect(chat_to_responses(CHAT_TEXT_STREAM, "t1", "m"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "response.created"
    deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
    assert "".join(deltas) == "hello"
    assert kinds[-1] == "response.completed"
    raw = b"".join(chat_to_responses(CHAT_TEXT_STREAM, "t1", "m")).decode()
    assert "[DONE]" not in raw


def test_chat_tool_deltas_become_function_call():
    events = collect(chat_to_responses(CHAT_TOOL_STREAM, "t1", "m"))
    added = [e for e in events if e["type"] == "response.output_item.added"]
    assert added[0]["item"]["name"] == "bash"
    done = [e for e in events if e["type"] == "response.function_call_arguments.done"]
    # Truncated mid-args stream: terminal coerces the partial to valid
    # JSON so downstream validation never sees a fragment.
    assert done[0]["arguments"] == "{}"
    assert events[-1]["type"] == "response.completed"


def test_chat_stream_without_done_still_completes():
    lines = ['data: {"choices":[{"delta":{"content":"hi"}}]}', ""]
    events = collect(chat_to_responses(lines, "t1", "m"))
    # EOF with no terminal frame is a cut stream, not a clean completion:
    # the terminal reads incomplete so truncation never bills as success.
    assert events[-1]["type"] == "response.incomplete"


def test_usage_flows_responses_to_messages_stream():
    lines = [
        'data: {"type":"response.output_text.delta","delta":"hi"}',
        "",
        'data: {"type":"response.completed","response":{"id":"r","status":"completed","usage":{"input_tokens":7,"output_tokens":3}}}',
        "",
    ]
    events = collect(responses_to_messages(lines, "t1", "m"))
    delta = next(e for e in events if e["type"] == "message_delta")
    # Full Anthropic shape: cache fields present (0 when upstream omits
    # them) so statusline-style clients summing all three input fields
    # keep working after compaction handoffs.
    assert delta["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def test_usage_flows_messages_to_responses_stream():
    lines = [
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":7,"output_tokens":3}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]
    events = collect(messages_to_responses(lines, "t1", "m"))
    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


def test_reasoning_deltas_flow_to_messages_thinking():
    lines = [
        'data: {"type":"response.reasoning_text.delta","delta":"ponder"}',
        "",
        'data: {"type":"response.completed"}',
        "",
    ]
    events = collect(responses_to_messages(lines, "t1", "m"))
    kinds = [e["type"] for e in events]
    assert "content_block_start" in kinds
    deltas = [e for e in events if e["type"] == "content_block_delta"]
    assert deltas[0]["delta"] == {"type": "thinking_delta", "thinking": "ponder"}
    assert kinds[-1] == "message_stop"


def test_thinking_deltas_flow_to_responses_reasoning():
    lines = [
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}',
        "",
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]
    events = collect(messages_to_responses(lines, "t1", "m"))
    kinds = [e["type"] for e in events]
    assert "response.reasoning_text.delta" in kinds
    assert kinds[-1] == "response.completed"


def test_chat_reasoning_content_delta_parses():
    from llms.proxy.stream_translate import parse_chat_sse

    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"deep"}}]}',
        "",
        "data: [DONE]",
        "",
    ]
    kinds = [type(d).__name__ for d in parse_chat_sse(lines)]
    assert kinds[0] == "ReasoningDelta"
    assert kinds[-1] == "StreamDone"


MSG_TEXT_STREAM = [
    "event: message_start",
    'data: {"type":"message_start","message":{"id":"msg-1"}}',
    "",
    "event: content_block_start",
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "",
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hel"}}',
    "",
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}',
    "",
    "event: content_block_stop",
    'data: {"type":"content_block_stop","index":0}',
    "",
    "event: message_delta",
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
    "",
    "event: message_stop",
    'data: {"type":"message_stop"}',
    "",
]

MSG_TOOL_STREAM = [
    "event: content_block_start",
    'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t9","name":"bash","input":{}}}',
    "",
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"c\\""}}',
    "",
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"}"}}',
    "",
    "event: message_delta",
    'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"}}',
    "",
    "event: message_stop",
    'data: {"type":"message_stop"}',
    "",
]


def test_messages_text_to_chat_chunks_and_done():
    from llms.proxy.stream_translate import messages_to_chat

    events = collect(messages_to_chat(MSG_TEXT_STREAM, "t1", "m"))
    contents = [
        c["choices"][0]["delta"].get("content", "") for c in events if "choices" in c
    ]
    assert "".join(contents) == "hello"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    raw = b"".join(messages_to_chat(MSG_TEXT_STREAM, "t1", "m")).decode()
    assert "data: [DONE]" in raw


def test_messages_tool_to_responses_function_call():
    events = collect(messages_to_responses(MSG_TOOL_STREAM, "t1", "m"))
    added = [e for e in events if e["type"] == "response.output_item.added"]
    assert added[0]["item"]["name"] == "bash"
    done = [e for e in events if e["type"] == "response.function_call_arguments.done"]
    # Same truncated-stream coercion on the messages leg.
    assert done[0]["arguments"] == "{}"
    assert events[-1]["type"] == "response.completed"


def test_complete_tool_args_pass_through_terminal():
    from llms.proxy.ir import StreamDone, ToolArgsDelta
    from llms.proxy.stream_translate import ResponsesEmitter

    emitter = ResponsesEmitter("t1", "m")
    emitter.feed_delta(ToolArgsDelta("c1", "bash", '{"cmd":"ls"}'))
    out = emitter.feed_delta(StreamDone("completed"))
    done = [
        e
        for e in (
            __import__("json").loads(c.decode().split("data: ", 1)[1])
            for c in out
            if c.startswith(b"data: ")
        )
        if e.get("type") == "response.function_call_arguments.done"
    ]
    assert done[0]["arguments"] == '{"cmd":"ls"}'


def test_responses_text_to_messages_events():
    events = collect(responses_to_messages(RESP_TEXT_STREAM, "t1", "m"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "message_start"
    deltas = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
    assert "".join(deltas) == "hello"
    assert kinds[-1] == "message_stop"
    stop = next(e for e in events if e["type"] == "message_delta")
    assert stop["delta"]["stop_reason"] == "end_turn"


def test_messages_tool_stream_emits_tool_calls_finish():
    from llms.proxy.stream_translate import messages_to_chat

    events = collect(messages_to_chat(MSG_TOOL_STREAM, "t1", "m"))
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_messages_reasoning_flows_to_chat_reasoning_content():
    from llms.proxy.stream_translate import messages_to_chat

    lines = [
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}',
        "",
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]
    events = collect(messages_to_chat(lines, "t1", "m"))
    deltas = [c["choices"][0]["delta"] for c in events if "choices" in c]
    assert any(d.get("reasoning_content") == "hmm" for d in deltas)


def test_usage_details_flow_messages_to_responses_stream():
    lines = [
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":7,"output_tokens":3,"cache_read_input_tokens":2}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]
    events = collect(messages_to_responses(lines, "t1", "m"))
    usage = events[-1]["response"]["usage"]
    assert usage["input_tokens_details"] == {"cached_tokens": 2}


def test_messages_terminal_carries_cache_read_tokens():
    """Terminal message_delta exposes cache_read_input_tokens downstream.

    Regression: the emitter dropped the cached count, so statusline-style
    clients summing input + cache_read + cache_creation under-reported
    context after prompt-cache warmup (reads as 0 tokens post-compaction).
    """
    from llms.proxy.stream_translate import MessagesEmitter

    emitter = MessagesEmitter("t-cache", "m")
    from llms.proxy.ir import StreamDone

    out = emitter.feed_delta(
        StreamDone(
            "completed",
            has_tool_calls=False,
            input_tokens=100,
            output_tokens=5,
            cached_tokens=80,
        )
    )
    import json as _json

    delta = next(
        _json.loads(b.decode().split("data: ", 1)[1])
        for b in out
        if b.startswith(b"event: message_delta")
    )
    assert delta["usage"]["input_tokens"] == 100
    assert delta["usage"]["cache_read_input_tokens"] == 80
    assert delta["usage"]["cache_creation_input_tokens"] == 0


def test_truncated_chat_stream_is_incomplete_not_completed():
    """EOF with no finish_reason must not synthesize a clean completion."""
    from llms.proxy.stream_translate import ChatParser

    p = ChatParser()
    for payload in [
        '{"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}'
    ]:
        p.feed_payload(payload)
    done = p.finish()
    assert done.status == "incomplete"


def test_truncated_responses_stream_is_incomplete():
    from llms.proxy.stream_translate import ResponsesParser

    p = ResponsesParser()
    p.feed_payload('{"type":"response.output_text.delta","delta":"hi"}')
    assert p.finish().status == "incomplete"


def test_truncated_messages_stream_is_incomplete():
    from llms.proxy.stream_translate import MessagesParser

    p = MessagesParser()
    p.feed_payload(
        '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}'
    )
    assert p.finish().status == "incomplete"


def test_failed_chat_terminal_surfaces_error_not_stop():
    """response.failed through the chat emitter: error object, never stop."""
    raw = b"".join(
        responses_to_chat(['data: {"type":"response.failed"}', ""], "t1", "m")
    ).decode()
    assert '"type": "upstream_failed"' in raw
    assert '"finish_reason": "stop"' not in raw


def test_failed_messages_terminal_surfaces_error_stop():
    """response.failed through the messages emitter: error, never end_turn."""
    from llms.proxy.stream_translate import responses_to_messages

    raw = b"".join(
        responses_to_messages(['data: {"type":"response.failed"}', ""], "t1", "m")
    ).decode()
    assert '"stop_reason": "error"' in raw
    assert '"stop_reason": "end_turn"' not in raw


def test_unknown_failure_reason_preserved_on_chat_response():
    """Non-streaming chat leg preserves e.g. finish_reason error verbatim."""
    from llms.proxy.translate_response import convert_response

    payload = {
        "id": "chatcmpl-x",
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant"}, "finish_reason": "error"}
        ],
        "usage": {},
    }
    out = convert_response("chat", "chat", payload, "m")
    assert out["choices"][0]["finish_reason"] == "error"


def test_unknown_failure_reason_surfaces_error_on_messages_response():
    from llms.proxy.translate_response import convert_response

    payload = {
        "id": "chatcmpl-x",
        "model": "m",
        "choices": [
            {"index": 0, "message": {"role": "assistant"}, "finish_reason": "error"}
        ],
        "usage": {},
    }
    out = convert_response("chat", "messages", payload, "m")
    assert out["stop_reason"] == "error"
