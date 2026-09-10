from __future__ import annotations

import json

from proxy.stream_translate import chat_to_responses, responses_to_chat


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
    events = collect(
        responses_to_chat(['data: {"type":"response.failed"}', ""], "t1", "m")
    )
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    raw = b"".join(
        responses_to_chat(['data: {"type":"response.failed"}', ""], "t1", "m")
    ).decode()
    assert "data: [DONE]" in raw


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
    assert done[0]["arguments"] == '{"x"'
    assert events[-1]["type"] == "response.completed"


def test_chat_stream_without_done_still_completes():
    lines = ['data: {"choices":[{"delta":{"content":"hi"}}]}', ""]
    events = collect(chat_to_responses(lines, "t1", "m"))
    assert events[-1]["type"] == "response.completed"
