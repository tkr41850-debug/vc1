from __future__ import annotations

from llms.proxy.translate_response import (
    chat_to_responses,
    messages_to_responses,
    responses_to_chat,
    responses_to_messages,
)

RESP_PAYLOAD = {
    "id": "resp_abc",
    "object": "response",
    "status": "completed",
    "model": "muse-spark-1.3-contributor-free",
    "output": [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
}

CHAT_PAYLOAD = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "mimo-v2.5-free",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def test_responses_to_chat_text():
    out = responses_to_chat(RESP_PAYLOAD, "muse-spark-1.3-contributor-free")
    assert out["id"] == "chatcmpl-abc"
    assert out["model"] == "muse-spark-1.3-contributor-free"
    choice = out["choices"][0]
    assert choice["message"]["content"] == "hello"
    assert choice["message"]["tool_calls"] is None
    assert choice["finish_reason"] == "stop"
    assert out["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 0},
    }


def test_responses_to_chat_tool_calls_finish():
    payload = dict(
        RESP_PAYLOAD,
        output=[
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "bash",
                "arguments": '{"command":"ls"}',
            }
        ],
    )
    out = responses_to_chat(payload, "m")
    choice = out["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
        }
    ]


def test_responses_to_chat_incomplete_maps_length():
    out = responses_to_chat(dict(RESP_PAYLOAD, status="incomplete", output=[]), "m")
    assert out["choices"][0]["finish_reason"] == "length"


def test_chat_to_responses_text():
    out = chat_to_responses(CHAT_PAYLOAD, "mimo-v2.5-free")
    assert out["id"] == "resp_abc"
    assert out["status"] == "completed"
    assert out["model"] == "mimo-v2.5-free"
    assert out["output"] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello", "annotations": []}],
        }
    ]
    assert out["usage"] == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


def test_chat_to_responses_tool_calls():
    payload = {
        "id": "chatcmpl-x",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c9",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
    }
    out = chat_to_responses(payload, "m")
    assert out["status"] == "completed"
    assert out["output"] == [
        {"type": "function_call", "call_id": "c9", "name": "bash", "arguments": "{}"}
    ]


def test_chat_to_responses_length_maps_incomplete():
    payload = dict(
        CHAT_PAYLOAD,
        choices=[
            {
                "message": {"role": "assistant", "content": "par"},
                "finish_reason": "length",
            }
        ],
    )
    out = chat_to_responses(payload, "m")
    assert out["status"] == "incomplete"


def test_response_round_trip_stable():
    assert (
        responses_to_chat(chat_to_responses(CHAT_PAYLOAD, "m"), "m")["choices"][0][
            "message"
        ]["content"]
        == "hello"
    )


def test_usage_details_survive_translation():
    chat = {
        "id": "chatcmpl-u",
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }
    out = chat_to_responses(chat, "m")
    assert out["usage"]["input_tokens_details"] == {"cached_tokens": 4}
    assert out["usage"]["output_tokens_details"] == {"reasoning_tokens": 2}
    back = responses_to_chat(out, "m")
    assert back["usage"]["prompt_tokens_details"] == {"cached_tokens": 4}
    assert back["usage"]["completion_tokens_details"] == {"reasoning_tokens": 2}


def test_incomplete_reason_survives_translation():
    chat = {
        "id": "chatcmpl-t",
        "choices": [
            {
                "message": {"role": "assistant", "content": "par"},
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18},
    }
    out = chat_to_responses(chat, "m")
    assert out["status"] == "incomplete"
    assert out["incomplete_details"] == {"reason": "max_output_tokens"}
    back = responses_to_chat(out, "m")
    assert back["choices"][0]["finish_reason"] == "length"


def test_incomplete_reason_normalized_through_messages():
    payload = {
        "id": "resp-t",
        "status": "incomplete",
        "output": [],
        "incomplete_details": {"reason": "content_filter"},
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    out = messages_to_responses(responses_to_messages(payload, "m"), "m")
    assert out["status"] == "incomplete"
    assert out["incomplete_details"] == {"reason": "max_output_tokens"}


def test_deltas_fold_into_response_ir():
    from llms.proxy.ir import ReasoningDelta, StreamDone, TextDelta, ToolArgsDelta
    from llms.proxy.translate_response import deltas_to_response_ir

    rir = deltas_to_response_ir(
        [
            TextDelta("hel"),
            TextDelta("lo"),
            ToolArgsDelta("c1", "bash", '{"cmd":'),
            ToolArgsDelta("c1", "bash", '"echo hi"}'),
            ReasoningDelta("think"),
            StreamDone("completed", input_tokens=4, output_tokens=2),
        ],
        "m",
    )
    assert rir.status == "completed"
    assert (rir.input_tokens, rir.output_tokens) == (4, 2)
    kinds = [type(b).__name__ for b in rir.messages[0].blocks]
    assert kinds == ["ThinkingBlock", "TextBlock", "ToolCallBlock"]
    call = rir.messages[0].blocks[2]
    assert (call.call_id, call.name) == ("c1", "bash")
    assert "echo hi" in call.arguments
