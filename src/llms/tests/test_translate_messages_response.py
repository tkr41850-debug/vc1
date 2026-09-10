from __future__ import annotations

from llms.proxy.translate_response import messages_to_responses, responses_to_messages

RESP_WITH_CALL = {
    "id": "resp_1",
    "status": "completed",
    "model": "muse-spark-1.3-contributor-free",
    "output": [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "bash",
            "arguments": '{"command":"ls"}',
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}

MSG_WITH_TOOL = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [
        {"type": "text", "text": "run it"},
        {"type": "tool_use", "id": "t1", "name": "bash", "input": {"c": "ls"}},
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}


def test_responses_to_messages_tool_use():
    out = responses_to_messages(RESP_WITH_CALL, "m")
    assert out["id"] == "msg_1"
    assert out["role"] == "assistant"
    assert out["stop_reason"] == "tool_use"
    assert out["content"] == [
        {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "ls"}}
    ]
    assert out["usage"] == {
        "input_tokens": 5,
        "output_tokens": 3,
        "cache_read_input_tokens": 0,
    }


def test_responses_to_messages_text_end_turn():
    payload = {
        "id": "resp-2",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }
    out = responses_to_messages(payload, "m")
    assert out["content"] == [{"type": "text", "text": "hi"}]
    assert out["stop_reason"] == "end_turn"


def test_responses_to_messages_incomplete_max_tokens():
    out = responses_to_messages({"id": "r", "status": "incomplete", "output": []}, "m")
    assert out["stop_reason"] == "max_tokens"


def test_messages_to_responses_tool_use():
    out = messages_to_responses(MSG_WITH_TOOL, "m")
    assert out["id"] == "resp_1"
    assert out["status"] == "completed"
    assert out["output"][0]["content"][0]["text"] == "run it"
    assert out["output"][1] == {
        "type": "function_call",
        "call_id": "t1",
        "name": "bash",
        "arguments": '{"c": "ls"}',
    }


def test_messages_to_responses_max_tokens_incomplete():
    payload = {
        "id": "msg-2",
        "content": [{"type": "text", "text": "par"}],
        "stop_reason": "max_tokens",
        "usage": {"input_tokens": 2, "output_tokens": 2},
    }
    out = messages_to_responses(payload, "m")
    assert out["status"] == "incomplete"
    assert out["usage"]["total_tokens"] == 4
