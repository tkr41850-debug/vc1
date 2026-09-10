from __future__ import annotations

import pytest

from llms.proxy.ir import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    ImageBlock,
    LlmMessage,
    LlmParams,
    TextBlock,
    ToolCallBlock,
    ToolDef,
    ToolResultBlock,
)
from llms.proxy.translate import (
    from_chat,
    from_responses,
    to_zen_chat,
    to_zen_responses,
)


def test_from_chat_plain_string_messages():
    req = from_chat(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
        }
    )
    assert req.model == "m"
    assert req.messages == (
        LlmMessage(role=ROLE_SYSTEM, blocks=(TextBlock("sys"),)),
        LlmMessage(role=ROLE_USER, blocks=(TextBlock("hi"),)),
        LlmMessage(role=ROLE_ASSISTANT, blocks=(TextBlock("yo"),)),
    )
    assert req.stream is False


def test_from_chat_multipart_and_params():
    req = from_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "image_url", "image_url": {"url": "http://x/i.png"}},
                    ],
                }
            ],
            "stream": True,
            "temperature": 0.5,
            "top_p": 0.9,
            "max_completion_tokens": 100,
            "stop": ["###"],
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
        }
    )
    assert req.messages[0].blocks == (TextBlock("see"), ImageBlock("http://x/i.png"))
    assert req.stream is True
    assert req.params == LlmParams(
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
        stop=["###"],
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )


def test_from_chat_max_tokens_fallback():
    req = from_chat({"model": "m", "messages": [], "max_tokens": 50})
    assert req.params.max_tokens == 50


def test_from_chat_tool_calls_and_results():
    req = from_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"ls"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "out"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": "auto",
        }
    )
    assert req.messages[0].blocks == (ToolCallBlock("c1", "bash", '{"command":"ls"}'),)
    assert req.messages[1] == LlmMessage(
        role=ROLE_TOOL, blocks=(ToolResultBlock("c1", "out"),)
    )
    assert req.tools == (ToolDef("bash", "run", {"type": "object"}),)
    assert req.tool_choice == "auto"


def test_from_chat_rejects_unknown_role():
    with pytest.raises(ValueError):
        from_chat({"model": "m", "messages": [{"role": "wizard", "content": "hi"}]})


def test_from_responses_string_input_and_instructions():
    req = from_responses({"model": "m", "instructions": "be brief", "input": "hi"})
    assert req.messages == (
        LlmMessage(role=ROLE_SYSTEM, blocks=(TextBlock("be brief"),)),
        LlmMessage(role=ROLE_USER, blocks=(TextBlock("hi"),)),
    )


def test_from_responses_list_input_with_history_and_outputs():
    req = from_responses(
        {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "run ls"}],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "bash",
                    "arguments": '{"command":"ls"}',
                },
                {"type": "function_call_output", "call_id": "c1", "output": "a b"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "bash",
                    "description": "run",
                    "parameters": {"type": "object"},
                },
                {"type": "web_search", "name": "search"},
            ],
            "max_output_tokens": 64,
            "stream": True,
        }
    )
    assert req.messages[0].blocks == (TextBlock("run ls"),)
    assert req.messages[1].blocks == (ToolCallBlock("c1", "bash", '{"command":"ls"}'),)
    assert req.messages[2] == LlmMessage(
        role=ROLE_TOOL, blocks=(ToolResultBlock("c1", "a b"),)
    )
    assert req.tools == (ToolDef("bash", "run", {"type": "object"}),)
    assert req.params.max_tokens == 64
    assert req.stream is True


def test_from_responses_rejects_unknown_item():
    with pytest.raises(ValueError):
        from_responses({"model": "m", "input": [{"type": "computer_call"}]})


def test_to_zen_chat_round_trip():
    req = from_responses(
        {
            "model": "m",
            "instructions": "sys",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hi"},
                        {"type": "input_image", "image_url": "http://x/i.png"},
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "bash",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "bash",
                    "description": "run",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_output_tokens": 77,
        }
    )
    body = to_zen_chat(req)
    assert body["model"] == "m"
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "hi"},
            {"type": "image_url", "image_url": {"url": "http://x/i.png"}},
        ],
    }
    assert body["messages"][2]["tool_calls"] == [
        {
            "id": "c1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
    ]
    assert body["messages"][3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "ok",
    }
    assert body["tools"][0]["function"]["name"] == "bash"
    assert body["max_tokens"] == 77
    assert body["temperature"] == 0.3


def test_to_zen_responses_round_trip():
    req = from_chat(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "temperature": 0.3,
            "max_tokens": 77,
        }
    )
    body = to_zen_responses(req)
    assert body["model"] == "m"
    assert body["instructions"] == "sys"
    assert body["input"][0] == {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "hi"}],
    }
    assert body["input"][1] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "bash",
        "arguments": "{}",
    }
    assert body["input"][2] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "ok",
    }
    assert body["tools"] == [
        {
            "type": "function",
            "name": "bash",
            "description": "run",
            "parameters": {"type": "object"},
        }
    ]
    assert body["max_output_tokens"] == 77


def test_chat_responses_chat_stable():
    original = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "temperature": 0.2,
        "max_tokens": 40,
    }
    assert (
        to_zen_chat(from_responses(to_zen_responses(from_chat(original)))) == original
    )
