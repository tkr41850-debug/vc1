from __future__ import annotations

import pytest

from proxy.translate import from_messages, to_zen_messages


def test_from_messages_text_and_system():
    req = from_messages(
        {
            "model": "claude-haiku-4-5",
            "system": "sys",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        }
    )
    assert req.messages[0].blocks[0].text == "sys"
    assert req.messages[0].role == "system"
    assert req.messages[1].blocks[0].text == "hi"
    assert req.params.max_tokens == 64


def test_from_messages_tool_use_and_result():
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "bash",
                            "input": {"c": "ls"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "a b"}
                    ],
                },
            ],
            "tools": [
                {
                    "name": "bash",
                    "description": "run",
                    "input_schema": {"type": "object"},
                }
            ],
            "tool_choice": {"type": "any"},
        }
    )
    call = req.messages[0].blocks[0]
    assert (call.call_id, call.name, call.arguments) == ("t1", "bash", '{"c": "ls"}')
    assert req.messages[1].blocks[0].output == "a b"
    assert req.tools[0].parameters == {"type": "object"}
    assert req.tool_choice == "required"


def test_from_messages_rejects_unknown_block():
    with pytest.raises(ValueError):
        from_messages(
            {
                "model": "m",
                "messages": [{"role": "user", "content": [{"type": "nonsense"}]}],
            }
        )


def test_to_zen_messages_round_trip():
    original = {
        "model": "claude-haiku-4-5",
        "system": "sys",
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {"c": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "a b"}
                ],
            },
        ],
        "tools": [
            {"name": "bash", "description": "run", "input_schema": {"type": "object"}}
        ],
        "max_tokens": 64,
    }
    rebuilt = to_zen_messages(from_messages(original))
    assert rebuilt["system"] == "sys"
    assert rebuilt["messages"][0] == {"role": "user", "content": "hi"}
    assert rebuilt["messages"][1]["content"][0]["name"] == "bash"
    assert rebuilt["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a b"}],
    }
    assert rebuilt["tools"][0]["input_schema"] == {"type": "object"}
    assert rebuilt["max_tokens"] == 64


def test_to_zen_messages_defaults_max_tokens():
    body = to_zen_messages(
        from_messages({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    )
    assert body["max_tokens"] == 1024
