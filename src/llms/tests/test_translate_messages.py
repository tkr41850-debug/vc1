from __future__ import annotations

import pytest

from llms.proxy.translate import (
    from_chat,
    from_messages,
    from_responses,
    to_zen_messages,
    to_zen_responses,
)


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


def test_from_messages_accepts_system_role_in_messages():
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {"role": "system", "content": "reminder"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    assert req.messages[0].role == "system"
    assert req.messages[0].blocks[0].text == "reminder"


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


def test_web_search_tool_kind_preserved():
    req = from_messages(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "web_search_20260205", "name": "web_search", "max_uses": 3}
            ],
        }
    )
    assert req.tools[0].kind == "web_search_20260205"
    assert req.tools[0].options == {"max_uses": 3}
    resp = to_zen_responses(req)
    assert resp["tools"] == [{"type": "web_search", "max_uses": 3}]
    msg = to_zen_messages(req)
    assert msg["tools"] == [
        {"type": "web_search_20260205", "name": "web_search", "max_uses": 3}
    ]


def test_web_search_rejected_on_chat_endpoint():
    import pytest

    from llms.proxy.translate import to_zen_chat

    req = from_messages(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_20260205", "name": "web_search"}],
        }
    )
    with pytest.raises(ValueError):
        to_zen_chat(req)


def test_messages_thinking_budget_maps_to_effort():
    req = from_messages(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 16384},
        }
    )
    assert req.params.reasoning_effort == "high"


def test_effort_maps_to_thinking_budget():
    body = to_zen_messages(
        from_chat({"model": "m", "messages": [], "reasoning_effort": "low"})
    )
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_effort_round_trip_messages_responses_messages():
    req = from_messages(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
    )
    rebuilt = to_zen_messages(from_responses(to_zen_responses(req)))
    assert rebuilt["thinking"] == {"type": "enabled", "budget_tokens": 4096}
