from __future__ import annotations

from llms.proxy.translate import (
    from_chat,
    from_messages,
    from_responses,
    to_zen_chat,
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


def test_from_messages_preserves_server_tool_blocks():
    # Claude Code echoes prior server-side blocks back verbatim;
    # 400ing breaks the session, so they ride through as part-level
    # opaques (verbatim on the messages leg, dropped elsewhere).
    block = {"type": "server_tool_use", "id": "srv_1", "name": "web_search"}
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}, block]}
            ],
        }
    )
    opaque = req.messages[0].blocks[1]
    assert opaque.part is True
    assert opaque.item == block
    rebuilt = to_zen_messages(req)
    assert block in rebuilt["messages"][0]["content"]
    # Cross-dialect: dropped, not crashed.
    assert to_zen_responses(req)["input"][0]["content"] == [
        {"type": "input_text", "text": "hi"}
    ]
    assert to_zen_chat(req)["messages"][-1]["content"] == "hi"


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
    assert resp["tools"] == [{"type": "web_search"}]
    msg = to_zen_messages(req)
    assert msg["tools"] == [
        {"type": "web_search_20260205", "name": "web_search", "max_uses": 3}
    ]


def test_web_search_dropped_on_chat_endpoint():
    # Chat completions have no web_search tool: the definition is
    # dropped (was an unhandled 500) while function tools survive.
    from llms.proxy.translate import to_zen_chat

    req = from_messages(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "web_search_20260205", "name": "web_search"},
                {
                    "name": "bash",
                    "description": "run",
                    "input_schema": {"type": "object"},
                },
            ],
        }
    )
    body = to_zen_chat(req)
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run",
                "parameters": {"type": "object"},
            },
        }
    ]

    only_search = from_messages(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "web_search_20260205", "name": "web_search"}],
        }
    )
    assert "tools" not in to_zen_chat(only_search)


def test_messages_file_id_image_round_trips():
    block = {"type": "image", "source": {"type": "file_id", "file_id": "fil_123"}}
    req = from_messages(
        {"model": "m", "messages": [{"role": "user", "content": [block]}]}
    )
    opaque = req.messages[0].blocks[0]
    assert opaque.part is True
    assert opaque.item == block
    assert to_zen_messages(req)["messages"][0]["content"] == [block]


def test_messages_cache_breakpoints_round_trip():
    # Claude Code pins cache_control on system/tools/content so the
    # provider caches the prefix; the rebuild must carry them through
    # (values included — ephemeral_1h buys the long TTL).
    original = {
        "model": "claude-haiku-4-5",
        "system": [
            {"type": "text", "text": "sys"},
            {
                "type": "text",
                "text": "more",
                "cache_control": {"type": "ephemeral_1h"},
            },
        ],
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "bash",
                        "input": {},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [
                            {
                                "type": "text",
                                "text": "a b",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {
                        "type": "text",
                        "text": "go",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
        "tools": [
            {"name": "bash", "description": "run", "input_schema": {}},
            {
                "name": "read",
                "description": "read",
                "input_schema": {},
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "max_tokens": 64,
    }
    rebuilt = to_zen_messages(from_messages(original))
    assert rebuilt["system"] == [
        {"type": "text", "text": "sys"},
        {
            "type": "text",
            "text": "more",
            "cache_control": {"type": "ephemeral_1h"},
        },
    ]
    assert rebuilt["messages"][0]["content"] == [
        {
            "type": "tool_use",
            "id": "t1",
            "name": "bash",
            "input": {},
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert rebuilt["messages"][1]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": "a b",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "go", "cache_control": {"type": "ephemeral"}},
    ]
    assert rebuilt["tools"][1] == {
        "name": "read",
        "description": "read",
        "input_schema": {},
        "cache_control": {"type": "ephemeral"},
    }
    # Cross-dialect legs have no breakpoint equivalent: dropped, no crash.
    import json as _json

    chat_body = to_zen_chat(from_messages(original))
    assert "cache_control" not in _json.dumps(chat_body)
    assert chat_body["messages"][-2]["content"] == [{"type": "text", "text": "go"}]
    assert chat_body["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "a b",
    }
    resp_body = to_zen_responses(from_messages(original))
    assert "cache_control" not in _json.dumps(resp_body)
    assert resp_body["input"][-1]["content"] == [{"type": "input_text", "text": "go"}]


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


def test_messages_image_inside_tool_result_round_trips():
    # Claude Code Read of an image file nests the bytes in tool_result
    # content; dropping them left the model blind with an empty result.
    data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mM="
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": data,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    )
    call, img = req.messages[0].blocks
    assert call.output == ""
    assert img.url == f"data:image/png;base64,{data}"
    assert to_zen_messages(req)["messages"][0]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": ""},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": data},
        },
    ]
