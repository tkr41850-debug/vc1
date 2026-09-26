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
    ThinkingBlock,
    ToolCallBlock,
    ToolDef,
    ToolResultBlock,
)
from llms.proxy.translate import (
    from_chat,
    from_messages,
    from_responses,
    to_zen_chat,
    to_zen_messages,
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
    assert req.tools[0] == ToolDef("bash", "run", {"type": "object"})
    assert req.tools[1].kind == "web_search"
    assert req.params.max_tokens == 64
    assert req.stream is True


def test_from_responses_preserves_server_tool_history():
    # Codex echoes prior outputs (web_search_call, computer_call, ...)
    # back in the next request's input; 400ing breaks the session,
    # so they ride through verbatim on the responses leg.
    for item in (
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "rust"},
        },
        {"type": "computer_call", "id": "co_1", "status": "completed"},
    ):
        req = from_responses({"model": "m", "input": [item]})
        assert req.messages[0].blocks[0].item == item
        assert item in to_zen_responses(req)["input"]


def test_from_responses_history_sweep_all_builtin_calls():
    # Every server-side call type Codex/OpenAI clients can echo back
    # must survive the round trip verbatim (regression sweep).
    items = [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "rust"},
        },
        {"type": "file_search_call", "id": "fs_1", "status": "completed"},
        {"type": "computer_call", "id": "co_1", "status": "completed"},
        {"type": "computer_call_output", "call_id": "co_1", "output": {}},
        {"type": "code_interpreter_call", "id": "ci_1", "status": "completed"},
        {"type": "image_generation_call", "id": "ig_1", "status": "completed"},
        {"type": "mcp_call", "id": "mcp_1", "status": "completed"},
        {"type": "tool_search_call", "id": "ts_1", "status": "completed"},
        {"type": "item_reference", "id": "ws_1"},
    ]
    req = from_responses({"model": "m", "input": items})
    out = to_zen_responses(req)["input"]
    for item in items:
        assert item in out
    # Cross-dialect legs drop them without crashing.
    assert to_zen_chat(req)["messages"] == []
    assert to_zen_messages(req)["messages"] == []


def test_from_responses_preserves_builtin_tool_defs():
    # file_search/mcp/computer tool definitions survive ingress and
    # emit on the responses leg (was: silently dropped, then a 500 in
    # the emitter for non-web_search kinds).
    req = from_responses(
        {
            "model": "m",
            "input": "hi",
            "tools": [
                {"type": "file_search", "vector_store_ids": ["vs_1"]},
                {
                    "type": "mcp",
                    "server_label": "gh",
                    "server_url": "https://mcp.example",
                },
            ],
        }
    )
    assert req.tools[0].kind == "file_search"
    assert req.tools[1].kind == "mcp"
    tools = to_zen_responses(req)["tools"]
    assert {"type": "file_search", "vector_store_ids": ["vs_1"]} in tools
    assert {
        "type": "mcp",
        "server_label": "gh",
        "server_url": "https://mcp.example",
    } in tools


def test_from_responses_input_file_becomes_placeholder():
    req = from_responses(
        {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_file", "filename": "notes.pdf"}],
                }
            ],
        }
    )
    assert req.messages[0].blocks[0].text == "[attached file: notes.pdf]"


def test_from_responses_malformed_item_is_400_not_500():
    # Non-object history entries are client errors (400 via ValueError),
    # not unhandled 500s (AttributeError).
    with pytest.raises(ValueError):
        from_responses({"model": "m", "input": [42]})


IMG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAE"
    "hQGAhKmMIQAAAABJRU5ErkJggg=="
)
IMG_DATAURL = f"data:image/png;base64,{IMG_B64}"


def test_image_bytes_survive_all_legs():
    # Base64 bytes must arrive as bytes on every egress — stuffing a
    # data: URL into Claude's url source (http(s) only) was rejected
    # upstream.
    req = from_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "image_url", "image_url": {"url": IMG_DATAURL}},
                    ],
                }
            ],
        }
    )
    assert req.messages[0].blocks[1].url == IMG_DATAURL
    assert to_zen_chat(req)["messages"][0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": IMG_DATAURL},
    }
    assert to_zen_responses(req)["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": IMG_DATAURL,
    }
    assert to_zen_messages(req)["messages"][0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64},
    }


def test_messages_base64_image_round_trip():
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": IMG_B64,
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert req.messages[0].blocks[0].url == IMG_DATAURL
    assert to_zen_messages(req)["messages"][0]["content"] == [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": IMG_B64},
        }
    ]


def test_http_image_uses_url_source():
    req = from_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://x/i.png"},
                        }
                    ],
                }
            ],
        }
    )
    assert to_zen_messages(req)["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "url", "url": "https://x/i.png"}}
    ]


def test_responses_file_id_image_round_trips():
    req = from_responses(
        {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "file_id": "fil_123"}],
                }
            ],
        }
    )
    assert req.messages[0].blocks[0].file_id == "fil_123"
    assert to_zen_responses(req)["input"][0]["content"] == [
        {"type": "input_image", "file_id": "fil_123"}
    ]
    # Chat cannot reference Files-API ids: visible placeholder, no crash.
    assert to_zen_chat(req)["messages"][-1]["content"] == "[attached file: fil_123]"
    # Empty image parts (neither url nor id) are dropped, not emitted
    # as broken empty image_urls.
    empty = from_responses(
        {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image"}],
                }
            ],
        }
    )
    assert to_zen_responses(empty)["input"] == []


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


def test_reasoning_effort_survives_chat_to_responses():
    req = from_chat({"model": "m", "messages": [], "reasoning_effort": "high"})
    assert req.params.reasoning_effort == "high"
    assert to_zen_responses(req)["reasoning"] == {"effort": "high"}


def test_thinking_enabled_maps_to_medium_effort():
    req = from_chat({"model": "m", "messages": [], "thinking": {"type": "enabled"}})
    assert req.params.reasoning_effort == "medium"


def test_responses_reasoning_to_chat_effort():
    req = from_responses({"model": "m", "input": "hi", "reasoning": {"effort": "low"}})
    assert req.params.reasoning_effort == "low"
    assert to_zen_chat(req)["reasoning_effort"] == "low"


def test_effort_round_trip_chat_responses_chat():
    original = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
    }
    assert (
        to_zen_chat(from_responses(to_zen_responses(from_chat(original)))) == original
    )


def test_structured_output_chat_to_responses():
    req = from_chat(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        }
    )
    assert req.params.structured_output == {
        "name": "answer",
        "schema": {"type": "object"},
        "strict": True,
    }
    body = to_zen_responses(req)
    assert body["text"] == {
        "format": {
            "type": "json_schema",
            "name": "answer",
            "schema": {"type": "object"},
            "strict": True,
        }
    }


def test_structured_output_responses_to_chat():
    req = from_responses(
        {
            "model": "m",
            "input": "hi",
            "text": {"format": {"type": "json_object"}},
        }
    )
    assert req.params.structured_output == {
        "name": None,
        "schema": None,
        "strict": False,
    }
    assert to_zen_chat(req)["response_format"] == {"type": "json_object"}


def test_structured_output_round_trip():
    original = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
    }
    rebuilt = to_zen_chat(from_responses(to_zen_responses(from_chat(original))))
    assert rebuilt["response_format"]["json_schema"]["schema"] == {"type": "object"}


def test_assistant_history_uses_output_text():
    req = from_chat(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
        }
    )
    body = to_zen_responses(req)
    assert body["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]
    assert body["input"][1] == {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hello"}],
    }


def test_from_responses_parses_assistant_output_text():
    req = from_responses(
        {
            "model": "m",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
        }
    )
    assert req.messages[0].blocks == (TextBlock("hello"),)


def test_thinking_blocks_round_trip_all_dialects():
    from llms.proxy.ir import ThinkingBlock

    req = from_chat(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "let me think",
                }
            ],
        }
    )
    assert req.messages[0].blocks[0] == ThinkingBlock("let me think")
    assert to_zen_chat(req)["messages"][0]["reasoning_content"] == "let me think"
    resp = to_zen_responses(req)
    assert resp["input"][0] == {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "let me think"}],
    }
    msg = to_zen_messages(req)
    assert msg["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "let me think"},
        {"type": "text", "text": "hi"},
    ]


def test_messages_thinking_blocks_parse():
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hmm"},
                        {"type": "redacted_thinking", "data": "secret"},
                    ],
                }
            ],
        }
    )
    assert req.messages[0].blocks == (ThinkingBlock("hmm"), ThinkingBlock("secret"))


def test_responses_reasoning_item_parses():
    req = from_responses(
        {
            "model": "m",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "deep"}],
                },
            ],
        }
    )
    assert req.messages[0].blocks == (ThinkingBlock("deep"),)


def test_tool_result_in_user_message_survives_to_responses():
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "c1", "name": "bash", "input": {}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": "ok"}
                    ],
                },
            ],
        }
    )
    body = to_zen_responses(req)
    kinds = [i["type"] for i in body["input"]]
    assert kinds == ["function_call", "function_call_output"]
    assert body["input"][1] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "ok",
    }


def test_tool_result_in_user_message_survives_to_chat():
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "done"},
                        {"type": "tool_result", "tool_use_id": "c1", "content": "ok"},
                    ],
                },
            ],
        }
    )
    body = to_zen_chat(req)
    assert body["messages"][0] == {
        "role": "user",
        "content": [{"type": "text", "text": "done"}],
    }
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "ok",
    }


def test_results_only_user_message_emits_only_tool_messages():
    req = from_messages(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "c1", "content": "ok"}
                    ],
                },
            ],
        }
    )
    body = to_zen_chat(req)
    assert body["messages"] == [{"role": "tool", "tool_call_id": "c1", "content": "ok"}]


def test_null_content_fields_tolerated():
    req = from_responses(
        {
            "model": "m",
            "input": [
                {"type": "reasoning", "summary": None, "content": None},
                {"type": "message", "role": "assistant", "content": None},
            ],
        }
    )
    assert req.messages == (LlmMessage(role="assistant", blocks=()),)
    chat = from_chat(
        {
            "model": "m",
            "messages": [{"role": "assistant", "content": None, "tool_calls": None}],
        }
    )
    assert chat.messages[0].blocks == ()
    msgs = from_messages(
        {"model": "m", "messages": [{"role": "user", "content": None}]}
    )
    assert msgs.messages[0].blocks == ()
