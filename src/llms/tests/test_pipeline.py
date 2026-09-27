from __future__ import annotations

from tests.conftest import TEST_HEADERS


def test_chat_ingress_routes_spark_to_responses(app_client, mock_upstream):
    tc, seen = app_client
    _, seen_dict = mock_upstream
    # Anonymous responses egress streams upstream even for single-shot
    # callers, folding SSE into one JSON body.
    seen_dict["mode"] = "stream"
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/responses")
    assert seen["json"]["stream"] is True
    assert seen["json"]["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hi"


def test_responses_ingress_routes_mimo_to_chat(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={"model": "mimo-v2.5-free", "input": "hi"},
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    # Anonymous chat egress leads with the canonical system message.
    assert seen["json"]["messages"][0]["role"] == "system"
    assert seen["json"]["messages"][-1] == {"role": "user", "content": "hi"}
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"


def test_responses_web_search_history_forwards_verbatim(app_client):
    # Codex session continuation: prior web_search_call output echoed
    # back in input must not 400; it forwards verbatim upstream.
    tc, seen = app_client
    ws = {
        "type": "web_search_call",
        "id": "ws_123",
        "status": "completed",
        "action": {"type": "search", "query": "rust async"},
    }
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                ws,
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/responses")
    assert ws in seen["json"]["input"]


def test_responses_builtin_tools_dropped_on_chat_egress(app_client):
    # A responses client declaring web_search routed to a chat model
    # must not 500: non-function tools are dropped, history passes on.
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "mimo-v2.5-free",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "rust"},
                },
            ],
            "tools": [{"type": "web_search"}],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert "tools" not in seen["json"]


OC_HEADERS = dict(
    TEST_HEADERS,
    **{
        "User-Agent": "opencode/latest/2.0.12/cli",
        "x-opencode-client": "cli",
        "x-opencode-project": "global",
        "x-opencode-session": "ses_abcdef1234567890abcdefghij12",
    },
)


def test_genuine_opencode_responses_passthrough(app_client):
    # Genuine opencode already carries the exact wire identity: its
    # tools must not be duplicated and instructions pass untouched.
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    tc, seen = app_client
    genuine = [
        {"type": "function", "name": n, "description": "d", "parameters": {}}
        for n in sorted(GENUINE_TOOL_NAMES)
    ]
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "instructions": "You are opencode.",
            "input": "hi",
            "tools": genuine,
        },
        headers=OC_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/responses")
    assert seen["json"]["instructions"] == "You are opencode."
    names = [t.get("name") for t in seen["json"]["tools"]]
    assert sorted(names) == sorted(GENUINE_TOOL_NAMES)


def test_genuine_opencode_chat_passthrough(app_client):
    # Genuine system prompt must not gain the TITLE_PREFIX lead.
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [
                {"role": "system", "content": "You are opencode."},
                {"role": "user", "content": "hi"},
            ],
        },
        headers=OC_HEADERS,
    )
    assert r.status_code == 200
    assert seen["json"]["messages"][0] == {
        "role": "system",
        "content": "You are opencode.",
    }


def test_nongenuine_overlapping_tools_deduped(app_client):
    # A third-party client reusing a genuine tool name keeps the
    # genuine definition once; true extras still append.
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "read",
                    "description": "mine",
                    "parameters": {},
                },
                {
                    "type": "function",
                    "name": "mine",
                    "description": "extra",
                    "parameters": {},
                },
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    tools = seen["json"]["tools"]
    assert [t.get("name") for t in tools].count("read") == 1
    assert tools[-1]["name"] == "mine"


def test_is_genuine_opencode_detection():
    from llms.proxy.pipeline import is_genuine_opencode

    assert is_genuine_opencode({"user-agent": "opencode/latest/2.0.12/cli"})
    assert is_genuine_opencode({"x-opencode-client": "cli"})
    assert not is_genuine_opencode({"user-agent": "claude-cli/2.1.0"})
    assert not is_genuine_opencode({})


def test_server_tool_specs_forwarded_never_500(app_client):
    # Their curl repros #5/#6 against our proxy: unknown server-tool
    # shapes (web_fetch, server_tool_use-as-tool) forward verbatim for
    # upstream to judge — JSON downstream, never a plaintext 500.
    tc, seen = app_client
    r = tc.post(
        "/v1/messages",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "max_tokens": 50,
            "tools": [
                {"type": "web_fetch_20260209", "name": "web_fetch"},
                {"type": "server_tool_use", "name": "web_search"},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    kinds = [t.get("type") for t in seen["json"]["tools"] if isinstance(t, dict)]
    assert "web_fetch_20260209" in kinds
    assert "server_tool_use" in kinds


def test_same_dialect_passes_through_untouched(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert r.json()["choices"][0]["message"]["content"] == "hello"


def test_cross_dialect_stream_translates(app_client, mock_upstream):
    tc, _seen = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "stream"
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert "hi" in r.text
    assert "data: [DONE]" in r.text
    assert "inference-cost" not in r.text


def test_messages_ingress_routes_spark_to_responses(app_client, mock_upstream):
    tc, seen = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "stream"
    r = tc.post(
        "/v1/messages",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/responses")
    assert seen["json"]["stream"] is True
    body = r.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "hi"}]


def test_responses_ingress_routes_claude_to_messages(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={"model": "claude-haiku-4-5", "input": "hi"},
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/messages")
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"


def test_model_alias_remaps_before_routing(mock_upstream, tmp_path):
    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    client, seen = mock_upstream
    settings = make_settings(
        data_dir=str(tmp_path),
        model_aliases=(
            ("gpt-*", "muse-spark-1.3-contributor-free"),
            ("claude-*", "muse-spark-1.3-contributor-free"),
        ),
    )
    with build_app_client(settings, client, seed_key=TEST_SECRET) as tc:
        r = tc.post(
            "/ak-team1/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert seen["url"].endswith("/responses")
        assert seen["json"]["model"] == "muse-spark-1.3-contributor-free"


def test_chat_image_forwards_on_chat_egress(app_client):
    # DSH path (chat dialect both sides): image parts pass through
    # byte-identical; the harness itself sends text-only, so this is
    # the wire contract for any chat client attaching images.
    tc, seen = app_client
    url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mM="
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colors?"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert seen["json"]["messages"][-1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": url},
    }


def test_compacted_claude_turn_flows(app_client):
    # Post-/compact wire shape from real transcripts: plain-text summary
    # (+ redacted thinking, document blocks) must flow without 400s.
    tc, seen = app_client
    r = tc.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "system": "You are Claude Code.",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "<summary>Earlier: fixed image bugs.</summary>",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "enc"},
                        {"type": "text", "text": "ack"},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "notes",
                            },
                        },
                        {"type": "text", "text": "continue"},
                    ],
                },
            ],
            "max_tokens": 64,
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/messages")
    bodies = seen["json"]["messages"]
    assert "<summary>" in bodies[0]["content"]
    assert {"type": "thinking", "thinking": "enc"} in bodies[1]["content"]
    assert bodies[2]["content"][0]["type"] == "document"


def test_compacted_codex_thread_keeps_session(app_client):
    # Codex compaction keeps thread-id: the summary turn must reuse the
    # pre-compaction session so the cache stays warm.
    tc, seen = app_client
    headers = dict(
        TEST_HEADERS,
        **{"originator": "codex_exec", "thread-id": "thread-compact-1"},
    )
    for i, text in enumerate(["do the thing", "<summary>did the thing</summary>"]):
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": text},
            headers=headers,
        )
        assert r.status_code == 200
        if i == 0:
            first_key = seen["json"]["prompt_cache_key"]
        else:
            assert seen["json"]["prompt_cache_key"] == first_key


def test_malformed_tool_specs_are_json_never_plaintext_500(app_client):
    # Malformed/unknown tool specs on any dialect: JSON downstream
    # (forwarded for upstream to judge, or 400) — never plaintext 500.
    cases = [
        (
            "/v1/responses",
            {
                "model": "muse-spark-1.3-contributor-free",
                "input": "hi",
                "tools": [{"type": "server_tool_use", "name": "web_search"}],
            },
        ),
        (
            "/v1/messages",
            {
                "model": "claude-haiku-4-5",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "server_tool_use", "name": "web_search"}],
            },
        ),
        (
            "/v1/chat/completions",
            {
                "model": "mimo-v2.5-free",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "weird", "name": "x"}],
            },
        ),
    ]
    tc, _ = app_client
    for path, body in cases:
        r = tc.post(path, json=body, headers=TEST_HEADERS)
        assert r.status_code != 500, path
        assert r.headers["content-type"].startswith("application/json"), path


def test_codex_web_search_tool_forwarded(app_client):
    # Codex equivalent of the web_search report: the declaration must
    # reach upstream (execution is the backend's job, never the proxy's
    # to fake by dropping).
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "search the web for SWE-smith",
            "tools": [{"type": "web_search"}],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["json"]["tools"][-1] == {"type": "web_search"}


def test_distinct_codex_threads_get_distinct_sessions(app_client):
    # Distinct conversations => distinct sessions (hence distinct
    # buckets/slots); same thread stays pinned (see compact test).
    tc, seen = app_client
    keys = []
    for thread in ("thread-A", "thread-B"):
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=dict(
                TEST_HEADERS, **{"originator": "codex_exec", "thread-id": thread}
            ),
        )
        assert r.status_code == 200
        keys.append(seen["json"]["prompt_cache_key"])
    assert keys[0] and keys[1] and keys[0] != keys[1]
