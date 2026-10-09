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


def test_nongenuine_overlapping_tools_client_definition_wins(app_client):
    # A third-party client reusing a genuine tool name keeps its OWN
    # definition in that slot (so the model calls the client's shape and
    # the call returns for the client to resolve); all genuine names
    # still ride at least once and true extras append after.
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

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
    sent_read = next(t for t in tools if t.get("name") == "read")
    assert sent_read["description"] == "mine"
    assert GENUINE_TOOL_NAMES <= {t.get("name") for t in tools}
    assert tools[-1]["name"] == "mine"


def test_is_genuine_opencode_detection():
    from llms.proxy.pipeline import is_genuine_opencode

    assert is_genuine_opencode({"user-agent": "opencode/latest/2.0.12/cli"})
    assert not is_genuine_opencode({"user-agent": "claude-cli/2.1.0"})
    assert not is_genuine_opencode({})
    # x-opencode-* headers alone never count: the proxy itself sets
    # them upstream, and codex echoes them back downstream. Honoring
    # them here silently disabled the notice + tool shaping on every
    # codex turn (live luna probe: no notice in egress instructions).
    assert not is_genuine_opencode({"x-opencode-client": "cli"})
    assert not is_genuine_opencode(
        {"x-opencode-client": "cli", "x-opencode-project": "global"}
    )


def test_genuine_calls_in_never_steers_client_named_calls():
    # Case-insensitive rule: a call naming a client-declared tool — even
    # one colliding with a genuine name in a different case (client
    # "Read" vs genuine "read") — is the client's to resolve.
    from fastapi.responses import JSONResponse

    from llms.proxy.pipeline import _genuine_calls_in

    def resp(*output):
        return JSONResponse(status_code=200, content={"output": list(output)})

    def split(r, names, tools=()):
        passed, steer = _genuine_calls_in(r, names, client_tools=tools)
        return passed + steer

    client_call = {
        "type": "function_call",
        "call_id": "c1",
        "name": "read",
        "arguments": "{}",
    }
    assert split(resp(client_call), {"read", "mine"}) == []
    # Case-variant declarations also win: client "Read" owns upstream
    # "read" (and vice versa) — returned verbatim, never steered.
    assert split(resp(client_call), {"Read", "mine"}) == []
    assert split(resp(dict(client_call, name="Read")), {"read", "mine"}) == []
    # Undeclared genuine names still steer...
    assert split(resp(client_call), {"mine"}) == [client_call]
    assert split(resp(client_call), {"Mine"}) == [client_call]
    # ...as do hallucinations with zero client tools...
    hallucinated = dict(client_call, call_id="c2", name="frobnicate")
    assert split(resp(hallucinated), set()) == [hallucinated]
    # ...while malformed items (missing/non-string name) stay steerable.
    assert split(resp({"type": "function_call"}), {"read"}) == [
        {"type": "function_call"}
    ]


def test_genuine_calls_in_splits_owned_by_required_keys():
    """With client_tools, declared names split: valid args pass through
    (converted casing), missing keys steer."""
    from fastapi.responses import JSONResponse

    from llms.proxy.ir import ToolDef
    from llms.proxy.pipeline import _genuine_calls_in

    shell = ToolDef(
        "Shell",
        "mine",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )

    def resp(*output):
        return JSONResponse(status_code=200, content={"output": list(output)})

    valid = {
        "type": "function_call",
        "call_id": "c1",
        "name": "shell",
        "arguments": '{"cmd": "echo hi"}',
    }
    passed, steer = _genuine_calls_in(resp(valid), {"Shell"}, client_tools=(shell,))
    assert steer == []
    assert [c["name"] for c in passed] == ["Shell"]
    bad = dict(valid, arguments='{"command": "echo hi"}')
    passed, steer = _genuine_calls_in(resp(bad), {"Shell"}, client_tools=(shell,))
    assert passed == []
    assert [c["name"] for c in steer] == ["shell"]


def test_genuine_calls_in_custom_route_passes_raw_input():
    """Custom-route calls (freeform nested tools) pass through into the
    exec channel with raw input — never steered on JSON required-keys."""
    from fastapi.responses import JSONResponse

    from llms.proxy.pipeline import _genuine_calls_in
    from tests.test_client_tools import _luna_namespace

    def resp(*output):
        return JSONResponse(status_code=200, content={"output": list(output)})

    ns = _luna_namespace()
    raw = {
        "type": "function_call",
        "call_id": "c1",
        "name": "apply_patch",
        "arguments": "*** Begin Patch ***",
    }
    passed, steer = _genuine_calls_in(resp(raw), set(), client_tools=(ns,))
    assert steer == []
    # Exec-channel rewrite rides a marker (the replay paths match
    # frames by the emitted name, then apply it): input preserved
    # verbatim for the freeform patch text.
    assert passed[0]["__exec_rewrite__"] == {
        "name": "exec",
        "input": "*** Begin Patch ***",
    }
    # Function-route nested calls still validate args-object shape:
    # valid args pass (into the exec channel), bare calls steer with
    # a correction.
    good = dict(
        raw,
        call_id="c2",
        name="exec_command",
        arguments='{"cmd": "cat f"}',
    )
    passed, steer = _genuine_calls_in(resp(good), set(), client_tools=(ns,))
    assert steer == [] and passed[0]["type"] == "function_call"
    assert passed[0]["__exec_rewrite__"]["name"] == "exec"
    bare = dict(raw, call_id="c3", name="exec_command", arguments="")
    passed, steer = _genuine_calls_in(resp(bare), set(), client_tools=(ns,))
    assert passed == [] and [c["name"] for c in steer] == ["exec_command"]


def test_genuine_calls_in_sees_wire_custom_tool_call_items():
    # Mechanism-13 live shape (luna natural-write probe 2026-10-07):
    # the non-streaming upstream response carries the model's call as
    # a wire `custom_tool_call` item (type custom_tool_call, payload
    # in `input`), NOT a function_call. The collector only scanned
    # function_call items, so the bare-patch exec input never reached
    # the classifier's rewrap arm — ([], []), silently dropped, and
    # the raw patch rode downstream to 4x harness SyntaxError. A wire
    # Custom item must feed the classifier like the streaming fold's
    # entries do (fold_stream_calls emits type custom_tool_call + input).
    from fastapi.responses import JSONResponse

    from llms.proxy.compat import rewrap_bare_patch_exec_input
    from llms.proxy.pipeline import _genuine_calls_in
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES
    from tests.test_client_tools import _luna_namespace

    def resp(*output):
        return JSONResponse(status_code=200, content={"output": list(output)})

    ns = (_luna_namespace(),)
    bare = "'*** Begin Patch\\n*** Add File: w.txt\\n+hi\\n*** End Patch'"
    wire = {
        "type": "custom_tool_call",
        "id": "call_m13",
        "call_id": "call_m13",
        "name": "exec",
        "input": bare,
    }
    passed, steer = _genuine_calls_in(
        resp(wire),
        set(),
        client_tools=ns,
        genuine_names=GENUINE_TOOL_NAMES,
        family="luna",
        trace_id="t1",
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec"]
    assert passed[0]["__exec_rewrite__"] == {
        "name": "exec",
        "input": rewrap_bare_patch_exec_input(bare),
    }


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


def test_messages_rebuild_is_byte_stable_across_turns(app_client):
    # Prefix caching upstream depends on our rebuild being deterministic:
    # identical logical turns must produce byte-identical bodies, and a
    # follow-up must extend (never rewrite) the prefix.
    import json as _json

    tc, seen = app_client
    bodies = []

    def post(messages):
        r = tc.post(
            "/v1/messages",
            json={
                "model": "claude-haiku-4-5",
                "system": [
                    {"type": "text", "text": "sys"},
                    {
                        "type": "text",
                        "text": "more",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
                "messages": messages,
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
            },
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        bodies.append(_json.dumps(seen["json"], sort_keys=True))
        return seen["headers"]["x-opencode-session"]

    history = [
        {"role": "user", "content": "do the thing"},
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
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a b"}],
        },
    ]
    s1 = post(history)
    s2 = post(history)
    assert bodies[0] == bodies[1]
    assert s1 == s2
    s3 = post([*history, {"role": "user", "content": "and another"}])
    assert s3 == s1
    first_items = _json.loads(bodies[0])["messages"]
    third_items = _json.loads(bodies[2])["messages"]
    assert third_items[: len(first_items)] == first_items


def test_case_variant_client_tool_appends_beside_genuine(app_client):
    # Client "Read" vs genuine "read": the gate needs the 12 genuine
    # definitions byte-identical, so genuine "read" stays untouched and
    # the client tool appends after as an extra (never replaces, never
    # dedups). Steering still treats them as the same name (see
    # _genuine_calls_in case-insensitive rule), so the call passes
    # back for the client to resolve.
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES, GENUINE_TOOLS

    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "mine",
                    "parameters": {},
                },
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    tools = seen["json"]["tools"]
    names = [t.get("name") for t in tools]
    # All 12 genuine names present with genuine definitions, in order...
    assert names[:12] == [t["name"] for t in GENUINE_TOOLS]
    assert GENUINE_TOOL_NAMES <= set(names)
    genuine_read = next(t for t in tools if t.get("name") == "read")
    assert genuine_read["description"] != "mine"
    # ...and the client variant rides after as an extra.
    assert names[12:] == ["Read"]


def test_genuine_tool_order_and_identity_preserved(app_client):
    """The 12 genuine definitions go out byte-identical, in order, first.

    Fingerprint contract: Zen's free-tier gate fuzzy-matches the set, so
    no reorder, no rename, no client definition in a genuine slot unless
    the client used the EXACT same name.
    """
    from llms.proxy.zen_tools import GENUINE_TOOLS

    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "hi",
            "tools": [
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
    assert tools[:12] == GENUINE_TOOLS
    assert [t.get("name") for t in tools][12:] == ["mine"]


def test_overlay_skips_genuine_twin_of_declared_name(app_client):
    """A client-declared 'shell' sees one shell slot: its own definition.

    Regression (live codex wire capture): the overlay prepended genuine
    'shell' ahead of codex's declared extras; the model called the
    overlay twin and codex failed the turn ('unsupported call: shell').
    Now the genuine twin is skipped and the client's definition rides
    in the slot position — exactly one shell, client-owned.
    """
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "shell",
                    "description": "codex shell",
                    "parameters": {"type": "object"},
                },
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "codex exec",
                    "parameters": {"type": "object"},
                },
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    tools = seen["json"]["tools"]
    assert [t.get("name") for t in tools].count("shell") == 1
    sent_shell = next(t for t in tools if t.get("name") == "shell")
    assert sent_shell["description"] == "codex shell"
    # Gate safety: every other genuine name still rides at least once,
    # and declared extras keep their slot-ahead-of-extras position.
    assert (GENUINE_TOOL_NAMES - {"shell"}) <= {t.get("name") for t in tools}
    assert tools.index(sent_shell) < tools.index(
        next(t for t in tools if t.get("name") == "exec_command")
    )


def test_overlay_case_variant_keeps_genuine_and_appends_client(app_client):
    """Client 'Read' vs genuine 'read': genuine untouched, client appends.

    Case-insensitive skip must not collapse the variant: renaming the
    genuine definition breaks the gate, so both ride (existing contract).
    """
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "Read",
                    "description": "mine",
                    "parameters": {},
                },
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    tools = seen["json"]["tools"]
    sent_read = next(t for t in tools if t.get("name") == "read")
    assert sent_read["description"] != "mine"
    assert next(t for t in tools if t.get("name") == "Read")["description"] == "mine"


def test_classify_namespaced_owned_call_passes_without_default_marker():
    """Code-mode `default.exec_command` classifies on the bare name and
    passes WITHOUT a route marker: the harness fills its own default
    namespace when absent, and a foreign `default` value poisons lookup
    client-side (live: replayed `default.exec_command` failed with
    `unsupported call`). Non-default namespaces still ride the
    marker for the dispatch replay."""
    from llms.proxy.client_tools import dispatchable_names
    from llms.proxy.ir import ToolDef
    from llms.proxy.pipeline import _classify_calls, owned_tool_names

    shell = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = owned_tool_names((shell,))
    defs = {"exec_command": shell}
    calls = [
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "default.exec_command",
            "arguments": '{"cmd": "cat f"}',
        }
    ]
    passed, steer = _classify_calls(calls, owned, defs, dispatchable_names((shell,)))
    assert steer == []
    assert passed[0]["name"] == "exec_command"
    assert "__route_namespace__" not in passed[0]


def test_shell_retransmit_same_call_id_steers_not_replays():
    """Same-call_id retransmit dedup (live spark harness 2026-10-06):
    the model emits the translated `shell` AND the client runner name
    for one action under one call_id. The duplicate steers (never
    replays): a same-call_id double-execution would run the command
    twice, and the client rejects the retransmit as a duplicate."""
    from llms.proxy.ir import ToolDef
    from llms.proxy.pipeline import _classify_calls

    runner = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    calls = [
        {"call_id": "c1", "name": "shell", "arguments": '{"command": "cat f"}'},
        {"call_id": "c1", "name": "shell", "arguments": '{"command": "cat f"}'},
    ]
    passed, steer = _classify_calls(calls, owned, defs, None, (runner,))
    assert len(passed) == 1
    assert len(steer) == 1
    assert steer[0]["name"] == "shell"
    # Distinct call ids are independent actions: both pass.
    calls = [
        {"call_id": "c1", "name": "shell", "arguments": '{"command": "cat f"}'},
        {"call_id": "c2", "name": "shell", "arguments": '{"command": "cat f"}'},
    ]
    passed, steer = _classify_calls(calls, owned, defs, None, (runner,))
    assert len(passed) == 2 and not steer


def test_classify_shell_rewrites_with_family_threaded():
    """Family-threaded genuine->client rewrite (compat Task 4): a `shell`
    emission on the luna leg rewrites onto the nested exec_command AND
    rides the exec-channel marker (a bare function_call exec_command
    fails lookup — only custom_tool_call exec dispatches). On a plain
    function leg the same rewrite is a bare rename with no marker."""
    from llms.proxy.client_tools import (
        dispatchable_names,
        nested_tool_defs,
        owned_tool_names,
    )
    from llms.proxy.ir import ToolDef
    from llms.proxy.pipeline import _classify_calls
    from tests.test_client_tools import _luna_namespace

    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    route = dispatchable_names(ns)
    calls = [{"call_id": "c1", "name": "shell", "arguments": '{"command": "cat f"}'}]
    passed, steer = _classify_calls(calls, owned, defs, route, ns, (), family="luna")
    assert not steer
    assert len(passed) == 1
    assert passed[0]["name"] == "exec_command"
    assert passed[0]["__exec_rewrite__"] == {
        "name": "exec",
        "input": 'await tools.exec_command({"cmd": "cat f"})',
    }
    # The JS input already carries the translated arguments: keeping
    # __translated_args__ alongside the marker would also rewrite the
    # done frame and refold a Frankenstein call.
    assert "__translated_args__" not in passed[0]
    # Plain function leg, unknown family: bare rename, no marker.
    runner = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    calls = [{"call_id": "c1", "name": "shell", "arguments": '{"command": "cat f"}'}]
    passed, steer = _classify_calls(
        calls, owned, defs, None, (runner,), family="unknown"
    )
    assert not steer
    assert len(passed) == 1
    assert passed[0]["name"] == "exec_command"
    assert "__exec_rewrite__" not in passed[0]
