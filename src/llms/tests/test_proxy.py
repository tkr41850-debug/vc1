from __future__ import annotations

from tests.conftest import TEST_HEADERS


def test_healthz(app_client):
    tc, _ = app_client
    r = tc.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_healthz_reports_degraded_store(app_client, tmp_path):
    from llms.proxy.keys import reset_cache

    tc, _ = app_client
    (tmp_path / "keys.yaml").write_text("{unclosed: [bracket\n  - nope")
    reset_cache()
    try:
        # any gated request trips the corrupt store into visibility
        assert (
            tc.post(
                "/v1/responses", json={"input": "hi"}, headers=TEST_HEADERS
            ).status_code
            == 503
        )
        r = tc.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "degraded"
        # Unauthenticated path: fixed degraded signal, never raw parser
        # text (YAML errors echo the offending line, a key/config leak).
        assert r.json()["store_error"] == "key store unavailable"
    finally:
        reset_cache()


def test_non_stream_passthrough_with_model_override(app_client, mock_upstream):
    # Anonymous responses egress streams upstream and folds SSE into JSON.
    tc, seen = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "stream"
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "say hi",
            "stream": False,
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["output"][0]["content"][0]["text"] == "hi"
    assert seen["url"].endswith("/responses")
    assert seen["json"]["model"] == "muse-spark-1.3-contributor-free"
    assert seen["json"]["stream"] is True
    assert seen["headers"]["x-opencode-client"] == "cli"
    assert seen["headers"]["user-agent"].startswith("opencode/")


def test_responses_egress_prompt_cache_key_matches_session(app_client):
    # Zen's free-tier gate requires body prompt_cache_key == session header.
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "say hi",
            "stream": False,
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["json"]["prompt_cache_key"] == seen["headers"]["x-opencode-session"]


def test_anonymous_instructions_pass_through_untouched(app_client):
    from llms.proxy.zen_tools import GENUINE_TOOLS

    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "instructions": "Be brief.",
            "input": "hi",
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["json"]["instructions"] == "Be brief."
    assert [t["name"] for t in seen["json"]["tools"]] == [
        t["name"] for t in GENUINE_TOOLS
    ]


def test_anonymous_bare_single_gets_genuine_tools_only(app_client):
    from llms.proxy.zen_tools import GENUINE_TOOLS

    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert "instructions" not in seen["json"]
    assert [t["name"] for t in seen["json"]["tools"]] == [
        t["name"] for t in GENUINE_TOOLS
    ]


def test_anonymous_dialogue_without_tools_gets_genuine_tools(app_client):
    from llms.proxy.zen_tools import GENUINE_TOOLS

    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "again"},
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert "instructions" not in seen["json"]
    assert [t["name"] for t in seen["json"]["tools"]] == [
        t["name"] for t in GENUINE_TOOLS
    ]


def test_anonymous_tooled_turn_sends_genuine_only(app_client):
    # Outbound is genuine-12 ONLY: client extras never ride the wire
    # (any one can fail the upstream validator and 400 the whole
    # request — seen live as `tools[12].description` length on a codex
    # session). The client tool lives in the notice instead.
    from llms.proxy.zen_tools import GENUINE_TOOLS

    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "instructions": "Be brief.",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "bash",
                    "description": "run",
                    "parameters": {"type": "object"},
                }
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    sent_tools = seen["json"]["tools"]
    assert [t["name"] for t in sent_tools] == [t["name"] for t in GENUINE_TOOLS]
    assert sent_tools == GENUINE_TOOLS
    # Client tool notice appends after client instructions.
    assert seen["json"]["instructions"].startswith("Be brief.")
    assert "'bash'" in seen["json"]["instructions"]


def test_steering_redirects_genuine_calls(tmp_path):
    """A genuine tool call is answered with a redirect and re-requested."""
    import json as _json

    import httpx

    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    calls: list = []

    async def handler(request):
        payload = _json.loads(request.content.decode())
        calls.append(payload)
        if len(calls) == 1:
            body = (
                'data: {"type":"response.output_item.added","output_index":1,'
                '"item":{"id":"call_shell1","type":"function_call","name":"shell",'
                '"arguments":"{}"}}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"output_index":1,"item_id":"call_shell1","delta":"{}"}\n\n'
                'data: {"type":"response.function_call_arguments.done",'
                '"output_index":1,"item_id":"call_shell1","arguments":"{}"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
            )
        else:
            body = (
                'data: {"type":"response.output_text.delta","delta":"done"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
            )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "name": "bash",
                        "description": "run",
                        "parameters": {"type": "object"},
                    }
                ],
            },
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert len(calls) == 2
        followup = calls[1]
        outputs = [
            i
            for i in followup["input"]
            if isinstance(i, dict) and i.get("type") == "function_call_output"
        ]
        assert outputs and "bash" in outputs[0]["output"]
        texts = [
            p.get("text", "")
            for m in r.json().get("output", [])
            if m.get("type") == "message"
            for p in m.get("content", [])
        ]
        assert "".join(texts) == "done"


def test_steering_names_client_tools_in_queued_message(tmp_path):
    """Steer redirect lists the client's own tools (codex-shaped toolset).

    The follow-up queued message carries function_call + function_call_output
    with matching call ids, exactly like a normal queued tool result.
    """
    import json as _json

    import httpx

    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    calls: list = []

    async def handler(request):
        payload = _json.loads(request.content.decode())
        calls.append(payload)
        if len(calls) == 1:
            body = (
                'data: {"type":"response.output_item.added","output_index":2,'
                '"item":{"id":"call_read1","type":"function_call","name":"read",'
                '"arguments":"{}"}}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"output_index":2,"item_id":"call_read1","delta":"{}"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
            )
        else:
            body = (
                'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
                'data: {"type":"response.completed","response":{"status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
            )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "read the readme",
                "tools": [
                    {
                        "type": "function",
                        "name": "local_shell",
                        "description": "run shell",
                        "parameters": {"type": "object"},
                    },
                    {
                        "type": "function",
                        "name": "read_file",
                        "description": "read file",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert len(calls) == 2
        followup = calls[1]
        items = [i for i in followup["input"] if isinstance(i, dict)]
        call = next(i for i in items if i.get("type") == "function_call")
        output = next(i for i in items if i.get("type") == "function_call_output")
        assert call["name"] == "read"
        assert output["call_id"] == call["call_id"] == "call_read1"
        assert "local_shell, read_file" in output["output"]


def _stream_seen_client(mock_upstream, tmp_path):
    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    client, seen = mock_upstream
    seen["mode"] = "stream"
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        yield tc, seen


def _tool_call_sse(call_id: str, name: str, arguments: str = "{}") -> bytes:
    """One upstream SSE body emitting a single function_call then completing."""
    import json as _json

    head = (
        f'data: {{"type":"response.output_item.added","output_index":1,'
        f'"item":{{"id":{_json.dumps(call_id)},"type":"function_call",'
        f'"name":{_json.dumps(name)},"arguments":{_json.dumps(arguments)}}}}}\n\n'
    )
    delta = (
        f"data: {_json.dumps({'type': 'response.function_call_arguments.delta', 'output_index': 1, 'item_id': call_id, 'delta': arguments})}\n\n"
        f"data: {_json.dumps({'type': 'response.function_call_arguments.done', 'output_index': 1, 'item_id': call_id, 'arguments': arguments})}\n\n"
    )
    tail = (
        'data: {"type":"response.completed","response":{"status":"completed",'
        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
    )
    return (head + delta + tail).encode()


def _text_sse(text: str) -> bytes:
    import json as _json

    body = (
        f'data: {{"type":"response.output_text.delta","delta":{_json.dumps(text)}}}\n\n'
        'data: {"type":"response.completed","response":{"status":"completed",'
        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
    )
    return body.encode()


def _steer_app_client(tmp_path, first_body: bytes, second_body: bytes):
    """App client whose upstream emits first_body, then second_body."""
    import httpx

    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        body = first_body if len(calls) == 1 else second_body
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    tc = build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    )
    return tc, calls


def test_client_named_genuine_tool_call_case_insensitive_passthrough(tmp_path):
    """Client "Read" owns upstream "read":-owned name converts to the
    declared casing for the client to resolve — no steer follow-up
    (exactly 1 upstream call)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_read1", "read", '{"path":"notes.txt"}'),
        _text_sse("unreached"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "read notes",
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
    assert len(calls) == 1
    returned = [
        i for i in r.json().get("output", []) if i.get("type") == "function_call"
    ]
    assert len(returned) == 1
    assert returned[0]["name"] == "Read"
    assert returned[0]["call_id"] == "call_read1"
    assert returned[0]["arguments"] == '{"path":"notes.txt"}'


def test_client_named_genuine_tool_call_passes_through_unsteered(tmp_path):
    """A call naming a client-declared tool colliding with a genuine name
    (read) is returned verbatim: no steer follow-up (exactly 1 upstream
    call). Outbound is genuine-12 ONLY — the client variant rides the
    notice (model-facing) and the response convert (client-facing)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_read1", "read", '{"path":"notes.txt"}'),
        _text_sse("unreached"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "read notes",
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
    assert len(calls) == 1
    from llms.proxy.zen_tools import GENUINE_TOOLS

    assert calls[0]["tools"] == GENUINE_TOOLS
    assert "'read'" in calls[0].get("instructions", "")
    returned = [
        i for i in r.json().get("output", []) if i.get("type") == "function_call"
    ]
    assert len(returned) == 1
    assert returned[0]["name"] == "read"
    assert returned[0]["call_id"] == "call_read1"
    assert returned[0]["arguments"] == '{"path":"notes.txt"}'


def test_undeclared_genuine_tool_call_still_steers(tmp_path):
    """A genuine tool the client did NOT declare (shell, client only sent
    mine) still steers: 2 upstream calls, redirect queued, text answered."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_shell1", "shell"),
        _text_sse("done"),
    )
    with tc_ctx as tc:
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
                        "parameters": {"type": "object"},
                    }
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    followup = calls[1]
    outputs = [
        i
        for i in followup["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert (
        outputs and "shell" in outputs[0]["output"] and "mine" in outputs[0]["output"]
    )
    texts = [
        p.get("text", "")
        for m in r.json().get("output", [])
        if m.get("type") == "message"
        for p in m.get("content", [])
    ]
    assert "".join(texts) == "done"


def test_hallucinated_tool_call_with_no_client_tools_still_steers(tmp_path):
    """An undeclared name with zero client tools still steers with the
    answer-directly nudge (no tool list)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_h1", "frobnicate"),
        _text_sse("ok"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    followup = calls[1]
    outputs = [
        i
        for i in followup["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert len(outputs) == 1
    assert "frobnicate" in outputs[0]["output"]
    assert "answer directly" in outputs[0]["output"]
    assert "Use one of these tools instead" not in outputs[0]["output"]
    texts = [
        p.get("text", "")
        for m in r.json().get("output", [])
        if m.get("type") == "message"
        for p in m.get("content", [])
    ]
    assert "".join(texts) == "ok"


def test_anonymous_nonstream_synthesizes_json_from_upstream_sse(
    mock_upstream, tmp_path
):  # Anonymous Zen requires stream:true even for single-shot callers: the
    # proxy streams upstream and folds SSE into one JSON body.
    from tests.conftest import TEST_HEADERS

    for tc, seen in _stream_seen_client(mock_upstream, tmp_path):
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert seen["json"]["stream"] is True
        assert r.json()["status"] == "completed"


def test_anonymous_nonstream_chat_synthesizes_across_dialects(mock_upstream, tmp_path):
    from tests.conftest import TEST_HEADERS

    for tc, seen in _stream_seen_client(mock_upstream, tmp_path):
        r = tc.post(
            "/v1/chat/completions",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert seen["json"]["stream"] is True
        assert seen["url"].endswith("/responses")
        assert r.json()["choices"][0]["finish_reason"] == "stop"


def test_model_defaults_when_missing(mock_upstream, tmp_path):
    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    client, seen = mock_upstream
    with build_app_client(
        make_settings(data_dir=str(tmp_path), default_model="custom-resp"),
        client,
        seed_key=TEST_SECRET,
    ) as tc:
        r = tc.post("/v1/responses", json={"input": "hi"}, headers=TEST_HEADERS)
        assert r.status_code == 200
        assert seen["json"]["model"] == "custom-resp"


def test_bare_responses_alias(app_client):
    tc, seen = app_client
    r = tc.post(
        "/responses",
        json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["json"]["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }
    ]


def test_invalid_json_rejected(app_client):
    tc, _ = app_client
    r = tc.post(
        "/v1/responses",
        content="not-json",
        headers={"Content-Type": "application/json", **TEST_HEADERS},
    )
    assert r.status_code == 400


def test_upstream_error_forwarded(app_client, mock_upstream):
    tc, _seen = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "upstream_error"
    r = tc.post(
        "/v1/responses",
        json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
        headers=TEST_HEADERS,
    )
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "AuthError"


def test_stream_strips_cost_frames(app_client, mock_upstream):
    tc, _seen = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "stream"
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "hi",
            "stream": True,
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert "inference-cost" not in r.text
    assert "response.output_text.delta" in r.text


def test_affinity_prefix_still_routes(app_client):
    # ak- in the path is unauthenticated bucket routing, not auth: the same
    # sk- header works with or without a prefix, and the client sk- never
    # leaks upstream (only the operator/public marker does).
    tc, seen = app_client
    for path in ("/v1/responses", "/ak-team1/v1/responses"):
        r = tc.post(
            path,
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200, path
        assert seen["url"].endswith("/responses")
    assert seen["headers"]["authorization"] == "Bearer public"


def test_tool_call_arguments_survive_synthesize(tmp_path):
    """Multi-delta tool args (incl. workdir) reassemble byte-exact.

    Non-streaming downstream on the anonymous responses leg folds
    upstream SSE; argument chunks must concatenate verbatim or the
    harness rejects the call (e.g. missing workdir).
    """
    import json as _json

    import httpx

    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    args = '{"command":"pwd","description":"x","workdir":"/tmp"}'
    chunks = [args[:17], args[17:]]

    async def handler(request):
        body = (
            'data: {"type":"response.output_item.added","output_index":1,'
            '"item":{"id":"call_wd1","type":"function_call","name":"bash",'
            '"arguments":"{}"}}\n\n'
            + "".join(
                f"data: {_json.dumps({'type': 'response.function_call_arguments.delta', 'output_index': 1, 'item_id': 'call_wd1', 'delta': c})}\n\n"
                for c in chunks
            )
            + 'data: {"type":"response.completed","response":{"status":"completed",'
            '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/chat/completions",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "messages": [{"role": "user", "content": "pwd please"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "description": "run",
                            "parameters": {},
                        },
                    }
                ],
            },
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        calls = r.json()["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"] == args


def test_redact_headers_drops_secrets():
    from llms.proxy.logging import redact_headers

    out = redact_headers(
        {
            "Authorization": "Bearer operator-secret-xyz",
            "x-api-key": "sk-live-abc123",
            "x-opencode-session": "ses_keep",
            "content-type": "application/json",
        }
    )
    assert out["Authorization"] == "<redacted>"
    assert out["x-api-key"] == "<redacted>"
    assert out["x-opencode-session"] == "ses_keep"
    assert out["content-type"] == "application/json"


def test_sk_guessing_earns_cooldown(app_client):
    """20 auth failures in a minute earn a 429 + retry-after, not a 401."""
    from llms.proxy import middleware as _mw

    tc, _ = app_client
    _mw._auth_failures.clear()
    _mw._auth_blocked_until.clear()
    try:
        last = None
        for _ in range(25):
            last = tc.post(
                "/v1/responses",
                json={"input": "hi"},
                headers={"Authorization": "Bearer sk-guess-same-prefix-0000"},
            )
            assert last.status_code in (401, 429), last.status_code
        assert last.status_code == 429
        assert last.headers.get("retry-after") == "60"
    finally:
        _mw._auth_failures.clear()
        _mw._auth_blocked_until.clear()


def test_sk_brake_scopes_to_key_prefix(app_client):
    """One bad actor's prefix must not 429 legitimate users' keys."""
    from llms.proxy import middleware as _mw

    tc, _ = app_client
    _mw._auth_failures.clear()
    _mw._auth_blocked_until.clear()
    try:
        for _ in range(25):
            r = tc.post(
                "/v1/responses",
                json={"input": "hi"},
                headers={"Authorization": "Bearer sk-evil-spray-0000"},
            )
            assert r.status_code in (401, 429)
        assert r.status_code == 429
        # Same source, different key prefix: still a plain 401.
        legit = tc.post(
            "/v1/responses",
            json={"input": "hi"},
            headers={"Authorization": "Bearer sk-legit-user-key"},
        )
        assert legit.status_code == 401
    finally:
        _mw._auth_failures.clear()
        _mw._auth_blocked_until.clear()


def test_streaming_steer_folds_undeclared_call_end_to_end(tmp_path):
    """POST stream:true, model calls undeclared 'shell': steered in-stream.

    Same wire shape as the live codex capture (stream:true, client tools
    exec_command, model calls overlay 'shell'): downstream SSE carries no
    executable shell call; upstream sees exactly 2 posts (turn + steer
    follow-up with function_call_output for shell). The clean turn replays
    upstream's own bytes verbatim (call_id/status fields, upstream
    response id) — a fresh emitter drops those fields and codex cannot
    dispatch the call (live probe finding: steered turn replayed but the
    client executed nothing).
    """
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        if len(calls) == 1:
            body = (
                'data: {"type":"response.output_item.added","output_index":1,'
                '"item":{"id":"call_shell1","type":"function_call","name":"shell",'
                '"arguments":"{}","call_id":"call_shell1",'
                '"status":"in_progress"}}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"output_index":1,"item_id":"call_shell1",'
                '"delta":"{\\"command\\":\\"echo hi\\"}"}\n\n'
                'data: {"type":"response.completed",'
                '"response":{"id":"resp_dead1","status":"completed",'
                '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
            )
        else:
            body = (
                'data: {"type":"response.output_item.added","output_index":0,'
                '"item":{"id":"call_exec1","type":"function_call",'
                '"name":"exec_command","arguments":"{}","call_id":"call_exec1",'
                '"status":"in_progress"}}\n\n'
                'data: {"type":"response.function_call_arguments.delta",'
                '"output_index":0,"item_id":"call_exec1",'
                '"delta":"{\\"command\\":\\"echo tool-ok\\"}"}\n\n'
                'data: {"type":"response.completed",'
                '"response":{"id":"resp_clean1","status":"completed",'
                '"usage":{"input_tokens":5,"output_tokens":6,"total_tokens":11}}}\n\n'
            )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert '"name":"shell"' not in r.text
    assert '"name": "shell"' not in r.text
    # Clean-turn replay: the call id and upstream response id survive
    # (a fresh emitter drops those fields and codex cannot dispatch
    # the call). Frame bytes normalize through the JSON round-trip
    # (`"call_id": "..."` spacing), so match the spaced form.
    assert '"call_id": "call_exec1"' in r.text
    assert "resp_clean1" in r.text
    assert len(calls) == 2
    followup = calls[1]
    outputs = [
        i
        for i in followup["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert outputs and outputs[0]["call_id"] == "call_shell1"
    assert "exec_command" in outputs[0]["output"]


def test_streaming_owned_name_converts_casing_and_passes_through(tmp_path):
    """Client declares 'Shell'; model calls 'shell' with required keys:
    downstream SSE carries the renamed 'Shell' call, exactly 1 upstream
    call (streaming legs replay SSE bytes, not JSON)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_shell1", "shell", '{"cmd":"echo hi"}'),
        _text_sse("unreached"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "Shell",
                        "description": "mine",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    assert '"name": "Shell"' in r.text
    assert '"name": "shell"' not in r.text
    assert '"name":"Shell"' in r.text or '"name": "Shell"' in r.text
    assert '"id": "call_shell1"' in r.text
    assert "echo hi" in r.text


def test_streaming_owned_name_missing_key_steers(tmp_path):
    """Same setup but args lack required 'cmd': 2 upstream calls, redirect
    is an argument correction (not not-available), followup function_call
    args are valid JSON upstream."""
    import json as _json

    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_shell1", "shell", '{"command":"echo hi"}'),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "Shell",
                        "description": "mine",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    followup = calls[1]
    outputs = [
        i
        for i in followup["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert outputs and "Shell" in outputs[0]["output"]
    echoed = [
        i
        for i in followup["input"]
        if isinstance(i, dict) and i.get("type") == "function_call"
    ]
    assert echoed
    for item in echoed:
        _json.loads(item["arguments"])  # never raw "" upstream


def test_owned_name_converts_casing_and_passes_through_synthesize(tmp_path):
    """Non-streaming mirror: declared 'Shell', model calls 'shell' with
    required keys — converted name, exactly 1 upstream call."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_shell1", "shell", '{"cmd":"echo hi"}'),
        _text_sse("unreached"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "tools": [
                    {
                        "type": "function",
                        "name": "Shell",
                        "description": "mine",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    returned = [
        i for i in r.json().get("output", []) if i.get("type") == "function_call"
    ]
    assert len(returned) == 1
    assert returned[0]["name"] == "Shell"
    assert returned[0]["arguments"] == '{"cmd":"echo hi"}'


def test_owned_name_missing_key_steers_synthesize(tmp_path):
    """Non-streaming mirror: args lack required 'cmd' — steers with an
    argument correction naming 'Shell' and the missing key (not a
    not-available redirect)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_shell1", "shell", '{"command":"echo hi"}'),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "tools": [
                    {
                        "type": "function",
                        "name": "Shell",
                        "description": "mine",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    followup = calls[1]
    outputs = [
        i
        for i in followup["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert outputs and "Shell" in outputs[0]["output"]


def _empty_args_tool_call_sse(call_id: str, name: str) -> bytes:
    """Upstream SSE emitting a function_call whose wire arguments are "".

    Fires the followup-coercion branch on both steer paths: the steer
    followup must replay valid JSON ("{}"), never raw "", upstream.
    """
    return _tool_call_sse(call_id, name, "")


def test_streaming_empty_args_steer_coerces_to_empty_object(tmp_path):
    """Streaming: upstream call with "" args steers; the followup replays
    "{}" (not raw "") so the upstream validator never 400s."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _empty_args_tool_call_sse("call_shell1", "shell"),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "Shell",
                        "description": "mine",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    echoed = [
        i
        for i in calls[1]["input"]
        if isinstance(i, dict) and i.get("type") == "function_call"
    ]
    assert echoed
    assert all(item["arguments"] == "{}" for item in echoed)


def test_synthesize_empty_args_steer_coerces_to_empty_object(tmp_path):
    """Non-streaming mirror: "" args steer; followup replays "{}"."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _empty_args_tool_call_sse("call_shell1", "shell"),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "tools": [
                    {
                        "type": "function",
                        "name": "Shell",
                        "description": "mine",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    echoed = [
        i
        for i in calls[1]["input"]
        if isinstance(i, dict) and i.get("type") == "function_call"
    ]
    assert echoed
    assert all(item["arguments"] == "{}" for item in echoed)


def test_owned_missing_key_redirect_corrects_arguments(tmp_path):
    """Owned-but-wrong-keys redirect says the tool exists and names the
    missing keys + declared shape — never 'not available' (which
    contradicts the tool list the model was given)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_shell1", "shell", '{"command": "echo hi"}'),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "tools": [
                    {
                        "type": "function",
                        "name": "Shell",
                        "description": "mine",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    outputs = [
        i
        for i in calls[1]["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert outputs
    text = outputs[0]["output"]
    assert "'Shell'" in text and "not available" not in text
    assert "cmd" in text and "Retry" in text


def test_genuine_nonstreaming_valid_turn_never_steers(tmp_path):
    """Genuine opencode, valid call, non-streaming: exactly 1 upstream
    call — no redirect round-trip, no redirect text (fidelity)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_read1", "read", '{"path":"/tmp/x"}'),
        _text_sse("unreached"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "instructions": "You are opencode.",
                "input": "hi",
                "tools": [
                    {
                        "type": "function",
                        "name": "read",
                        "description": "d",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                ],
            },
            headers=dict(
                TEST_HEADERS,
                **{
                    "User-Agent": "opencode/latest/2.0.12/cli",
                    "x-opencode-client": "cli",
                    "x-opencode-project": "global",
                    "x-opencode-session": "ses_abcdef1234567890abcdefghij12",
                },
            ),
        )
    assert r.status_code == 200
    assert len(calls) == 1
    # Genuine path: no notice appended, valid call passes verbatim.
    assert "'read'" not in calls[0].get("instructions", "")
    returned = [
        i for i in r.json().get("output", []) if i.get("type") == "function_call"
    ]
    assert len(returned) == 1
    assert returned[0]["name"] == "read"


def test_deferred_only_tools_notice_lists_them_e2e(app_client):
    """Request carrying only `additional_tools` items: the outbound notice
    (in instructions) lists the dissolved deferred tool."""
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "instructions": "Be brief.",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
                {
                    "type": "additional_tools",
                    "id": "at_1",
                    "role": "developer",
                    "tools": [
                        {
                            "type": "function",
                            "name": "deferred_exec",
                            "description": "run it",
                            "parameters": {
                                "type": "object",
                                "properties": {"cmd": {"type": "string"}},
                                "required": ["cmd"],
                            },
                        }
                    ],
                },
            ],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    # No additional_tools item rides upstream (Zen rejects the shape)...
    assert not [
        i
        for i in seen["json"].get("input", [])
        if isinstance(i, dict) and i.get("type") == "additional_tools"
    ]
    # ...but the dissolved tool is named in the notice (instructions),
    # never in outbound tools (genuine-12 ONLY — client extras must not
    # ride the wire where the upstream validator can 400 them).
    names = [
        t.get("name")
        for t in seen["json"].get("tools", [])
        if isinstance(t, dict) and t.get("name")
    ]
    assert "deferred_exec" not in names
    assert "'deferred_exec'" in seen["json"]["instructions"]


def _mixed_turn_sse() -> bytes:
    """One upstream body: declared exec_command + undeclared shell calls."""
    return (
        b'data: {"type":"response.output_item.added","output_index":0,'
        b'"item":{"id":"call_exec9","type":"function_call","name":"exec_command",'
        b'"arguments":"{}","call_id":"call_exec9","status":"in_progress"}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta",'
        b'"output_index":0,"item_id":"call_exec9",'
        b'"delta":"{\\"cmd\\":\\"echo kept\\"}"}\n\n'
        b'data: {"type":"response.output_item.added","output_index":1,'
        b'"item":{"id":"call_shell9","type":"function_call","name":"shell",'
        b'"arguments":"{}","call_id":"call_shell9","status":"in_progress"}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta",'
        b'"output_index":1,"item_id":"call_shell9",'
        b'"delta":"{\\"command\\":\\"echo dropped\\"}"}\n\n'
        b'data: {"type":"response.completed",'
        b'"response":{"id":"resp_mixed1","status":"completed",'
        b'"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
    )


def _clean_turn_sse() -> bytes:
    """Upstream clean turn answering after the redirect (no tool calls)."""
    return (
        b'data: {"type":"response.output_text.delta","delta":"answered"}\n\n'
        b'data: {"type":"response.completed",'
        b'"response":{"id":"resp_mixed2","status":"completed",'
        b'"usage":{"input_tokens":9,"output_tokens":3,"total_tokens":12}}}\n\n'
    )


def test_streaming_steer_mixed_turn_replaces_whole_dead_turn(tmp_path):
    """Mixed declared+undeclared turn: whole-turn replacement, like synthesize.

    The dead turn (even its declared exec_command frame) never emits;
    only the re-requested clean turn replays downstream, while the dead
    turn's calls — declared and undeclared alike — stay upstream as
    input history (the follow-up carries function_call_output for the
    undeclared call and the declared call is present for the model to
    re-issue). This matches _steer_genuine_calls whole-response
    replacement on the synthesize path: the model re-issues whatever
    declared call it still needs in the clean turn (observed live).
    """
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        body = _mixed_turn_sse() if len(calls) == 1 else _clean_turn_sse()
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    # Dead turn suppressed wholesale: neither call frame replays.
    assert "call_shell9" not in r.text
    assert "call_exec9" not in r.text
    assert "answered" in r.text
    assert len(calls) == 2
    followup = calls[1]
    outputs = [
        i
        for i in followup["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert [o["call_id"] for o in outputs] == ["call_shell9"]
    # Usage attributes the final turn only (single billing).
    usage = tc.app.state.usage.snapshot()["keys"][TEST_SECRET]
    assert usage["input_tokens"] == 9
    assert usage["output_tokens"] == 3


def test_streaming_fold_case_variant_declared_name_passes(tmp_path):
    """Fold path honors the case-insensitive contract: client "Read" owns
    upstream "read" — no steer follow-up (exactly 1 upstream call)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_read1", "read", '{"path":"notes.txt"}'),
        _text_sse("unreached"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "read notes",
                "stream": True,
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
    assert len(calls) == 1
    assert "call_read1" in r.text


def test_streaming_fold_genuine_client_passes_through(tmp_path):
    """Genuine opencode legs never fold-steer: even an undeclared-name
    call streams verbatim with exactly 1 upstream call (no shaping,
    no steer)."""
    import httpx

    from tests.conftest import TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        return httpx.Response(
            200,
            content=_tool_call_sse("call_shell1", "shell", '{"command":"x"}'),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    genuine_headers = {
        "Authorization": f"Bearer {TEST_SECRET}",
        "User-Agent": "opencode/2.0.22/cli",
        "x-opencode-client": "cli",
    }
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run x",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=genuine_headers,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    assert "call_shell1" in r.text


def test_streaming_steer_reports_chained_response_id(tmp_path):
    """Fold usage carries the replayed turn's upstream response id.

    streaming_steer_folds path must mirror TappedStream._record: the
    session tracker learns chain:<upstream-id> so the follow-up turn
    reuses the session and the prompt cache stays warm.
    """
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    async def handler(request):
        body = _clean_turn_sse().replace(b"resp_mixed2", b"resp_chain7")
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "say hi",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert "resp_chain7" in r.text
    assert tc.app.state.sessions.lookup(TEST_SECRET, "chain:resp_chain7") is not None


def test_streaming_steer_rerequest_error_signals_not_replays(tmp_path):
    """Steer re-request 500: downstream gets an error frame, not the dead turn.

    Replaying the violating turn on steer failure would emit exactly the
    undeclared-call bytes the feature exists to suppress; fail closed
    with a terminal SSE error frame (slow_send_stream precedent) while
    the usage sink records the dead turn as incomplete.
    """
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        if len(calls) == 1:
            return httpx.Response(
                200,
                content=_mixed_turn_sse(),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(500, content=b"upstream blew up")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert "call_shell9" not in r.text
    assert '"type": "error"' in r.text or '"type":"error"' in r.text
    assert len(calls) == 2
    # Fail-closed billing: sniffing the emptied lines yields an incomplete
    # StreamDone with no tokens, recorded with count_request=False — the
    # dead turn is never billed. The stowed stream_outcome keeps the
    # provider from promoting on a steered-then-failed turn and records
    # the recents error.
    usage = tc.app.state.usage.snapshot()["keys"][TEST_SECRET]
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0


def _undeclared_turn_sse(call_id: str, name: str) -> bytes:
    """Upstream SSE emitting one undeclared function_call turn (test helper)."""
    return _tool_call_sse(call_id, name, "{}")


def test_streaming_steer_exhaustion_fails_closed_with_redirect_text(tmp_path):
    """Steer budget spent on fresh undeclared names: no unjudged turn replays.

    Live shape: the model emits a NEW undeclared name every turn (shell,
    execute, read, glob), so the loop never sees a repeat and the budget
    dies — the 4th turn must not replay downstream (the client fails it
    with `unsupported call`). Instead the client gets the last redirect
    text as a terminal turn.
    """
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    names = ["shell", "execute", "read", "glob"]
    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        idx = len(calls) - 1
        body = _undeclared_turn_sse(f"call_ex{idx}", names[min(idx, 3)])
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    # No unjudged dead turn reaches the client: no function_call frames.
    assert "function_call" not in r.text
    # The client surfaces the last correction instead.
    assert "not available in this session" in r.text
    # Budget honored: 1 initial + STEER_MAX_ITERS re-requests.
    from llms.proxy.pipeline import STEER_MAX_ITERS

    assert len(calls) == 1 + STEER_MAX_ITERS


def test_synthesize_steer_exhaustion_fails_closed_with_redirect_json(tmp_path):
    """Synthesize-path mirror: budget exhaustion returns redirect text.

    The JSON body carries the last redirect as the assistant message —
    never a dead turn with calls the client cannot execute.
    """
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    names = ["shell", "execute", "read", "glob"]
    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        idx = len(calls) - 1
        body = _undeclared_turn_sse(f"call_ex{idx}", names[min(idx, 3)])
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {"type": "object"},
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    body = r.json()
    assert not [i for i in body.get("output", []) if i.get("type") == "function_call"]
    texts = [
        c.get("text", "")
        for i in body.get("output", [])
        if i.get("type") == "message"
        for c in i.get("content", [])
    ]
    assert texts and "not available in this session" in texts[0]
    from llms.proxy.pipeline import STEER_MAX_ITERS

    assert len(calls) == 1 + STEER_MAX_ITERS


def test_genuine_shell_translates_onto_client_exec_command(tmp_path):
    """Genuine-overlay `shell` with usable args translates onto the
    client's `exec_command` (same capability, client name) instead of
    steering: downstream SSE carries the renamed call, exactly 1
    upstream call (no steer follow-up)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_shell1", "shell", '{"command": "echo hi"}'),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "run echo",
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    returned = [
        i for i in r.json().get("output", []) if i.get("type") == "function_call"
    ]
    assert len(returned) == 1
    assert returned[0]["name"] == "exec_command"
    assert returned[0]["arguments"] == '{"cmd": "echo hi"}'


def test_streaming_nested_exec_command_replays_as_exec_custom_call(tmp_path):
    """Streaming mirror: the same nested-call replay applies frame by
    frame — the added item frame re-types to custom_tool_call named
    `exec` with the JS input, and the delta/done frames follow."""
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings
    from tests.test_client_tools import _luna_namespace

    ns = _luna_namespace()
    nested = ns.options["tools"]
    exec_desc = next(t["description"] for t in nested if t["name"] == "exec")
    # Live luna shape: the deferred namespace rides the input as an
    # additional_tools item (dissolved into req.tools by from_responses),
    # not as a top-level tool.
    tools = [
        {
            "type": "additional_tools",
            "id": "at_1",
            "role": "developer",
            "tools": [
                {
                    "type": "namespace",
                    "name": "functions",
                    "description": "",
                    "tools": [
                        {"type": "custom", "name": "exec", "description": exec_desc},
                        {
                            "type": "function",
                            "name": "wait",
                            "description": "Wait.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    ],
                }
            ],
        }
    ]
    calls: list = []

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        body = (
            'data: {"type":"response.output_item.added","output_index":1,'
            '"item":{"id":"call_n1","type":"function_call",'
            '"name":"exec_command",'
            '"arguments":"{\\"cmd\\": \\"echo hi\\"}",'
            '"call_id":"call_n1","status":"in_progress"}}\n\n'
            'data: {"type":"response.completed",'
            '"response":{"id":"resp_n1","status":"completed",'
            '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-luna",
                "input": ["run echo", tools[0]],
                "stream": True,
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    assert '"name": "exec"' in r.text
    assert '"type": "custom_tool_call"' in r.text
    assert "await tools.exec_command" in r.text


def test_nested_exec_command_replays_as_exec_custom_call(tmp_path):
    """Luna leg: a valid nested `exec_command` call replays downstream
    as ONE `custom_tool_call` named `exec` whose input is the JS
    invocation (the harness only executes nested tools through the
    `exec` orchestrator — a bare nested call fails lookup)."""
    from tests.conftest import TEST_HEADERS
    from tests.test_client_tools import _luna_namespace

    ns = _luna_namespace()
    nested = ns.options["tools"]
    exec_desc = next(t["description"] for t in nested if t["name"] == "exec")
    tools = [
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "tools": [
                {"type": "custom", "name": "exec", "description": exec_desc},
                {
                    "type": "function",
                    "name": "wait",
                    "description": "Wait.",
                    "parameters": {"type": "object", "properties": {}},
                },
            ],
        }
    ]
    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_nested1", "exec_command", '{"cmd": "echo hi"}'),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-luna",
                "input": "run echo",
                "tools": tools,
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    customs = [
        i for i in r.json().get("output", []) if i.get("type") == "custom_tool_call"
    ]
    assert len(customs) == 1
    assert customs[0]["name"] == "exec"
    assert "await tools.exec_command" in customs[0]["input"]


def test_steer_redirect_plain_leg_omits_nested_channel(tmp_path):
    """Plain function leg: the undeclared-name redirect must not teach
    the `exec` custom_tool_call channel (that harness exposes no `exec`
    — the text would contradict the usable tool list)."""
    from llms.proxy.ir import ToolDef
    from llms.proxy.pipeline import _steer_output_for, owned_tool_names

    shell = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    tools = (shell,)
    text = _steer_output_for(
        {"name": "glob", "call_id": "c1"},
        "{}",
        {"exec_command"},
        owned_tool_names(tools),
        {"exec_command": shell},
        tools,
    )
    assert "not available in this session" in text
    assert "custom_tool_call" not in text
    assert "answer directly without calling a tool" in text


def test_steer_redirect_nested_leg_keeps_exec_channel():
    """Deferred-namespace leg: the redirect keeps the `exec` channel
    guidance (the harness runs nested tools through it)."""
    from llms.proxy.pipeline import _steer_output_for, owned_tool_names
    from tests.test_client_tools import _luna_namespace

    ns = (_luna_namespace(),)
    text = _steer_output_for(
        {"name": "glob", "call_id": "c1"},
        "{}",
        {"exec_command"},
        owned_tool_names(ns),
        {},
        ns,
    )
    assert "not available in this session" in text
    assert "custom_tool_call" in text and "`exec`" in text


def test_genuine_read_steers_with_cat_redirect_not_view_image(tmp_path):
    """Spark-leg regression (live probe): undeclared `read` must NOT
    rename onto `view_image` (an image-path viewer — the harness
    fails it client-side). It steers, and the follow-up upstream turn
    carries the directed shell/command redirect (1 steer re-request,
    then the clean turn replays — the client runner has no wire
    schema, so the redirect names upstream `shell`; live spark A/B
    2026-10-06)."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_read1", "read", '{"path": "/tmp/f.txt"}'),
        _tool_call_sse("call_exec1", "exec_command", '{"cmd": "cat /tmp/f.txt"}')
        + _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "read file",
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                    {
                        "type": "function",
                        "name": "view_image",
                        "description": "view",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    # Steer re-request happened (dead turn + redirect replayed
    # upstream), and the redirect named the cat invocation — never a
    # view_image rename.
    assert len(calls) == 2
    followup = calls[1]
    outputs = [
        i for i in followup.get("input", []) if i.get("type") == "function_call_output"
    ]
    assert len(outputs) == 1
    assert "shell" in outputs[0]["output"]
    assert '"command"' in outputs[0]["output"]
    assert "cat /tmp/f.txt" in outputs[0]["output"]
    assert "view_image" not in outputs[0]["output"]
    assert "Retry the call as 'shell'" in outputs[0]["output"]
    returned = [
        i for i in r.json().get("output", []) if i.get("type") == "function_call"
    ]
    assert len(returned) == 1
    assert returned[0]["name"] == "exec_command"


def test_streaming_delta_shell_translates_without_steer(tmp_path):
    """Live spark shape (resp-45 turn 1): added frame announces with
    arguments:"", then delta frame(s) carry the genuine payload, then
    the done frame repeats it. The fold prefers the deltas — so the
    replay must swap the delta payload too, not just added + done:
    stale `command` deltas refold as exec_command + `{"command":...}`
    (missing `cmd`) and the tail check fails closed after a clean
    first fold. Exactly 1 upstream call (no steer follow-up)."""
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    calls: list = []
    body = (
        'data: {"type":"response.output_item.added","output_index":2,'
        '"item":{"id":"fc_live1","type":"function_call","status":"in_progress",'
        '"name":"shell","call_id":"call_live1","arguments":""}}\n\n'
        'data: {"type":"response.function_call_arguments.delta",'
        '"output_index":2,"item_id":"fc_live1",'
        '"delta":"{\\"command\\":\\"cat /tmp/f.txt\\"}"}\n\n'
        'data: {"type":"response.function_call_arguments.done",'
        '"output_index":2,"item_id":"fc_live1",'
        '"arguments":"{\\"command\\":\\"cat /tmp/f.txt\\"}","name":"shell"}\n\n'
        'data: {"type":"response.completed",'
        '"response":{"id":"resp_live1","status":"completed",'
        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
    )

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "read file",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    # Translated replay: added + delta + done frames all carry the
    # translated arguments, and no stale genuine payload survives.
    assert '"name": "exec_command"' in r.text
    assert "cat /tmp/f.txt" in r.text
    assert "command" not in r.text.replace("exec_command", "")
    assert '"name":"shell"' not in r.text
    assert '"name": "shell"' not in r.text


def test_streaming_done_only_shell_translates_without_steer(tmp_path):
    """Live Zen shape (resp-26): added frame announces with
    arguments:"", NO delta frames, payload rides solely the done
    frame. The fold must judge the done payload (shell+command ->
    exec_command/cmd translation) — not steer a valid call as
    owned-but-invalid. Exactly 1 upstream call (no steer follow-up)."""
    import httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    calls: list = []
    body = (
        'data: {"type":"response.output_item.added","output_index":2,'
        '"item":{"id":"fc_live1","type":"function_call","status":"in_progress",'
        '"name":"shell","call_id":"call_live1","arguments":""}}\n\n'
        'data: {"type":"response.function_call_arguments.done",'
        '"output_index":2,"item_id":"fc_live1",'
        '"arguments":"{\\"command\\":\\"cat /tmp/f.txt\\"}","name":"shell"}\n\n'
        'data: {"type":"response.completed",'
        '"response":{"id":"resp_live1","status":"completed",'
        '"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}\n\n'
    )

    async def handler(request):
        import json as _json

        calls.append(_json.loads(request.content.decode()))
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        _make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "read file",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 1
    # Translated replay: the added frame's name flips to the client
    # tool (re-serialized frames use the spaced form), the done
    # frame is renamed with the translated payload too (it carries
    # no `item` object, so the added-frame branch never matches it),
    # and no shell frame survives.
    assert '"name": "exec_command"' in r.text
    assert "cat /tmp/f.txt" in r.text
    assert '"name":"shell"' not in r.text
    assert '"name": "shell"' not in r.text


def test_streaming_namespaced_emitter_gets_argument_correction(tmp_path):
    """Live spark shape: declared bare `exec_command`, model emits
    `default.exec_command` with `{}`. The classifier must treat it as
    owned-but-invalid (argument correction naming the upstream `shell`
    alias + `command` — the client runner has no wire schema, so a
    retry-as-`exec_command` names keys the model cannot fill; live
    spark A/B 2026-10-06), NOT as an undeclared name (generic
    not-available text): the generic text never converted the live
    emitter, which repeated `{}` x3 then the budget died. Exactly 2
    upstream calls on the mock leg."""
    from tests.conftest import TEST_HEADERS

    tc_ctx, calls = _steer_app_client(
        tmp_path,
        _tool_call_sse("call_ns1", "default.exec_command", "{}"),
        _text_sse("done"),
    )
    with tc_ctx as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "print file",
                "stream": True,
                "tools": [
                    {
                        "type": "function",
                        "name": "exec_command",
                        "description": "run",
                        "parameters": {
                            "type": "object",
                            "properties": {"cmd": {"type": "string"}},
                            "required": ["cmd"],
                        },
                    },
                ],
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    assert len(calls) == 2
    outputs = [
        i
        for i in calls[1]["input"]
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert outputs
    text = outputs[0]["output"]
    # Owned-but-invalid flavor: names the emitted form + missing key,
    # never the generic undeclared text.
    assert "'default.exec_command'" in text
    assert "Retry as 'shell'" in text and '"command"' in text
    assert "not available" not in text
