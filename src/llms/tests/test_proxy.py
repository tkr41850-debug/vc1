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


def test_anonymous_tooled_turn_sends_genuine_superset(app_client):
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
    assert [t["name"] for t in sent_tools[: len(GENUINE_TOOLS)]] == [
        t["name"] for t in GENUINE_TOOLS
    ]
    assert sent_tools[-1]["name"] == "bash"
    assert seen["json"]["instructions"] == "Be brief."


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
    """Client "Read" owns upstream "read": returned verbatim for the
    client to resolve — no steer follow-up (exactly 1 upstream call)."""
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
    assert returned[0]["name"] == "read"
    assert returned[0]["call_id"] == "call_read1"
    assert returned[0]["arguments"] == '{"path":"notes.txt"}'


def test_client_named_genuine_tool_call_passes_through_unsteered(tmp_path):
    """A call naming a client-declared tool colliding with a genuine name
    (read) is returned verbatim: no steer follow-up (exactly 1 upstream
    call), and upstream saw the CLIENT's read definition."""
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
    sent = {t.get("name"): t for t in calls[0]["tools"]}
    assert sent["read"]["description"] == "mine"
    assert sent["mine"]["description"] == "extra"
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
    # Verbatim replay: the clean turn's own call bytes (not re-framed).
    assert '"call_id":"call_exec1"' in r.text
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
