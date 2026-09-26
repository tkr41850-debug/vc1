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
        assert "unparsable keys.yaml" in r.json()["store_error"]
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
