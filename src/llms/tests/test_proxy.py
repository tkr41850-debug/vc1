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


def test_anonymous_instructions_lead_with_canonical_prefix(app_client):
    from llms.proxy.zen_prompts import SEAM, TITLE_PREFIX

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
    sent = seen["json"]["instructions"]
    assert sent.startswith(TITLE_PREFIX)
    assert sent.endswith(SEAM + "Be brief.")


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
