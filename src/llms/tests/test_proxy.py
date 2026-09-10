from __future__ import annotations


def test_healthz(app_client):
    tc, _ = app_client
    r = tc.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_non_stream_passthrough_with_model_override(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/responses",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "input": "say hi",
            "stream": False,
        },
    )
    assert r.status_code == 200
    assert r.json()["output_text"] == "hello"
    assert seen["url"].endswith("/responses")
    assert seen["json"]["model"] == "muse-spark-1.3-contributor-free"
    assert seen["headers"]["x-opencode-client"] == "cli"
    assert seen["headers"]["user-agent"] == "opencode/1.18.4"


def test_model_defaults_when_missing(app_client):
    tc, seen = app_client
    r = tc.post("/v1/responses", json={"input": "hi"})
    assert r.status_code == 200
    assert seen["json"]["model"] == "muse-spark-1.3-contributor-free"


def test_bare_responses_alias(app_client):
    tc, seen = app_client
    r = tc.post(
        "/responses", json={"model": "muse-spark-1.3-contributor-free", "input": "hi"}
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
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_upstream_error_forwarded(app_client, mock_upstream):
    tc, _seen = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "upstream_error"
    r = tc.post(
        "/v1/responses",
        json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
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
    )
    assert r.status_code == 200
    assert "inference-cost" not in r.text
    assert "response.output_text.delta" in r.text


def test_deepseek_harness_style_override(app_client):
    tc, seen = app_client
    harness_payload = {
        "model": "deepseek-v4-flash",
        "input": [{"role": "user", "content": "write a python function"}],
        "stream": False,
        "max_output_tokens": 64,
    }
    harness_payload["model"] = "muse-spark-1.3-contributor-free"
    r = tc.post(
        "/v1/responses",
        json=harness_payload,
        headers={"Authorization": "Bearer test-key"},
    )
    assert r.status_code == 200
    assert seen["json"]["model"] == "muse-spark-1.3-contributor-free"
    assert "authorization" not in seen["headers"]
