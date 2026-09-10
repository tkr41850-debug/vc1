from __future__ import annotations


def test_chat_passthrough_with_model(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert seen["json"]["model"] == "mimo-v2.5-free"
    assert seen["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_model_defaults_when_missing(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert r.status_code == 200
    assert seen["json"]["model"] == "muse-spark-1.3-contributor-free"
    assert seen["url"].endswith("/responses")


def test_chat_drops_unknown_fields(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [{"role": "user", "content": "hi"}],
            "harness_session": "abc",
        },
    )
    assert r.status_code == 200
    assert "harness_session" not in seen["json"]
