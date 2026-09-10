from __future__ import annotations


def test_chat_ingress_routes_spark_to_responses(app_client):
    tc, seen = app_client
    r = tc.post(
        "/ak-test/v1/chat/completions",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/responses")
    assert seen["json"]["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello"


def test_responses_ingress_routes_mimo_to_chat(app_client):
    tc, seen = app_client
    r = tc.post(
        "/ak-test/v1/responses", json={"model": "mimo-v2.5-free", "input": "hi"}
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert seen["json"]["messages"] == [{"role": "user", "content": "hi"}]
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"


def test_same_dialect_passes_through_untouched(app_client):
    tc, seen = app_client
    r = tc.post(
        "/ak-test/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert r.json()["choices"][0]["message"]["content"] == "hello"


def test_cross_dialect_stream_translates(app_client, mock_upstream):
    tc, _seen = app_client
    _, seen_dict = mock_upstream
    seen_dict["mode"] = "stream"
    r = tc.post(
        "/ak-test/v1/chat/completions",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert "hi" in r.text
    assert "data: [DONE]" in r.text
    assert "inference-cost" not in r.text


def test_messages_ingress_routes_spark_to_responses(app_client):
    tc, seen = app_client
    r = tc.post(
        "/ak-test/v1/messages",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        },
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/responses")
    body = r.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "hello"}]


def test_responses_ingress_routes_claude_to_messages(app_client):
    tc, seen = app_client
    r = tc.post(
        "/ak-test/v1/responses", json={"model": "claude-haiku-4-5", "input": "hi"}
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/messages")
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"


def test_model_alias_remaps_before_routing(mock_upstream):
    from tests.conftest import build_app_client, make_settings

    client, seen = mock_upstream
    settings = make_settings(
        model_aliases=(
            ("gpt-*", "muse-spark-1.3-contributor-free"),
            ("claude-*", "muse-spark-1.3-contributor-free"),
        )
    )
    with build_app_client(settings, client) as tc:
        r = tc.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        assert seen["url"].endswith("/responses")
        assert seen["json"]["model"] == "muse-spark-1.3-contributor-free"
