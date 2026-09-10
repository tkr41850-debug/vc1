from __future__ import annotations


def test_chat_ingress_routes_spark_to_responses(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions",
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
    r = tc.post("/v1/responses", json={"model": "mimo-v2.5-free", "input": "hi"})
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert seen["json"]["messages"] == [{"role": "user", "content": "hi"}]
    body = r.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"


def test_same_dialect_passes_through_untouched(app_client):
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
    assert r.json()["choices"][0]["message"]["content"] == "hello"


def test_cross_dialect_stream_rejected_for_now(app_client):
    tc, _ = app_client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "muse-spark-1.3-contributor-free",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert r.status_code == 400
