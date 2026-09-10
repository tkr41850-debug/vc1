from __future__ import annotations


def test_messages_passthrough(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        },
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/messages")
    assert seen["json"]["messages"] == [{"role": "user", "content": "hi"}]
    body = r.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "hello"}]


def test_messages_model_defaults(app_client):
    tc, seen = app_client
    r = tc.post("/v1/messages", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert seen["json"]["model"] == "claude-haiku-4-5"
    assert seen["json"]["max_tokens"] == 1024
