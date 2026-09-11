from __future__ import annotations

from tests.conftest import TEST_HEADERS


def test_messages_passthrough(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 64,
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/messages")
    assert seen["json"]["messages"] == [{"role": "user", "content": "hi"}]
    body = r.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "hello"}]


def test_messages_model_defaults(mock_upstream, tmp_path):
    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    client, seen = mock_upstream
    with build_app_client(
        make_settings(data_dir=str(tmp_path), default_messages_model="custom-msg"),
        client,
        seed_key=TEST_SECRET,
    ) as tc:
        r = tc.post(
            "/v1/messages",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert seen["json"]["model"] == "custom-msg"
        assert seen["json"]["max_tokens"] == 1024
