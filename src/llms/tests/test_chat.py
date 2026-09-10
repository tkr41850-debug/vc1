from __future__ import annotations

from tests.conftest import TEST_HEADERS


def test_chat_passthrough_with_model(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert seen["url"].endswith("/chat/completions")
    assert seen["json"]["model"] == "mimo-v2.5-free"
    assert seen["json"]["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_model_defaults_when_missing(mock_upstream, tmp_path):
    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    client, seen = mock_upstream
    with build_app_client(
        make_settings(data_dir=str(tmp_path), default_chat_model="custom-chat"),
        client,
        seed_key=TEST_SECRET,
    ) as tc:
        r = tc.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert seen["json"]["model"] == "custom-chat"
        assert seen["url"].endswith("/chat/completions")


def test_chat_drops_unknown_fields(app_client):
    tc, seen = app_client
    r = tc.post(
        "/v1/chat/completions",
        json={
            "model": "mimo-v2.5-free",
            "messages": [{"role": "user", "content": "hi"}],
            "harness_session": "abc",
        },
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    assert "harness_session" not in seen["json"]
