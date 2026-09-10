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


def test_chat_model_defaults_when_missing(mock_upstream):
    from tests.conftest import build_app_client, make_settings

    client, seen = mock_upstream
    with build_app_client(
        make_settings(default_chat_model="custom-chat"), client
    ) as tc:
        r = tc.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
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
    )
    assert r.status_code == 200
    assert "harness_session" not in seen["json"]
