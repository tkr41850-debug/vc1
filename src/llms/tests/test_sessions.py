from __future__ import annotations

import re
import time

from llms.proxy.sessions import (
    SessionTracker,
    conversation_ref,
    mint_session_id,
)


def test_mint_matches_genuine_shape():
    assert re.fullmatch(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}", mint_session_id())
    assert mint_session_id() != mint_session_id()


def test_conversation_ref_previous_response_id():
    assert (
        conversation_ref("responses", {"previous_response_id": "resp_1"}, {})
        == "chain:resp_1"
    )


def test_conversation_ref_codex_thread():
    headers = {
        "originator": "codex_exec",
        "thread-id": "thread-9",
        "session-id": "sess-9",
    }
    assert conversation_ref("responses", {}, headers) == "codex-thread:thread-9"


def test_conversation_ref_ignores_foreign_session_id():
    assert conversation_ref("responses", {}, {"session-id": "sess-9"}) is None
    assert conversation_ref("chat", {"previous_response_id": "x"}, {}) is None
    assert conversation_ref("responses", {}, {}) is None


def test_tracker_lookup_remember_expiry():
    tracker = SessionTracker(ttl_s=0.05)
    assert tracker.lookup("k", "chain:r1") is None
    tracker.remember("k", "chain:r1", "ses_abc")
    assert tracker.lookup("k", "chain:r1") == "ses_abc"
    assert tracker.lookup("other", "chain:r1") is None
    time.sleep(0.08)
    assert tracker.lookup("k", "chain:r1") is None


def test_chained_turns_reuse_session(mock_upstream, tmp_path):
    """A previous_response_id continuation rides the first turn's session."""
    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    client, seen = mock_upstream
    seen["mode"] = "stream"
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        first = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert first.status_code == 200
        first_session = seen["headers"]["x-opencode-session"]
        second = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "hi again",
                "previous_response_id": first.json()["id"],
            },
            headers=TEST_HEADERS,
        )
        assert second.status_code == 200
        assert seen["headers"]["x-opencode-session"] == first_session


def test_warming_fires_for_new_conversation(mock_upstream, tmp_path, monkeypatch):
    """New responses conversations background a title-shaped warming call."""
    import json as _json
    import time as _time

    import httpx

    from llms.proxy import sessions
    from llms.proxy.zen_prompts import TITLE_PREFIX
    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    monkeypatch.setattr(sessions, "SESSION_WARMING", True)
    calls: list = []

    async def handler(request):
        payload = _json.loads(request.content.decode())
        calls.append((request.headers.get("x-opencode-session"), payload))
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
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        for _ in range(100):
            if len(calls) >= 2:
                break
            _time.sleep(0.05)
        assert len(calls) >= 2, "warming call never arrived"
        main_session, _ = calls[0]
        warm_session, warm_body = calls[1]
        assert warm_session == main_session
        assert warm_body["prompt_cache_key"] == main_session
        assert warm_body["instructions"].startswith(TITLE_PREFIX)
