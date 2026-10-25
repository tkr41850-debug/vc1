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


def test_mint_head_is_current_time():
    import time as _time

    def _head_mod(sid: str) -> int:
        value = int(sid.removeprefix("ses_")[:12], 16)
        return (((~value) & ((1 << 48) - 1)) - 1) // 0x1000

    before = int(_time.time() * 1000)
    sid = mint_session_id()
    after = int(_time.time() * 1000)
    mod = 2**36
    for stamp in (before, after):
        delta = (_head_mod(sid) - stamp) % mod
        assert min(delta, mod - delta) <= 5000


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


def test_conversation_ref_codex_client_metadata_thread():
    # Live codex-exec shape: thread identity rides in the BODY
    # (client_metadata.thread_id == session_id per turn) because
    # body-only relays strip the thread-id/session-id headers. Without
    # this, every codex turn mints a fresh session and the upstream
    # prompt cache never warms across turns.
    body = {
        "client_metadata": {"thread_id": "01a10b8b-xyz", "session_id": "01a10b8b-xyz"}
    }
    assert conversation_ref("responses", body, {}) == "codex-thread:01a10b8b-xyz"
    # previous_response_id still wins when present.
    chained = dict(body, previous_response_id="resp_9")
    assert conversation_ref("responses", chained, {}) == "chain:resp_9"


def test_tracker_lookup_remember_expiry():
    tracker = SessionTracker(ttl_s=0.05)
    assert tracker.lookup("k", "chain:r1") is None
    tracker.remember("k", "chain:r1", "ses_abc")
    assert tracker.lookup("k", "chain:r1") == "ses_abc"
    assert tracker.lookup("other", "chain:r1") is None
    time.sleep(0.08)
    assert tracker.lookup("k", "chain:r1") is None


def test_plausible_session_id_accepts_fresh_rejects_garbage():
    import time as _time

    from llms.proxy.zen_headers import descending_session_id, plausible_session_id

    assert plausible_session_id(descending_session_id())
    assert not plausible_session_id("ses_" + "ab" * 6 + "0" * 14)
    assert not plausible_session_id("not-a-session")
    assert not plausible_session_id("")
    assert not plausible_session_id(None)
    old = descending_session_id(now_ms=int(_time.time() * 1000) - 90 * 86400 * 1000)
    assert not plausible_session_id(old)
    future = descending_session_id(now_ms=int(_time.time() * 1000) + 30 * 60 * 1000)
    assert plausible_session_id(future)


def test_genuine_incoming_session_rides_unchanged(mock_upstream, tmp_path):
    """Relay/complete-passthrough: a genuine caller carrying a plausible
    session keeps it end to end (headers + prompt_cache_key) instead of
    a re-mint; implausible ids fall back to minted."""
    from llms.proxy.zen_headers import descending_session_id
    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    client, seen = mock_upstream
    genuine = descending_session_id()
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        r = tc.post(
            "/v1/responses",
            json={"model": "glm-5.3-flash-free", "input": "hi"},
            headers={
                "Authorization": f"Bearer {TEST_SECRET}",
                "User-Agent": "opencode/latest/2.0.29/cli",
                "x-opencode-session": genuine,
            },
        )
        assert r.status_code == 200
        assert seen["headers"]["x-opencode-session"] == genuine
        assert seen["headers"]["x-opencode-session-id"] == genuine
        assert seen["json"]["prompt_cache_key"] == genuine
        r2 = tc.post(
            "/v1/responses",
            json={"model": "glm-5.3-flash-free", "input": "hi"},
            headers={
                "Authorization": f"Bearer {TEST_SECRET}",
                "User-Agent": "opencode/latest/2.0.29/cli",
                "x-opencode-session": "ses_deadbeefdead00",
            },
        )
        assert r2.status_code == 200
        assert seen["headers"]["x-opencode-session"] != "ses_deadbeefdead00"


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
            json={"model": "glm-5.3-flash-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert first.status_code == 200
        first_session = seen["headers"]["x-opencode-session"]
        second = tc.post(
            "/v1/responses",
            json={
                "model": "glm-5.3-flash-free",
                "input": "hi again",
                "previous_response_id": first.json()["id"],
            },
            headers=TEST_HEADERS,
        )
        assert second.status_code == 200
        assert seen["headers"]["x-opencode-session"] == first_session


def test_streaming_chained_turns_reuse_session(tmp_path):
    """Streaming passthrough chains previous_response_id to the session.

    Streaming legs hand the client the upstream response id; without
    chaining it at stream end the follow-up fell back to the per-key
    session and the prompt cache went cold every turn.
    """
    import json as _json

    import httpx

    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    seen_keys: list = []

    async def handler(request):
        payload = _json.loads(request.content.decode())
        seen_keys.append(payload.get("prompt_cache_key"))
        body = (
            'data: {"type":"response.created",'
            '"response":{"id":"resp_up1","status":"in_progress"}}\n\n'
            'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
            'data: {"type":"response.completed","response":{"status":"completed",'
            '"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
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
        first = tc.post(
            "/v1/responses",
            json={
                "model": "glm-5.3-flash-free",
                "input": "hi",
                "stream": True,
            },
            headers=TEST_HEADERS,
        )
        assert first.status_code == 200
        assert "resp_up1" in first.text
        second = tc.post(
            "/v1/responses",
            json={
                "model": "glm-5.3-flash-free",
                "input": "hi again",
                "previous_response_id": "resp_up1",
                "stream": True,
            },
            headers=TEST_HEADERS,
        )
        assert second.status_code == 200
    assert len(seen_keys) == 2
    assert seen_keys[0] and seen_keys[0] == seen_keys[1]


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
            json={"model": "glm-5.3-flash-free", "input": "hi"},
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


def test_claude_session_header_maps_to_conversation_ref():
    """X-Claude-Code-Session-Id is a stable per-conversation signal.

    Wire-verified 2026-10-02 (/tmp/claude-flow.jsonl): main-agent and
    Task-delegated subagent turns carry distinct session ids on every
    messages-leg request, so the header separates conversations the
    same way codex thread-id does on the responses leg.
    """
    main = {"X-Claude-Code-Session-Id": "98ec1b78-15f9-49f1-bb54-e5fff7a46a37"}
    sub = {"X-Claude-Code-Session-Id": "b0c8fd8e-ead1-48c1-92fe-5a0c5edcafba"}

    def _headers(d: dict):
        from starlette.datastructures import Headers

        return Headers(d)

    body: dict = {"model": "m"}
    assert conversation_ref("responses", body, _headers(main)) == (
        "claude-session:98ec1b78-15f9-49f1-bb54-e5fff7a46a37"
    )
    assert conversation_ref("responses", body, _headers(sub)) == (
        "claude-session:b0c8fd8e-ead1-48c1-92fe-5a0c5edcafba"
    )
    assert conversation_ref("responses", body, _headers(main)) != conversation_ref(
        "responses", body, _headers(sub)
    )
