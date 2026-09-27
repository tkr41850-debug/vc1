from __future__ import annotations

import asyncio
import time

import httpx


def _slow_client(delay_s: float, calls: list):
    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        await asyncio.sleep(delay_s)
        # Synthesize path folds upstream SSE: emit real frames.
        body = (
            'data: {"type":"response.output_text.delta","delta":"slow hi"}\n\n'
            'data: {"type":"response.completed","response":{"status":"completed",'
            '"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )


def _app(tmp_path, client, timeout_s=0.2):
    from tests.conftest import TEST_SECRET, build_app_client, make_settings

    return build_app_client(
        make_settings(data_dir=str(tmp_path), max_timeout_s=timeout_s),
        client,
        seed_key=TEST_SECRET,
    )


def test_slow_request_429s_then_claims_held_response(tmp_path):
    from tests.conftest import TEST_HEADERS

    calls: list = []
    with _app(tmp_path, _slow_client(1.0, calls)) as tc:
        body = {"model": "muse-spark-1.3-contributor-free", "input": "hi"}
        first = tc.post("/v1/responses", json=body, headers=TEST_HEADERS)
        assert first.status_code == 429
        assert first.headers["retry-after"] == "20"
        assert first.headers["x-llms-dedup"] == "inflight"
        time.sleep(1.5)
        second = tc.post("/v1/responses", json=body, headers=TEST_HEADERS)
        assert second.status_code == 200
        assert second.headers["x-llms-dedup"] == "hit"
        assert second.json()["output"][0]["content"][0]["text"] == "slow hi"
        # Claimed responses evict: a third identical request does NOT get
        # a stale hit — it runs fresh (and 429s again while still slow).
        third = tc.post("/v1/responses", json=body, headers=TEST_HEADERS)
        assert third.status_code == 429
        assert third.headers["x-llms-dedup"] == "inflight"
        time.sleep(1.5)
        fourth = tc.post("/v1/responses", json=body, headers=TEST_HEADERS)
        assert fourth.status_code == 200
        assert fourth.headers["x-llms-dedup"] == "hit"
        assert fourth.json()["output"][0]["content"][0]["text"] == "slow hi"
    assert len(calls) == 2


def test_streaming_bypasses_dedup(tmp_path):
    from tests.conftest import TEST_HEADERS

    async def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
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
    with _app(tmp_path, client, timeout_s=0.01) as tc:
        r = tc.post(
            "/v1/responses",
            json={
                "model": "muse-spark-1.3-contributor-free",
                "input": "hi",
                "stream": True,
            },
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200
        assert "x-llms-dedup" not in r.headers
        assert len(tc.app.state.dedup._entries) == 0


def test_distinct_bodies_do_not_coalesce(tmp_path):
    from tests.conftest import TEST_HEADERS

    calls: list = []
    with _app(tmp_path, _slow_client(0.6, calls)) as tc:
        for text in ("alpha", "beta"):
            r = tc.post(
                "/v1/responses",
                json={"model": "muse-spark-1.3-contributor-free", "input": text},
                headers=TEST_HEADERS,
            )
            assert r.status_code == 429
    assert len(calls) == 2


def test_fresh_retries_reuse_reserved_session(tmp_path):
    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    seen: list = []

    async def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.append(_json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "id": "resp_x",
                "status": "completed",
                "model": "m",
                "output": [],
                "usage": {},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        make_settings(data_dir=str(tmp_path), max_timeout_s=60.0),
        client,
        seed_key=TEST_SECRET,
    ) as tc:
        for _ in range(2):
            # Clear dedup so both attempts reach upstream: this isolates
            # the session reservation (same prompt_cache_key) from
            # result coalescing.
            tc.app.state.dedup._entries.clear()
            r = tc.post(
                "/v1/responses",
                json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
                headers=TEST_HEADERS,
            )
            assert r.status_code == 200
    assert len(seen) == 2
    assert seen[0]["prompt_cache_key"] == seen[1]["prompt_cache_key"]


def test_dedup_429_does_not_poison_pool(tmp_path):
    from tests.conftest import TEST_HEADERS

    calls: list = []
    with _app(tmp_path, _slow_client(1.0, calls)) as tc:
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 429
        assert r.headers["retry-after"] == "20"
        registry = tc.app.state.providers
        for p in registry.load():
            assert registry.runtime(p.id).retry_until == 0.0


def test_error_responses_are_never_held(tmp_path):
    from tests.conftest import (
        TEST_HEADERS,
        TEST_SECRET,
        build_app_client,
        make_settings,
    )

    calls: list = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={"error": {"message": "slow down"}},
                headers={"retry-after": "1"},
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_ok",
                "status": "completed",
                "model": "m",
                "output": [],
                "usage": {},
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://opencode.ai/zen/v1"
    )
    with build_app_client(
        make_settings(data_dir=str(tmp_path), max_timeout_s=60.0),
        client,
        seed_key=TEST_SECRET,
    ) as tc:
        body = {"model": "muse-spark-1.3-contributor-free", "input": "hi"}
        first = tc.post("/v1/responses", json=body, headers=TEST_HEADERS)
        assert first.status_code == 429
        assert "x-llms-dedup" not in first.headers
        second = tc.post("/v1/responses", json=body, headers=TEST_HEADERS)
        assert second.status_code == 200
        assert "x-llms-dedup" not in second.headers
    assert len(calls) == 2
