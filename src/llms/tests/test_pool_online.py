from __future__ import annotations

import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from llms.proxy.buckets import BucketTable
from llms.proxy.egress import DirectEgress, ProviderEgress
from llms.proxy.main import create_app
from llms.proxy.providers import Provider, ProviderRegistry
from tests.conftest import TEST_HEADERS, TEST_SECRET, make_settings


def _zen_chat_body() -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-pool",
            "object": "chat.completion",
            "model": "mimo-v2.5-free",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "via-pool"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    ).encode()


class FakePool:
    def __init__(self) -> None:
        self.exits: list[dict] = []
        self.fetches = 0

    def health_payload(self) -> dict:
        return {"active": 1, "warps": self.exits}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json=self.health_payload())
        if request.url.path == "/fetch":
            self.fetches += 1
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "status": 200,
                    "headers": {"content-type": "application/json"},
                    "body_b64": base64.b64encode(_zen_chat_body()).decode(),
                },
            )
        return httpx.Response(404, json={"error": "nope"})


def _zen_mock_client(seen: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["direct_hits"] = seen.get("direct_hits", 0) + 1
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-direct",
                "object": "chat.completion",
                "model": "mimo-v2.5-free",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "direct"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture()
def pool_world(tmp_path):
    from llms.proxy.keys import reset_cache
    from llms.proxy.store import ApiKey, Store

    reset_cache()
    Store(data_dir=tmp_path).save_keys([ApiKey(key=TEST_SECRET, label="t")])
    pool = FakePool()
    pool_client = httpx.AsyncClient(transport=httpx.MockTransport(pool.handler))
    registry = ProviderRegistry(data_dir=tmp_path)
    registry._client = pool_client
    seen: dict = {}
    zen_client = _zen_mock_client(seen)
    settings = make_settings(data_dir=str(tmp_path))
    app = create_app(settings)
    egress = ProviderEgress(DirectEgress(zen_client), registry=registry)
    app.state.egress = egress
    app.state.bucket_table = BucketTable(num_buckets=32, num_slots=1)
    with TestClient(app) as tc:
        yield tc, registry, egress, pool, seen, tmp_path, pool_client


def _chat(model: str = "mimo-v2.5-free") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def test_pool_offline_falls_back_to_direct(pool_world):
    import anyio

    tc, registry, _egress, pool, seen, _, _pool_client = pool_world
    registry.save(
        [
            Provider(
                id="pool1", kind="warp", base_url="http://pool:8080", models=["mimo-*"]
            )
        ]
    )
    provider = registry.load()[1]
    anyio.run(registry.refresh_health, provider, True)
    r = tc.post("/v1/chat/completions", json=_chat(), headers=TEST_HEADERS)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "direct"
    assert seen.get("direct_hits") == 1
    assert pool.fetches == 0


def test_pool_online_takes_traffic_and_resizes(pool_world):
    from llms.proxy.egress import WarpPoolEgress

    tc, registry, egress, pool, seen, _, pool_client = pool_world
    registry.save(
        [
            Provider(
                id="pool1", kind="warp", base_url="http://pool:8080", models=["mimo-*"]
            )
        ]
    )
    egress._warp["pool1"] = WarpPoolEgress("http://pool:8080", client=pool_client)
    pool.exits = [
        {"idx": 1, "ready": True, "socks": 40001, "registered": True, "error": ""},
        {"idx": 2, "ready": True, "socks": 40002, "registered": True, "error": ""},
        {
            "idx": 3,
            "ready": False,
            "socks": 40003,
            "registered": False,
            "error": "booting",
        },
    ]
    import anyio

    provider = registry.load()[1]
    anyio.run(registry.refresh_health, provider, True)
    assert tc.app.state.bucket_table.num_slots == 1
    r = tc.post("/v1/chat/completions", json=_chat(), headers=TEST_HEADERS)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "via-pool"
    assert pool.fetches == 1
    assert r.headers.get("x-egress-provider") == "pool1"
    assert tc.app.state.bucket_table.num_slots == 2
    assert seen.get("direct_hits", 0) == 0


def test_pool_outage_returns_to_direct(pool_world):
    tc, registry, _egress, pool, _seen, _, _pool_client = pool_world
    registry.save(
        [
            Provider(
                id="pool1", kind="warp", base_url="http://pool:8080", models=["mimo-*"]
            )
        ]
    )
    pool.exits = [
        {"idx": 1, "ready": True, "socks": 40001, "registered": True, "error": ""}
    ]
    import anyio

    provider = registry.load()[1]
    anyio.run(registry.refresh_health, provider, True)
    assert tc.app.state.bucket_table.num_slots == 1
    pool.exits = []
    anyio.run(registry.refresh_health, provider, True)
    r = tc.post("/v1/chat/completions", json=_chat(), headers=TEST_HEADERS)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "direct"


def test_add_then_remove_provider_lifecycle(pool_world):
    from llms.proxy.providers import ProviderHealth, WarpExit

    tc, registry, egress, _pool, _seen, tmp_path, _pool_client = pool_world
    assert registry.ready_exits("pool9") is None
    registry.save(
        [
            Provider(
                id="pool9", kind="warp", base_url="http://pool:8080", models=["mimo-*"]
            )
        ]
    )
    registry.ensure_warp_dir("pool9")
    assert (tmp_path / "warps" / "pool9").is_dir()
    assert egress.sync_bucket_slots(tc.app.state.bucket_table) is False
    registry.save_warp_status(
        "pool9",
        ProviderHealth(
            active=1,
            exits=[WarpExit(idx=1, ready=True), WarpExit(idx=2, ready=True)],
            fetched_at=1.0,
        ),
    )
    assert egress.sync_bucket_slots(tc.app.state.bucket_table) is True
    assert tc.app.state.bucket_table.num_slots == 2
    providers = [p for p in registry.load() if p.id != "pool9"]
    registry.save(providers)
    registry.drop_warp_dir("pool9")
    assert egress.sync_bucket_slots(tc.app.state.bucket_table) is True
    assert tc.app.state.bucket_table.num_slots == 1
    assert not (tmp_path / "warps" / "pool9").exists()
