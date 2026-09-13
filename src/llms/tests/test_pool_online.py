from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from llms.proxy.buckets import BucketTable
from llms.proxy.egress import DirectEgress, ProviderEgress, WarpSocksEgress
from llms.proxy.main import create_app
from llms.proxy.providers import Provider, ProviderRegistry
from tests.conftest import TEST_HEADERS, TEST_SECRET, make_settings


def _chat_body(content: str) -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "mimo-v2.5-free",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _mock_client(content: str, seen: dict, key: str) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen[key] = seen.get(key, 0) + 1
        return httpx.Response(200, json=_chat_body(content))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture()
def pool_world(tmp_path, monkeypatch):
    from llms.proxy.keys import reset_cache
    from llms.proxy.store import ApiKey, Store

    reset_cache()
    Store(data_dir=tmp_path).save_keys([ApiKey(key=TEST_SECRET, label="t")])
    registry = ProviderRegistry(data_dir=tmp_path)
    seen: dict = {}
    zen_client = _mock_client("direct", seen, "direct_hits")
    warp_client = _mock_client("via-warp", seen, "warp_hits")
    monkeypatch.setattr(
        WarpSocksEgress, "client_for", lambda self, bucket, slot: warp_client
    )
    settings = make_settings(data_dir=str(tmp_path))
    app = create_app(settings)
    app.state.providers = registry
    egress = ProviderEgress(DirectEgress(zen_client), registry=registry)
    app.state.egress = egress
    app.state.bucket_table = BucketTable(num_buckets=32, num_slots=1)
    # lifespan would overwrite our ProviderEgress with a fresh DirectEgress
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as tc:
        yield tc, registry, egress, seen, tmp_path


def _save_pool(registry: ProviderRegistry) -> Provider:
    registry.save([Provider(id="pool1", kind="warp", slots=2, models=["mimo-*"])])
    return registry.load()[1]


def _fake_health(monkeypatch, registry, exits: list[dict]):
    import time

    from llms.proxy.providers import ProviderHealth, WarpExit

    async def _health(provider, force=False):
        rt = registry.runtime(provider.id)
        rt.health = ProviderHealth(
            active=1,
            exits=[WarpExit(**w) for w in exits],
            fetched_at=time.monotonic(),
        )
        return rt.health

    monkeypatch.setattr(registry, "refresh_health", _health)


def _chat(model: str = "mimo-v2.5-free") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def _ready2() -> list[dict]:
    return [
        {"idx": 1, "ready": True, "socks": 40001, "registered": True},
        {"idx": 2, "ready": True, "socks": 40002, "registered": True},
        {"idx": 3, "ready": False, "socks": 40003, "error": "booting"},
    ]


def test_pool_offline_falls_back_to_direct(pool_world, monkeypatch):
    (
        tc,
        registry,
        _egress,
        seen,
        _,
    ) = pool_world
    provider = _save_pool(registry)
    _fake_health(monkeypatch, registry, [])
    import anyio

    anyio.run(registry.refresh_health, provider, True)
    r = tc.post("/v1/chat/completions", json=_chat(), headers=TEST_HEADERS)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "direct"
    assert seen.get("direct_hits") == 1
    assert seen.get("warp_hits", 0) == 0
    assert r.headers.get("x-egress-provider") is None


def test_pool_online_takes_traffic_and_resizes(pool_world, monkeypatch):
    (
        tc,
        registry,
        _egress,
        seen,
        _,
    ) = pool_world
    provider = _save_pool(registry)
    _fake_health(monkeypatch, registry, _ready2())
    import anyio

    anyio.run(registry.refresh_health, provider, True)
    assert tc.app.state.bucket_table.num_slots == 1
    r = tc.post("/v1/chat/completions", json=_chat(), headers=TEST_HEADERS)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "via-warp"
    assert seen.get("warp_hits") == 1
    assert r.headers.get("x-egress-provider") == "pool1"
    assert r.headers.get("x-pool-active-warp") == "1"
    assert tc.app.state.bucket_table.num_slots == 2
    assert seen.get("direct_hits", 0) == 0


def test_pool_outage_returns_to_direct(pool_world, monkeypatch):
    (
        tc,
        registry,
        _egress,
        _seen,
        _,
    ) = pool_world
    provider = _save_pool(registry)
    _fake_health(monkeypatch, registry, _ready2()[:1])
    import anyio

    anyio.run(registry.refresh_health, provider, True)
    assert tc.app.state.bucket_table.num_slots == 1
    _fake_health(monkeypatch, registry, [])
    anyio.run(registry.refresh_health, provider, True)
    r = tc.post("/v1/chat/completions", json=_chat(), headers=TEST_HEADERS)
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "direct"


def test_add_then_remove_provider_lifecycle(pool_world):
    from llms.proxy.providers import ProviderHealth, WarpExit

    (
        tc,
        registry,
        egress,
        _seen,
        tmp_path,
    ) = pool_world
    assert registry.ready_exits("pool9") is None
    registry.save([Provider(id="pool9", kind="warp", slots=2, models=["mimo-*"])])
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
