from __future__ import annotations

from llms.proxy.providers import Provider, ProviderRegistry


def test_seed_has_noproxy_serving_free_models(tmp_path):
    from llms.proxy.router import FREE_MODELS

    registry = ProviderRegistry(data_dir=tmp_path)
    providers = registry.load()
    assert [p.id for p in providers] == ["noproxy"]
    assert providers[0].kind == "noproxy"
    assert providers[0].enabled is True
    for m in FREE_MODELS:
        assert providers[0].serves(m) is True
    assert providers[0].serves("gpt-unknown-thing") is False


def test_provider_serves_patterns(tmp_path):
    p = Provider(id="w", kind="warp", models=["gpt-*", "claude-exact"])
    assert p.serves("GPT-4o") is True
    assert p.serves("claude-exact") is True
    assert p.serves("claude-other") is False
    q = Provider(id="a", kind="warp", models=["*"])
    assert q.serves("anything-at-all") is True


def test_registry_roundtrip_and_route(tmp_path):
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(
        registry.load()
        + [
            Provider(
                id="warp-1",
                label="Local warps",
                kind="warp",
                models=["gpt-*"],
                enabled=True,
                exits=4,
            )
        ]
    )
    routed = registry.route("gpt-5")
    assert routed is not None and routed.id == "warp-1"
    assert routed.exits == 4
    routed = registry.route("muse-spark-1.3-contributor-free")
    assert routed is not None and routed.id == "noproxy"


def test_route_skips_disabled_warp(tmp_path):
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(
        registry.load()
        + [Provider(id="warp-1", kind="warp", models=["*"], enabled=False)]
    )
    routed = registry.route("gpt-5")
    assert routed is None


def test_base_url_rejected_with_migration_error(tmp_path):
    import yaml

    from llms.proxy.store import StoreError

    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(registry.load())
    path = registry.path()
    raw = yaml.safe_load(path.read_text()) or []
    raw.append(
        {
            "id": "warp-1",
            "kind": "warp",
            "base_url": "http://pool:8080",
            "models": ["*"],
        }
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    try:
        registry.load()
    except StoreError as exc:
        assert "no longer supported" in str(exc)
    else:
        raise AssertionError("expected StoreError for base_url")


def test_retry_in_decays_and_records():
    registry = ProviderRegistry(data_dir="/tmp/does-not-matter")
    rt = registry.runtime("warp-1")
    assert rt.retry_in() == 0.0
    rt.note_ratelimited(None, "slow down")
    assert 55.0 < rt.retry_in() <= 60.0
    assert rt.retry_reason == "slow down"
    rt.retry_until = 0.0
    assert rt.retry_in() == 0.0


def test_recent_ring_caps_at_ten():
    import asyncio

    from llms.proxy.providers import RecentRequest

    registry = ProviderRegistry(data_dir="/tmp/does-not-matter")
    rt = registry.runtime("warp-1")

    async def fill():
        for i in range(15):
            await rt.record(RecentRequest(ts=float(i), model="m", status=200, ms=1.0))

    asyncio.run(fill())
    snap = rt.recent_snapshot()
    assert len(snap) == 10
    assert snap[0]["ts"] == 5.0
    assert snap[-1]["ts"] == 14.0


def test_provider_admin_crud(admin_client, tmp_path):
    from llms.proxy.store import Store

    tc, _ = admin_client
    r = tc.get("/api/admin/providers")
    assert r.status_code == 200
    assert [p["id"] for p in r.json()["providers"]] == ["noproxy"]

    r = tc.post(
        "/api/admin/providers",
        json={
            "id": "warp-1",
            "label": "Pool",
            "kind": "warp",
            "exits": 4,
            "models": ["gpt-*"],
            "enabled": True,
        },
    )
    assert r.status_code == 201
    assert (
        tc.post(
            "/api/admin/providers",
            json={"id": "warp-1", "kind": "warp", "exits": 2},
        ).status_code
        == 409
    )
    assert (
        tc.post(
            "/api/admin/providers", json={"id": "w2", "kind": "warp", "exits": 0}
        ).status_code
        == 400
    )
    assert (
        tc.post("/api/admin/providers", json={"id": "w2", "kind": "bogus"}).status_code
        == 400
    )

    r = tc.put("/api/admin/providers/warp-1", json={"enabled": False})
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert tc.delete("/api/admin/providers/noproxy").status_code == 403
    assert tc.delete("/api/admin/providers/warp-1").status_code == 200
    assert tc.delete("/api/admin/providers/warp-1").status_code == 404
    assert Store(data_dir=tmp_path).load_keys() is not None


def test_provider_admin_updates_models(admin_client):
    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers",
        json={
            "id": "warp-models",
            "label": "M",
            "kind": "warp",
            "exits": 1,
            "models": ["gpt-*"],
            "enabled": True,
        },
    )
    assert r.status_code == 201
    r = tc.put(
        "/api/admin/providers/warp-models",
        json={"models": ["claude-*", "muse-spark-*"]},
    )
    assert r.status_code == 200
    providers = tc.get("/api/admin/providers").json()["providers"]
    entry = next(p for p in providers if p["id"] == "warp-models")
    assert entry["models"] == ["claude-*", "muse-spark-*"]
    assert tc.put("/api/admin/providers/nope", json={"models": []}).status_code == 404


def test_warp_egress_socks_and_retry_tracking(monkeypatch):
    import asyncio

    import httpx as _httpx

    from llms.proxy.egress import ProviderEgress, WarpSocksEgress
    from llms.proxy.main import create_app
    from llms.proxy.store import ApiKey, Store
    from tests.conftest import TEST_HEADERS, TEST_SECRET, make_settings

    def _zen_handler(request: _httpx.Request) -> _httpx.Response:
        assert str(request.url).endswith("/responses")
        return _httpx.Response(
            200,
            json={
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "model": "muse-spark-1.3-contributor-free",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    zen_client = _httpx.AsyncClient(transport=_httpx.MockTransport(_zen_handler))

    import pathlib
    import tempfile

    data_dir = pathlib.Path(tempfile.mkdtemp())
    Store(data_dir=data_dir).save_keys([ApiKey(key=TEST_SECRET)])
    registry = ProviderRegistry(data_dir=data_dir)
    registry.save(
        registry.load()
        + [
            Provider(
                id="warp-1",
                kind="warp",
                exits=2,
                models=["muse-spark*"],
            )
        ]
    )

    from fastapi.testclient import TestClient

    from llms.proxy.buckets import BucketTable
    from llms.proxy.egress import DirectEgress

    upstream = _httpx.AsyncClient(
        transport=_httpx.MockTransport(
            lambda req: _httpx.Response(200, json={"unexpected": True})
        )
    )
    app = create_app(make_settings(data_dir=str(data_dir)))
    app.state.providers = registry
    app.state.egress = ProviderEgress(DirectEgress(upstream), registry=registry)
    app.state.bucket_table = BucketTable(num_buckets=1024, num_slots=1)
    # lifespan would overwrite our ProviderEgress with a fresh DirectEgress
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    app.router.lifespan_context = _noop_lifespan

    async def _fake_health(provider, force=False):
        from llms.proxy.providers import ProviderHealth, WarpExit

        rt = registry.runtime("warp-1")
        rt.health = ProviderHealth(
            exits=[
                WarpExit(idx=1, ready=True, socks=40001, registered=True),
                WarpExit(idx=2, ready=True, socks=40002, registered=True),
            ],
            fetched_at=1.0,
        )
        return rt.health

    monkeypatch.setattr(registry, "refresh_health", _fake_health)
    monkeypatch.setattr(
        WarpSocksEgress, "client_for", lambda self, bucket, slot: zen_client
    )
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "completed"
        assert r.headers.get("x-egress-provider") == "warp-1"
        # admin routes need the require_admin override; direct registry check:
        rt = registry.runtime("warp-1")
        snap = rt.recent_snapshot()
        assert len(snap) == 1
        assert snap[0]["model"] == "muse-spark-1.3-contributor-free"
        assert snap[0]["status"] == 200
    asyncio.run(registry.aclose())
    asyncio.run(zen_client.aclose())
    asyncio.run(upstream.aclose())
