from __future__ import annotations

import httpx

from llms.proxy.providers import (
    Provider,
    ProviderRegistry,
    fetch_spec,
    parse_fetch_result,
    warp_exit_for,
)


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
                label="Pool",
                kind="warp",
                base_url="http://pool:8080",
                token="t",
                models=["gpt-*"],
                enabled=True,
            )
        ]
    )
    routed = registry.route("gpt-5")
    assert routed is not None and routed.id == "warp-1"
    routed = registry.route("muse-spark-1.3-contributor-free")
    assert routed is not None and routed.id == "noproxy"


def test_route_skips_disabled_warp(tmp_path):
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(
        registry.load()
        + [
            Provider(
                id="warp-1",
                kind="warp",
                base_url="http://pool:8080",
                models=["*"],
                enabled=False,
            )
        ]
    )
    routed = registry.route("gpt-5")
    assert routed is None


def test_warp_exit_for_pins_ready_exits():
    from llms.proxy.providers import ProviderHealth, WarpExit

    health = ProviderHealth(
        active=1,
        exits=[
            WarpExit(idx=1, ready=True),
            WarpExit(idx=2, ready=False),
            WarpExit(idx=3, ready=True),
        ],
    )
    assert warp_exit_for(health, 0, 0) in (1, 3)
    assert warp_exit_for(health, 0, 0) == warp_exit_for(health, 0, 0)
    empty = ProviderHealth()
    assert warp_exit_for(empty, 0, 0) is None


def test_fetch_spec_and_parse_roundtrip():
    path, headers, body = fetch_spec(
        "https://opencode.ai/zen/v1/responses",
        {"Content-Type": "application/json", "Host": "x", "X-Keep": "y"},
        b'{"a":1}',
        token="tok",
    )
    assert path == "/fetch"
    assert headers["Authorization"] == "Bearer tok"
    assert "host" not in {k.lower() for k in headers}
    import base64
    import json as _json

    spec = _json.loads(body.decode())
    assert spec["url"] == "https://opencode.ai/zen/v1/responses"
    assert base64.b64decode(spec["body_b64"]) == b'{"a":1}'
    status, out_headers, out_body = parse_fetch_result(
        {"ok": True, "status": 200, "headers": {"a": "b"}, "body_b64": spec["body_b64"]}
    )
    assert (status, out_body) == (200, b'{"a":1}')
    assert out_headers == {"a": "b"}


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


def _pool_handler(request: httpx.Request) -> httpx.Response:
    import base64
    import json as _json

    if request.url.path == "/health":
        return httpx.Response(
            200,
            json={
                "active": 2,
                "warps": [
                    {
                        "idx": 2,
                        "ready": True,
                        "status": "Connected",
                        "reason": "",
                        "socks": 40002,
                        "registered": True,
                        "error": "",
                    }
                ],
            },
        )
    if request.url.path == "/fetch":
        spec = _json.loads(request.content.decode())
        assert spec["url"].endswith("/responses")
        inner = _json.dumps(
            {
                "id": "resp_1",
                "object": "response",
                "status": "completed",
                "model": "muse-spark-1.3-contributor-free",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ).encode()
        return httpx.Response(
            200,
            json={
                "ok": True,
                "status": 200,
                "headers": {},
                "body_b64": base64.b64encode(inner).decode(),
            },
        )
    return httpx.Response(404, json={"ok": False})


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
            "base_url": "http://pool:8080",
            "token": "t",
            "models": ["gpt-*"],
            "enabled": True,
        },
    )
    assert r.status_code == 201
    assert (
        tc.post(
            "/api/admin/providers",
            json={"id": "warp-1", "kind": "warp", "base_url": "http://x"},
        ).status_code
        == 409
    )
    assert (
        tc.post("/api/admin/providers", json={"id": "w2", "kind": "warp"}).status_code
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


def test_warp_egress_relay_and_retry_tracking(monkeypatch):
    import asyncio

    import httpx as _httpx

    from llms.proxy.egress import ProviderEgress, WarpPoolEgress
    from llms.proxy.main import create_app
    from llms.proxy.store import ApiKey, Store
    from tests.conftest import TEST_HEADERS, TEST_SECRET, make_settings

    transport = _httpx.MockTransport(_pool_handler)
    pool_client = _httpx.AsyncClient(transport=transport, base_url="http://pool:8080")

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
                base_url="http://pool:8080",
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
    monkeypatch.setattr(
        WarpPoolEgress, "client_for", lambda self, bucket, slot: pool_client
    )
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "completed"
        # admin routes need the require_admin override; direct registry check:
        rt = registry.runtime("warp-1")
        snap = rt.recent_snapshot()
        assert len(snap) == 1
        assert snap[0]["model"] == "muse-spark-1.3-contributor-free"
        assert snap[0]["status"] == 200
    asyncio.run(registry.aclose())
    asyncio.run(pool_client.aclose())
    asyncio.run(upstream.aclose())
