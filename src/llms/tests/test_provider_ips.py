from __future__ import annotations


def test_provider_ips_warp_per_exit(admin_client, tmp_path, monkeypatch):
    from llms.proxy.providers import Provider, ProviderRegistry, WarpExit
    from llms.proxy.routes import providers as routes

    tc, _ = admin_client
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save([Provider(id="pool1", kind="warp", exits=2, models=["muse-*"])])
    rt = registry.runtime("pool1")
    rt.health.exits = [
        WarpExit(idx=0, ready=True, status="Connected", socks=40001),
        WarpExit(idx=1, ready=True, status="Connected", socks=40002),
        WarpExit(idx=2, ready=False, status="Down", socks=40003),
    ]
    tc.app.state.providers = registry

    seen: list = []

    async def _fake_fetch(proxy_url):
        seen.append(proxy_url)
        if proxy_url is not None and proxy_url.endswith(":40002"):
            raise RuntimeError("boom")
        return "9.9.9.9" if proxy_url else "1.2.3.4"

    monkeypatch.setattr(routes, "_fetch_ip", _fake_fetch)
    routes._ip_cache.clear()
    r = tc.get(
        "/api/admin/providers/pool1/ips", headers={"Authorization": "Bearer sk-test"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "pool1"
    assert body["cached"] is False
    assert body["ips"][0] == {
        "idx": 0,
        "port": 40001,
        "ip": "9.9.9.9",
        "error": None,
    }
    assert body["ips"][1]["idx"] == 1
    assert body["ips"][1]["ip"] is None
    assert "boom" in body["ips"][1]["error"]
    assert body["ips"][2]["error"] == "exit not ready"
    assert "socks5://127.0.0.1:40001" in seen
    # Second call serves the cache without refetching.
    r2 = tc.get(
        "/api/admin/providers/pool1/ips", headers={"Authorization": "Bearer sk-test"}
    )
    assert r2.json()["cached"] is True
    assert len(seen) == 2


def test_provider_ips_noproxy_direct(admin_client, tmp_path, monkeypatch):
    from llms.proxy.providers import Provider, ProviderRegistry
    from llms.proxy.routes import providers as routes

    tc, _ = admin_client
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save([Provider(id="direct", kind="noproxy", models=["*"])])
    tc.app.state.providers = registry

    async def _fake_fetch(proxy_url):
        assert proxy_url is None
        return "5.6.7.8"

    monkeypatch.setattr(routes, "_fetch_ip", _fake_fetch)
    routes._ip_cache.clear()
    r = tc.get(
        "/api/admin/providers/direct/ips", headers={"Authorization": "Bearer sk-test"}
    )
    assert r.status_code == 200
    assert r.json()["ips"] == [
        {"idx": None, "port": None, "ip": "5.6.7.8", "error": None}
    ]


def test_provider_ips_unknown_404(admin_client):
    tc, _ = admin_client
    r = tc.get(
        "/api/admin/providers/nope/ips", headers={"Authorization": "Bearer sk-test"}
    )
    assert r.status_code == 404


def test_provider_ips_local_requires_key(app_client, tmp_path, monkeypatch):
    from llms.proxy.providers import Provider, ProviderRegistry

    tc, _ = app_client
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save([Provider(id="pool1", kind="warp", exits=1, models=["*"])])
    tc.app.state.providers = registry
    r = tc.get("/api/providers/pool1/ips")
    assert r.status_code == 401


def test_reconnect_invalidates_ips(admin_client, tmp_path, monkeypatch):
    from llms.proxy.providers import Provider, ProviderRegistry, WarpExit
    from llms.proxy.routes import providers as routes

    tc, _ = admin_client

    class FakePool:
        async def reconnect(self):
            return {"ok": True, "before": {}, "after": {}}

        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {
                "error": "",
                "exits": [
                    {
                        "idx": 0,
                        "ready": True,
                        "status": "Connected",
                        "reason": "",
                        "socks": 40001,
                        "registered": True,
                        "error": "",
                    }
                ],
            }

    async def _ensure_pool(provider):
        return FakePool()

    registry = ProviderRegistry(data_dir=tmp_path)
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    registry.save([Provider(id="pool1", kind="warp", exits=1, models=["*"])])
    registry.runtime("pool1").health.exits = [
        WarpExit(idx=0, ready=True, status="Connected", socks=40001)
    ]
    tc.app.state.providers = registry

    calls: list = []

    async def _fake_fetch(proxy_url):
        calls.append(proxy_url)
        return "9.9.9.9"

    monkeypatch.setattr(routes, "_fetch_ip", _fake_fetch)
    routes._ip_cache.clear()
    headers = {"Authorization": "Bearer sk-test"}
    assert (
        tc.get("/api/admin/providers/pool1/ips", headers=headers).json()["cached"]
        is False
    )
    assert (
        tc.get("/api/admin/providers/pool1/ips", headers=headers).json()["cached"]
        is True
    )
    assert len(calls) == 1
    assert (
        tc.post("/api/admin/providers/pool1/reconnect", headers=headers).status_code
        == 200
    )
    assert (
        tc.get("/api/admin/providers/pool1/ips", headers=headers).json()["cached"]
        is False
    )
    assert len(calls) == 2


def test_exit_churn_busts_cache(admin_client, tmp_path, monkeypatch):
    from llms.proxy.providers import Provider, ProviderRegistry, WarpExit
    from llms.proxy.routes import providers as routes

    tc, _ = admin_client
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save([Provider(id="pool1", kind="warp", exits=2, models=["*"])])
    rt = registry.runtime("pool1")
    rt.health.exits = [WarpExit(idx=0, ready=True, status="Up", socks=40001)]
    tc.app.state.providers = registry

    calls: list = []

    async def _fake_fetch(proxy_url):
        calls.append(proxy_url)
        return "9.9.9.9"

    monkeypatch.setattr(routes, "_fetch_ip", _fake_fetch)
    routes._ip_cache.clear()
    headers = {"Authorization": "Bearer sk-test"}
    assert (
        tc.get("/api/admin/providers/pool1/ips", headers=headers).json()["cached"]
        is False
    )
    rt.health.exits.append(WarpExit(idx=1, ready=True, status="Up", socks=40002))
    body = tc.get("/api/admin/providers/pool1/ips", headers=headers).json()
    assert body["cached"] is False
    assert [e["port"] for e in body["ips"]] == [40001, 40002]
    assert len(calls) == 3
