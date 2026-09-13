from __future__ import annotations


class FakePool:
    """Stand-in for the in-process WarpPool behind ProviderRegistry."""

    def __init__(self, ready: tuple[int, ...] = (1, 2)) -> None:
        self.ready = ready
        self.reconnects = 0

    async def reconnect(self) -> dict:
        self.reconnects += 1
        return {
            "ok": True,
            "before": {"ready": 0, "exits": 2},
            "after": {"ready": 2, "exits": 2},
        }

    async def refresh_statuses(self) -> None:
        return None

    def snapshot(self) -> dict:
        return {
            "error": "",
            "exits": [
                {
                    "idx": i,
                    "ready": True,
                    "status": "Connected",
                    "reason": "",
                    "socks": 40000 + i,
                    "registered": True,
                    "error": "",
                }
                for i in self.ready
            ],
        }


def test_reconnect_bounces_pool_and_resyncs(admin_client, tmp_path, monkeypatch):
    import json

    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderRegistry

    pool = FakePool()

    async def _ensure_pool(provider):
        return pool

    tc, _ = admin_client
    table = tc.app.state.bucket_table
    registry = ProviderRegistry(data_dir=tmp_path)
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    registry.save([Provider(id="pool1", kind="warp", exits=2, models=["muse-*"])])
    egress = ProviderEgress(tc.app.state.egress, registry=registry)
    assert egress.resolve("muse-spark")[0] == "pool1"
    tc.app.state.egress = egress
    tc.app.state.providers = registry
    assert table.num_slots == 1
    r = tc.post(
        "/api/admin/providers/pool1/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["before"]["ready"] == 0
    assert body["after"]["ready"] == 2
    assert pool.reconnects == 1
    assert table.num_slots == 2
    assert json.loads((tmp_path / "warps" / "pool1" / "status.json").read_text())[
        "exits"
    ]


def test_reconnect_rejects_noproxy(admin_client):
    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers/noproxy/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 400


def test_reconnect_unknown_provider_404(admin_client):
    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers/nope/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 404


def test_local_reconnect_pool_uses_secret_key(app_client, tmp_path, monkeypatch):
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderRegistry

    pool = FakePool()

    async def _ensure_pool(provider):
        return pool

    tc, _ = app_client
    registry = ProviderRegistry(data_dir=tmp_path)
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    registry.save([Provider(id="pool1", kind="warp", exits=2, models=["muse-*"])])
    tc.app.state.egress = ProviderEgress(tc.app.state.egress, registry=registry)
    tc.app.state.providers = registry
    r = tc.post(
        "/api/providers/pool1/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "pool1"
    assert body["ok"] is True
    assert pool.reconnects == 1
    assert body["exits"][0]["status"] == "Connected"
    assert "reason" in body["exits"][0]


def test_local_reconnect_pool_requires_key(app_client):
    tc, _ = app_client
    assert tc.post("/api/providers/pool1/reconnect").status_code == 401


def test_local_reconnect_pool_rejects_noproxy(app_client):
    tc, _ = app_client
    r = tc.post(
        "/api/providers/noproxy/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 400


def test_local_reconnect_pool_unknown_404(app_client):
    tc, _ = app_client
    r = tc.post(
        "/api/providers/nope/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 404
