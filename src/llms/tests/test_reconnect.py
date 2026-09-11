from __future__ import annotations

import httpx


def _pool_handler(state: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rotate":
            state["rotates"] = state.get("rotates", 0) + 1
            return httpx.Response(200, json={"ok": True, "old": 1, "active": 2})
        if request.url.path == "/health":
            exits = state.get("exits", [])
            return httpx.Response(200, json={"active": 1, "warps": exits})
        return httpx.Response(404, json={"error": "nope"})

    return handler


def test_reconnect_bounces_pool_and_resyncs(admin_client, tmp_path):
    import json

    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderRegistry

    state: dict = {
        "exits": [
            {"idx": 1, "ready": True, "socks": 40001, "registered": True, "error": ""},
            {"idx": 2, "ready": True, "socks": 40002, "registered": True, "error": ""},
        ]
    }
    pool_client = httpx.AsyncClient(transport=httpx.MockTransport(_pool_handler(state)))
    tc, _ = admin_client
    table = tc.app.state.bucket_table
    registry = ProviderRegistry(data_dir=tmp_path)
    registry._client = pool_client
    registry.save(
        [
            Provider(
                id="pool1", kind="warp", base_url="http://pool:8080", models=["muse-*"]
            )
        ]
    )
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
    assert state["rotates"] == 1
    assert "pool1" not in egress._warp
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
