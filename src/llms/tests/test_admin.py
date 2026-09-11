from __future__ import annotations

from llms.proxy.store import Store
from tests.conftest import TEST_HEADERS, TEST_SECRET


def test_admin_keys_crud(admin_client, tmp_path):
    tc, _ = admin_client
    r = tc.get("/api/admin/keys")
    assert r.status_code == 200
    assert [k["key"] for k in r.json()["keys"]] == [TEST_SECRET]

    r = tc.post(
        "/api/admin/keys", json={"key": "sk-new", "label": "New", "enabled": True}
    )
    assert r.status_code == 201
    assert Store(data_dir=tmp_path).key_allowed("sk-new") is True

    # ak- values are affinity, not API keys: rejected at creation
    assert tc.post("/api/admin/keys", json={"key": "ak-nope"}).status_code == 400

    r = tc.put("/api/admin/keys/sk-new", json={"label": "Renamed", "enabled": False})
    assert r.status_code == 200
    assert r.json() == {"key": "sk-new", "label": "Renamed", "enabled": False}
    assert Store(data_dir=tmp_path).key_allowed("sk-new") is False

    r = tc.post("/api/admin/keys/sk-new/rotate", json={"new_key": "sk-rotated"})
    assert r.status_code == 201
    assert r.json()["key"] == "sk-rotated"
    store = Store(data_dir=tmp_path)
    assert store.key_allowed("sk-rotated") is True
    assert store.key_allowed("sk-new") is False
    assert tc.post("/api/admin/keys/sk-new/rotate", json={}).status_code == 201

    r = tc.delete("/api/admin/keys/sk-new")
    assert r.status_code == 200
    assert Store(data_dir=tmp_path).find_key("sk-new") is None

    assert tc.post("/api/admin/keys", json={"key": TEST_SECRET}).status_code == 409
    assert tc.put("/api/admin/keys/sk-missing", json={}).status_code == 404
    assert tc.delete("/api/admin/keys/sk-missing").status_code == 404


def test_admin_models_crud(admin_client, tmp_path):
    tc, _ = admin_client
    store = Store(data_dir=tmp_path)
    assert tc.get("/api/admin/models").status_code == 200

    r = tc.post(
        "/api/admin/models",
        json={"id": "probe-only-model", "label": "Probe", "enabled": True},
    )
    assert r.status_code == 201
    assert "probe-only-model" in store.enabled_model_ids()

    r = tc.put("/api/admin/models/probe-only-model", json={"enabled": False})
    assert r.status_code == 200
    assert "probe-only-model" not in store.enabled_model_ids()

    assert tc.delete("/api/admin/models/probe-only-model").status_code == 200
    assert tc.put("/api/admin/models/nope", json={}).status_code == 404

    # /v1/models reflects the store: authed access works and lists enabled models
    r = tc.get("/v1/models", headers=TEST_HEADERS)
    assert r.status_code == 200
    assert "probe-only-model" not in [m["id"] for m in r.json()["data"]]


def test_admin_usage_endpoint(admin_client):
    tc, _ = admin_client
    r = tc.post(
        "/v1/responses",
        json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
        headers=TEST_HEADERS,
    )
    assert r.status_code == 200
    r = tc.get("/api/admin/usage")
    assert r.status_code == 200
    assert r.json()["keys"][TEST_SECRET]["requests"] == 1
    r = tc.get("/api/admin/keys")
    assert r.json()["keys"][0]["usage"]["requests"] == 1


def test_oauth_login_redirect_and_unconfigured(app_client):
    tc, _ = app_client
    # no github client id in test settings -> 500, not a redirect
    r = tc.get("/api/admin/login", follow_redirects=False)
    assert r.status_code == 500
    assert tc.post("/api/admin/logout").status_code == 200
