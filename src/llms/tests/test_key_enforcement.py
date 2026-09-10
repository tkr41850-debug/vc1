from __future__ import annotations

from llms.proxy.store import Store


def test_healthz_open_without_key(app_client):
    tc, _ = app_client
    assert tc.get("/healthz").status_code == 200


def test_keyless_ingress_rejected(app_client):
    tc, _ = app_client
    for path in (
        "/v1/responses",
        "/responses",
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/models",
        "/models",
    ):
        kwargs = {"json": {"input": "hi"}} if "models" not in path else {}
        r = tc.get(path, **kwargs) if "models" in path else tc.post(path, **kwargs)
        assert r.status_code == 401, path
        assert r.json()["error"]["message"] == "unknown or disabled API key"


def test_unknown_key_rejected(app_client):
    tc, _ = app_client
    r = tc.post("/ak-nope/v1/responses", json={"input": "hi"})
    assert r.status_code == 401


def test_disabled_key_rejected(app_client, tmp_path):
    tc, _ = app_client
    store = Store(data_dir=tmp_path)
    keys = store.load_keys()
    for k in keys:
        if k.key == "ak-test":
            k.enabled = False
    store.save_keys(keys)
    assert tc.post("/ak-test/v1/responses", json={"input": "hi"}).status_code == 401


def test_known_key_forwards(app_client, mock_upstream):
    from tests.conftest import TEST_KEY

    tc, _seen = app_client
    _, seen_dict = mock_upstream
    r = tc.post(
        f"/{TEST_KEY}/v1/responses",
        json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
    )
    assert r.status_code == 200
    assert seen_dict["url"].endswith("/responses")


def test_models_requires_key_but_lists_for_known_key(app_client):
    from tests.conftest import TEST_KEY

    tc, _ = app_client
    assert tc.get("/v1/models").status_code == 401
    r = tc.get(f"/{TEST_KEY}/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_admin_and_ui_require_login(app_client):
    tc, _ = app_client
    r = tc.get("/api/admin/keys")
    assert r.status_code == 401
    assert r.json() == {"error": {"message": "admin login required"}}
    r = tc.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers["location"] == "/api/admin/login"
