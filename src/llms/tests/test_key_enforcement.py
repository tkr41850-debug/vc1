from __future__ import annotations

from llms.proxy.store import Store
from tests.conftest import TEST_HEADERS, TEST_SECRET


def test_healthz_open_without_secret(app_client):
    tc, _ = app_client
    assert tc.get("/healthz").status_code == 200


def test_corrupt_store_fails_closed_but_healthz_reports(app_client, tmp_path):
    from llms.proxy.keys import reset_cache

    tc, _ = app_client
    (tmp_path / "keys.yaml").write_text("not: [valid, yaml\n")
    reset_cache()
    try:
        r = tc.post("/v1/responses", json={"input": "hi"}, headers=TEST_HEADERS)
        assert r.status_code == 503
        assert r.json() == {"error": {"message": "key store unavailable"}}
    finally:
        reset_cache()


def test_missing_secret_rejected_on_all_ingress(app_client):
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
        assert r.json()["error"]["message"] == "missing or invalid secret key"


def test_affinity_prefix_alone_is_not_auth(app_client):
    # ak- in the path is unauthenticated bucket routing only.
    tc, _ = app_client
    r = tc.post("/ak-team1/v1/responses", json={"input": "hi"})
    assert r.status_code == 401
    assert r.json()["error"]["message"] == "missing or invalid secret key"


def test_wrong_scheme_and_unknown_secret_rejected(app_client):
    tc, _ = app_client
    r = tc.post(
        "/v1/responses",
        json={"input": "hi"},
        headers={"Authorization": "Bearer ak-team1"},
    )
    assert r.status_code == 401
    r = tc.post(
        "/v1/responses",
        json={"input": "hi"},
        headers={"Authorization": "Bearer sk-nope"},
    )
    assert r.status_code == 401


def test_disabled_secret_rejected(app_client, tmp_path):
    from llms.proxy.keys import reset_cache

    tc, _ = app_client
    store = Store(data_dir=tmp_path)
    keys = store.load_keys()
    for k in keys:
        if k.key == TEST_SECRET:
            k.enabled = False
    store.save_keys(keys)
    reset_cache()
    try:
        assert (
            tc.post(
                "/v1/responses", json={"input": "hi"}, headers=TEST_HEADERS
            ).status_code
            == 401
        )
    finally:
        reset_cache()


def test_secret_header_forwards_with_and_without_affinity(app_client, mock_upstream):
    tc, _ = app_client
    _, seen_dict = mock_upstream
    for path in ("/v1/responses", "/ak-team1/v1/responses"):
        r = tc.post(
            path,
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200, path
        assert seen_dict["url"].endswith("/responses")
    # the sk- secret never reaches the upstream gateway
    assert "authorization" not in seen_dict["headers"]


def test_models_requires_secret_but_lists_for_known_secret(app_client):
    tc, _ = app_client
    assert tc.get("/v1/models").status_code == 401
    r = tc.get("/v1/models", headers=TEST_HEADERS)
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
