from __future__ import annotations

from llms.proxy.router import FREE_MODELS


def test_models_lists_free_catalog(app_client):
    tc, _ = app_client
    r = tc.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == list(FREE_MODELS)
    assert all(m["object"] == "model" and m["owned_by"] == "llms" for m in body["data"])


def test_models_bare_alias(app_client):
    tc, _ = app_client
    assert tc.get("/models").status_code == 200


def test_models_behind_affinity_prefix(app_client):
    tc, _ = app_client
    r = tc.get("/ak-team1/v1/models")
    assert r.status_code == 200
    assert r.json()["object"] == "list"
