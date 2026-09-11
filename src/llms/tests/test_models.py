from __future__ import annotations

from llms.proxy.catalog import BY_ID
from llms.proxy.routes.models import entry_for
from tests.conftest import TEST_HEADERS


def test_models_lists_free_catalog(app_client):
    tc, _ = app_client
    r = tc.get("/v1/models", headers=TEST_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == list(BY_ID)
    assert all(m["object"] == "model" and m["owned_by"] == "llms" for m in body["data"])


def test_models_bare_alias(app_client):
    tc, _ = app_client
    assert tc.get("/models", headers=TEST_HEADERS).status_code == 200


def test_models_missing_secret_rejected(app_client):
    tc, _ = app_client
    r = tc.get("/v1/models")
    assert r.status_code == 401
    assert r.json() == {"error": {"message": "missing or invalid secret key"}}


def test_models_exposes_limits_effort_routing_and_tools(app_client):
    tc, _ = app_client
    data = {
        m["id"]: m for m in tc.get("/v1/models", headers=TEST_HEADERS).json()["data"]
    }
    spark = data["muse-spark-1.3-contributor-free"]
    assert spark["context_window"] == 1000000
    assert spark["max_output_tokens"] == 131072
    assert spark["reasoning_effort"] == ["low", "medium", "high"]
    assert spark["zen_endpoint"] == "responses"
    assert spark["tools"] is True
    assert spark["streaming"] is True
    assert spark["pricing"] == {"input": 0, "output": 0}
    assert spark["contributor_terms"] is True
    deepseek = data["deepseek-v4-flash-free"]
    assert deepseek["zen_endpoint"] == "chat"
    assert deepseek["thinking_toggle"] is True
    assert deepseek["tools"] is True


def test_unknown_model_gets_minimal_entry():
    entry = entry_for("future-model-99", 123)
    assert entry == {
        "id": "future-model-99",
        "object": "model",
        "created": 123,
        "owned_by": "llms",
    }
