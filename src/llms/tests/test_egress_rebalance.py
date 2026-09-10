from __future__ import annotations

import pathlib

import httpx
import pytest
from fastapi.testclient import TestClient

from llms.proxy.affinity import bucket_for
from llms.proxy.buckets import BucketTable
from llms.proxy.main import create_app
from tests.conftest import make_settings

NUM_BUCKETS = 32
NUM_SLOTS = 3


def _zen_body(url: str) -> dict:
    if url.endswith("/chat/completions"):
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": "fake",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    return {
        "id": "resp-fake",
        "object": "response",
        "status": "completed",
        "model": "fake",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


class FakeEgress:
    def __init__(self, calls: list, fail_first_on_slot_0: bool = True) -> None:
        self.calls = calls
        self._seen_slot_zero = False
        self._fail_first = fail_first_on_slot_0
        self.clients = [self._client_for_slot(i) for i in range(NUM_SLOTS)]

    def num_slots(self) -> int:
        return NUM_SLOTS

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient:
        return self.clients[slot % NUM_SLOTS]

    def _client_for_slot(self, slot: int) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            self.calls.append(slot)
            if slot == 0 and self._fail_first and not self._seen_slot_zero:
                self._seen_slot_zero = True
                return httpx.Response(
                    429,
                    json={"error": {"message": "Too many requests today"}},
                    headers={"retry-after": "1"},
                )
            return httpx.Response(200, json=_zen_body(str(request.url)))

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def aclose(self) -> None:
        return None


@pytest.fixture()
def fake_world(tmp_path_factory):
    calls: list[int] = []
    egress = FakeEgress(calls)
    table = BucketTable(
        num_buckets=NUM_BUCKETS, num_slots=NUM_SLOTS, slot_cooldown_s=60.0
    )
    import tempfile

    from llms.proxy.store import ApiKey, Store

    data_dir = pathlib.Path(tempfile.mkdtemp())
    Store(data_dir=data_dir).save_keys([ApiKey(key="ak-team1")])
    settings = make_settings(data_dir=str(data_dir))
    app = create_app(settings)
    app.state.egress = egress
    app.state.bucket_table = table
    with TestClient(app) as tc:
        yield tc, table, calls


def _model_in_slot(slot: int, prefix: str = "probe-model") -> str:
    for i in range(500):
        model = f"{prefix}-{i}"
        if bucket_for("ak-team1", model, NUM_BUCKETS) % NUM_SLOTS == slot:
            return model
    raise AssertionError("no model found for slot")


def _model_in_affinity_slot(affinity: str, slot: int, prefix: str = "aff-model") -> str:
    for i in range(500):
        model = f"{prefix}-{i}"
        if bucket_for(affinity, model, NUM_BUCKETS) % NUM_SLOTS == slot:
            return model
    raise AssertionError("no model found for affinity slot")


def test_ratelimit_fails_fast_and_rebalances(fake_world):
    tc, table, calls = fake_world
    model = _model_in_slot(0)
    bucket = bucket_for("ak-team1", model, NUM_BUCKETS)
    first = tc.post("/ak-team1/v1/responses", json={"model": model, "input": "hi"})
    assert first.status_code == 429
    assert first.headers.get("retry-after") == "1"
    assert table.slot_for(bucket) == 1
    second = tc.post("/ak-team1/v1/responses", json={"model": model, "input": "hi"})
    assert second.status_code == 200
    assert calls == [0, 1]


def test_affinity_prefix_routes_and_buckets_independently(fake_world):
    tc, table, calls = fake_world
    model = _model_in_affinity_slot("ak-team1", 1)
    plain_bucket = bucket_for(None, model, NUM_BUCKETS)
    aff_bucket = bucket_for("ak-team1", model, NUM_BUCKETS)
    r = tc.post("/ak-team1/v1/responses", json={"model": model, "input": "hi"})
    assert r.status_code == 200
    assert calls == [1]
    assert table.slot_for(aff_bucket) == 1
    assert table.slot_for(plain_bucket) == plain_bucket % NUM_SLOTS


def test_stream_ratelimit_rebalances_next_request(fake_world):
    tc, table, calls = fake_world
    model = _model_in_slot(0)
    bucket = bucket_for("ak-team1", model, NUM_BUCKETS)
    first = tc.post(
        "/ak-team1/v1/responses", json={"model": model, "input": "hi", "stream": True}
    )
    assert first.status_code == 429
    assert table.slot_for(bucket) == 1
    second = tc.post(
        "/ak-team1/v1/responses", json={"model": model, "input": "hi", "stream": True}
    )
    assert second.status_code == 200
    assert calls == [0, 1]
