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


SECRET_HEADERS = {"Authorization": "Bearer sk-team1"}


@pytest.fixture()
def fake_world(tmp_path_factory):
    calls: list[int] = []
    egress = FakeEgress(calls)
    table = BucketTable(
        num_buckets=NUM_BUCKETS, num_slots=NUM_SLOTS, slot_cooldown_s=60.0
    )
    import tempfile

    from llms.proxy.keys import reset_cache
    from llms.proxy.store import ApiKey, Store

    reset_cache()
    data_dir = pathlib.Path(tempfile.mkdtemp())
    Store(data_dir=data_dir).save_keys([ApiKey(key="sk-team1")])
    settings = make_settings(data_dir=str(data_dir))
    app = create_app(settings)
    app.state.egress = egress
    app.state.bucket_table = table
    with TestClient(app) as tc:
        yield tc, table, calls


def _post(tc, path, body):
    return tc.post(path, json=body, headers=SECRET_HEADERS)


def _model_in_slot(slot: int, prefix: str = "probe-model") -> str:
    for i in range(500):
        model = f"{prefix}-{i}"
        if bucket_for("ak-team1", model, NUM_BUCKETS, "sk-team1") % NUM_SLOTS == slot:
            return model
    raise AssertionError("no model found for slot")


def _model_in_affinity_slot(affinity: str, slot: int, prefix: str = "aff-model") -> str:
    for i in range(500):
        model = f"{prefix}-{i}"
        if bucket_for(affinity, model, NUM_BUCKETS, "sk-team1") % NUM_SLOTS == slot:
            return model
    raise AssertionError("no model found for affinity slot")


def test_ratelimit_fails_fast_and_rebalances(fake_world):
    tc, table, calls = fake_world
    model = _model_in_slot(0)
    bucket = bucket_for("ak-team1", model, NUM_BUCKETS, "sk-team1")
    first = _post(tc, "/ak-team1/v1/responses", {"model": model, "input": "hi"})
    assert first.status_code == 429
    assert first.headers.get("retry-after") == "1"
    assert table.slot_for(bucket) == 1
    second = _post(tc, "/ak-team1/v1/responses", {"model": model, "input": "hi"})
    assert second.status_code == 200
    assert calls == [0, 1]


def test_affinity_prefix_routes_and_buckets_independently(fake_world):
    tc, table, calls = fake_world
    model = _model_in_affinity_slot("ak-team1", 1)
    plain_bucket = bucket_for(None, model, NUM_BUCKETS, "sk-team1")
    aff_bucket = bucket_for("ak-team1", model, NUM_BUCKETS, "sk-team1")
    r = _post(tc, "/ak-team1/v1/responses", {"model": model, "input": "hi"})
    assert r.status_code == 200
    assert calls == [1]
    assert table.slot_for(aff_bucket) == 1
    assert table.slot_for(plain_bucket) == plain_bucket % NUM_SLOTS


def test_stream_ratelimit_rebalances_next_request(fake_world):
    tc, table, calls = fake_world
    model = _model_in_slot(0)
    bucket = bucket_for("ak-team1", model, NUM_BUCKETS, "sk-team1")
    first = _post(
        tc, "/ak-team1/v1/responses", {"model": model, "input": "hi", "stream": True}
    )
    assert first.status_code == 429
    assert table.slot_for(bucket) == 1
    second = _post(
        tc, "/ak-team1/v1/responses", {"model": model, "input": "hi", "stream": True}
    )
    assert second.status_code == 200
    assert calls == [0, 1]


def test_ring_spreads_buckets_across_warps(tmp_path):
    """Ring routing (§7): buckets partition across warp providers."""
    from llms.proxy.config import Settings
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderHealth, ProviderRegistry, WarpExit

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save(
        [
            Provider(id="warp-1", kind="warp", models=["gpt-*"], exits=1),
            Provider(id="warp-2", kind="warp", models=["gpt-*"], exits=1),
            Provider(id="warp-3", kind="warp", models=["gpt-*"], exits=1),
        ]
    )
    for pid in ("warp-1", "warp-2", "warp-3"):
        rt = registry.runtime(pid)
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.boot_epoch = 1.0
    egress = ProviderEgress(None, registry=registry)
    seen = {egress.resolve("gpt-5", bucket=b)[0] for b in range(60)}
    assert seen == {"warp-1", "warp-2", "warp-3"}


def test_ring_skips_ratelimited_unless_all_limited(tmp_path):
    import time as _time

    from llms.proxy.config import Settings
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderHealth, ProviderRegistry, WarpExit

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save(
        [
            Provider(id="warp-1", kind="warp", models=["gpt-*"], exits=1),
            Provider(id="warp-2", kind="warp", models=["gpt-*"], exits=1),
        ]
    )
    for pid in ("warp-1", "warp-2"):
        rt = registry.runtime(pid)
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.boot_epoch = 1.0
    egress = ProviderEgress(None, registry=registry)
    registry.runtime("warp-1").note_ratelimited(120.0, "slow")
    # warp-1 limited: every bucket lands on warp-2.
    assert {egress.resolve("gpt-5", bucket=b)[0] for b in range(20)} == {"warp-2"}
    # Both limited: least-wait wins instead of failing open.
    registry.runtime("warp-2").note_ratelimited(300.0, "slower")
    assert egress.resolve("gpt-5", bucket=7)[0] == "warp-1"


def test_ring_skips_draining_provider(tmp_path):
    from llms.proxy.config import Settings
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderHealth, ProviderRegistry, WarpExit

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save(
        [
            Provider(id="warp-1", kind="warp", models=["gpt-*"], exits=1, enabled=False),
            Provider(id="warp-2", kind="warp", models=["gpt-*"], exits=1),
        ]
    )
    for pid in ("warp-1", "warp-2"):
        rt = registry.runtime(pid)
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.boot_epoch = 1.0
    # Disabled with in-flight work reads draining: cordoned from the ring.
    registry.runtime("warp-1").in_flight = 2
    egress = ProviderEgress(None, registry=registry)
    assert {egress.resolve("gpt-5", bucket=b)[0] for b in range(20)} == {"warp-2"}
