from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llms.proxy.providers import Provider, ProviderRegistry
from tests.conftest import TEST_HEADERS, make_settings


class FakeSlot:
    def __init__(self, idx: int, socks_port: int) -> None:
        self.idx = idx
        self.socks_port = socks_port


class FakeCyclePool:
    """WarpPool stand-in with bounce_exit + port→slot mapping."""

    def __init__(self, ports=(40001, 40002)) -> None:
        self.instances = [FakeSlot(i, p) for i, p in enumerate(ports)]
        self.bounces: list[int] = []
        self.block = False

    async def bounce_exit(self, idx: int) -> dict:
        import asyncio

        self.bounces.append(idx)
        if self.block:
            await asyncio.sleep(30)
        return {"ok": True, "idx": idx, "before": True, "after": True}

    async def refresh_statuses(self) -> None:
        return None


class Warp429Egress:
    """Egress whose warp client always 429s, direct client 200s.

    Mirrors ProviderEgress.resolve(): a cycling warp is skipped so the
    request fails over to direct.
    """

    def __init__(self, direct, calls: list, is_cycling=None) -> None:
        import httpx

        self._calls = calls
        self._is_cycling = is_cycling

        def warp_handler(request: httpx.Request) -> httpx.Response:
            self._calls.append("warp")
            return httpx.Response(
                429,
                json={"error": {"message": "Too many requests today"}},
                headers={"retry-after": "1"},
            )

        def direct_handler(request: httpx.Request) -> httpx.Response:
            self._calls.append("direct")
            return httpx.Response(
                200,
                json={
                    "id": "resp-fake",
                    "object": "response",
                    "status": "completed",
                    "model": "muse-spark",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        self._warp_client = httpx.AsyncClient(
            transport=httpx.MockTransport(warp_handler)
        )
        self._direct_client = httpx.AsyncClient(
            transport=httpx.MockTransport(direct_handler)
        )

    def num_slots(self) -> int:
        return 2

    def client_for(self, bucket: int, slot: int):
        return self._direct_client

    def resolve(self, model: str):
        # Mirror ProviderEgress: skip a cycling warp so traffic fails over.
        if self._is_cycling is not None and self._is_cycling():
            return "noproxy", "noproxy", None
        from llms.proxy.egress import WarpSocksEgress

        egress = WarpSocksEgress("pool1")
        egress.set_num_slots(2)
        egress.set_socks_ports([40001, 40002])

        async def _not_used(bucket: int, slot: int):
            raise AssertionError("unreachable")

        return "pool1", "warp", _WarpClientShim(self, egress)

    def sync_bucket_slots(self, table) -> bool:
        return False

    async def aclose(self) -> None:
        return None


class _WarpClientShim:
    """Wraps Warp429Egress to serve the warp client through resolve()."""

    def __init__(self, outer: Warp429Egress, egress) -> None:
        self._outer = outer
        self._egress = egress

    def client_for(self, bucket: int, slot: int):
        port = self._egress.pick_port(slot)
        assert port in (40001, 40002)
        return self._outer._warp_client

    def pick_port(self, slot: int):
        return self._egress.pick_port(slot)

    def ready_ports(self):
        return self._egress.ready_ports()


class FakeSupervisor:
    """WarpSupervisor stand-in: get() returns the test pool."""

    def __init__(self, pool) -> None:
        self._pool = pool

    def get(self, provider_id: str):
        return self._pool


@pytest.fixture()
def cycle_world(tmp_path, monkeypatch):
    from llms.proxy.buckets import BucketTable
    from llms.proxy.main import create_app

    calls: list[str] = []
    pool = FakeCyclePool()
    registry = ProviderRegistry(data_dir=tmp_path)
    registry._supervisor = FakeSupervisor(pool)

    async def _ensure_pool(provider):
        return pool

    async def _refresh(provider, force=False):
        from llms.proxy.providers import ProviderHealth, WarpExit

        rt = registry.runtime(provider.id)
        rt.health = ProviderHealth(
            fetched_at=1.0,
            exits=[
                WarpExit(idx=i, ready=True, socks=p)
                for i, p in ((0, 40001), (1, 40002))
            ],
        )
        return rt.health

    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    monkeypatch.setattr(registry, "refresh_health", _refresh)
    registry.save(
        [
            Provider(id="pool1", kind="warp", exits=2, models=["muse-*"]),
            Provider(
                id="noproxy",
                label="Direct",
                kind="noproxy",
                models=["muse-*"],
                enabled=True,
            ),
        ]
    )
    settings = make_settings(data_dir=str(tmp_path))
    from llms.proxy.keys import reset_cache
    from llms.proxy.store import ApiKey, Store

    reset_cache()
    Store(data_dir=tmp_path).save_keys([ApiKey(key="sk-test")])
    app = create_app(settings)
    app.state.egress = Warp429Egress(
        None, calls, is_cycling=lambda: registry.runtime("pool1").cycling
    )
    app.state.bucket_table = BucketTable(num_buckets=32, num_slots=8)
    app.state.providers = registry
    with TestClient(app) as tc:
        yield tc, registry, pool, calls


def _post(tc):
    return tc.post(
        "/v1/responses",
        json={"model": "muse-spark", "input": "hi"},
        headers=TEST_HEADERS,
    )


def _wait_cycling(registry, value: bool = True):
    import time

    for _ in range(100):
        if registry.runtime("pool1").cycling is value:
            break
        time.sleep(0.05)
    assert registry.runtime("pool1").cycling is value


def test_warp_429_triggers_bounce_and_returns_429(cycle_world):
    tc, registry, pool, calls = cycle_world
    r = _post(tc)
    assert r.status_code == 429
    # Fast-failover hint: warp just bounced, direct healthy -> ~1s.
    assert r.headers.get("retry-after") == "1"
    assert calls == ["warp"]
    # Bounce runs as a background task; TestClient portal lets it finish.
    import time

    for _ in range(100):
        if pool.bounces or not registry.runtime("pool1").cycling:
            break
        time.sleep(0.05)
    assert pool.bounces == [0] or pool.bounces == [1]
    assert registry.runtime("pool1").cycling is False


def test_next_request_fails_over_to_direct_while_cycling(cycle_world):
    """While the warp bounce is in flight, traffic fails open to direct."""
    tc, registry, pool, calls = cycle_world
    pool.block = True
    try:
        first = _post(tc)
        assert first.status_code == 429
        _wait_cycling(registry, True)
        # Direct is healthy: no shed — the request fails over and 200s.
        second = _post(tc)
        assert second.status_code == 200
        assert calls == ["warp", "direct"]
    finally:
        pool.block = False


def test_pool_dry_sheds_with_escalating_retry_after(cycle_world):
    """Direct also ratelimited + warp restarting: escalating 5s→60s shed."""
    tc, registry, pool, calls = cycle_world
    pool.block = True
    try:
        first = _post(tc)
        assert first.status_code == 429
        _wait_cycling(registry, True)
        # Direct now ratelimited too: pool is dry, shed escalates.
        registry.runtime("noproxy").note_ratelimited(30.0, "limited")
        shed1 = _post(tc)
        assert shed1.status_code == 429
        assert shed1.headers.get("retry-after") == "5"
        shed2 = _post(tc)
        assert shed2.headers.get("retry-after") == "6"
        shed3 = _post(tc)
        assert shed3.headers.get("retry-after") == "7"
        # Shed requests never reach upstream.
        assert calls == ["warp"]
    finally:
        pool.block = False


def test_retry_after_caps_at_60(cycle_world):
    tc, registry, _pool, _calls = cycle_world
    rt = registry.runtime("pool1")
    rt.cycling = True
    rt.cycle_hits = 0
    # Pool dry: direct ratelimited, warp mid-restart.
    registry.runtime("noproxy").note_ratelimited(600.0, "limited")
    try:
        last = None
        for _ in range(70):
            last = _post(tc)
        assert last.headers.get("retry-after") == "60"
        assert _calls == []
    finally:
        rt.cycling = False
        rt.cycle_hits = 0


def test_cooldown_suppresses_second_bounce(cycle_world):
    tc, registry, pool, _calls = cycle_world
    r = _post(tc)
    assert r.status_code == 429
    import time

    for _ in range(100):
        if pool.bounces or not registry.runtime("pool1").cycling:
            break
        time.sleep(0.05)
    assert len(pool.bounces) == 1
    # A second 429 within the cooldown must not bounce again.
    r2 = _post(tc)
    assert r2.status_code == 429
    for _ in range(20):
        time.sleep(0.05)
    assert len(pool.bounces) == 1


def test_noproxy_429_never_bounces(app_client, tmp_path, monkeypatch):
    import httpx

    tc, _ = app_client
    pool = FakeCyclePool()
    registry = ProviderRegistry(data_dir=tmp_path)
    monkeypatch.setattr(registry, "ensure_pool", lambda p: pool)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, json={"error": {"message": "Too many requests today"}}
        )

    from llms.proxy.egress import DirectEgress

    tc.app.state.egress = DirectEgress(
        httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    tc.app.state.providers = registry
    r = tc.post(
        "/v1/responses",
        json={"model": "muse-spark", "input": "hi"},
        headers=TEST_HEADERS,
    )
    assert r.status_code == 429
    assert pool.bounces == []


def test_bounce_exit_dedupes_concurrent():
    import asyncio

    from llms.proxy.warp import WarpPool

    pool = WarpPool.__new__(WarpPool)
    pool._bouncing = {3}
    pool.instances = []
    result = asyncio.run(pool.bounce_exit(3))
    assert result == {"ok": False, "idx": 3, "deduped": True}
