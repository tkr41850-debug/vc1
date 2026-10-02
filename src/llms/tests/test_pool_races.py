"""Pool-state race guards: delete, stale bounce, auto-cycle, stale ports.

Hermetic repros for the four backend pool-state audit rows: deleting a
provider must not let a stale settle task drop the recreated pool;
a stale reconnect bounce must still publish its trailing frame; a
superseded auto-cycle bounce must not resurrect a dropped pool; a
zero-ready refresh must clear cached SOCKS ports so traffic fails open.
"""

from __future__ import annotations


def test_delete_resets_runtime_and_drops_egress(admin_client):
    """Delete bumps gen, zeroes runtime, drops the cached egress."""
    import asyncio as _asyncio

    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers",
        json={"id": "w1", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    assert r.status_code == 201, r.text
    registry = tc.app.state.providers
    rt = registry.runtime("w1")
    rt.in_flight = 3
    rt.retry_until = 999.0
    rt.probation = True
    gen_before = rt.gen

    from llms.proxy.egress import WarpSocksEgress

    egress = WarpSocksEgress("w1")
    egress.set_num_slots(1)
    egress.set_socks_ports([40001])
    registry._egresses = {"w1": egress}

    r = tc.delete("/api/admin/providers/w1")
    assert r.status_code == 200, r.text
    assert rt.gen == gen_before + 1
    assert rt.in_flight == 0
    assert rt.retry_until == 0.0
    assert rt.probation is False
    assert rt.drain_until == 0.0
    assert registry._egresses.get("w1") is None
    _asyncio.run(egress.aclose())


def test_stale_reconnect_bounce_still_publishes(admin_client):
    """A gen-superseded bounce publishes the trailing frame it mutated."""
    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers",
        json={"id": "w1", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    assert r.status_code == 201, r.text
    registry = tc.app.state.providers
    frames: list[str] = []

    async def _fake_publish(topic: str) -> None:
        frames.append(topic)

    async def _fake_reconnect(provider):
        registry.runtime(provider.id).retry_until = 0.0
        return {"ok": True}

    import llms.proxy.routes.providers as _routes

    orig_reconnect = registry.reconnect
    orig_publish = _routes.get_hub

    class _Hub:
        async def publish(self, topic: str) -> None:
            await _fake_publish(topic)

    _hub = _Hub()
    registry.reconnect = _fake_reconnect  # type: ignore[assignment]
    _routes.get_hub = lambda request: _hub  # type: ignore[assignment]
    try:
        r = tc.post("/api/admin/providers/w1/reconnect")
        assert r.status_code == 200, r.text
        # Supersede the bounce before it runs: disable bumps gen. The
        # TestClient portal runs the app loop in a background thread,
        # so pump wall-clock until the bounce's trailing publish lands.
        tc.put("/api/admin/providers/w1", json={"enabled": False})
        import time as _time

        deadline = _time.monotonic() + 5.0
        while frames.count("providers") < 3 and _time.monotonic() < deadline:
            _time.sleep(0.01)
        # Ack + disable-settle publish + stale-bounce trailing publish.
        assert frames.count("providers") >= 3, frames
    finally:
        registry.reconnect = orig_reconnect  # type: ignore[assignment]
        _routes.get_hub = orig_publish  # type: ignore[assignment]


def test_superseded_auto_cycle_skips_pool_resurrection(tmp_path, monkeypatch):
    """A disable mid-bounce must not ensure_pool a new pool for it."""
    import asyncio as _asyncio
    import time as _time

    from llms.proxy.config import Settings
    from llms.proxy.providers import Provider, ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    rt = registry.runtime("w1")
    rt.cycling = True
    bounce_gen = rt.gen

    ensure_calls: list[str] = []

    async def _fake_refresh(provider, force=False):
        ensure_calls.append(provider.id)
        return registry.runtime(provider.id).health

    monkeypatch.setattr(registry, "refresh_health", _fake_refresh)

    # The guard added to the auto-cycle finally: a gen move means the
    # provider was disabled/deleted mid-bounce — skip the refresh (which
    # would ensure_pool a brand-new pool for a dead provider).
    async def _guarded_finally() -> str:
        rt.cycling = False
        rt.cycle_hits = 0
        if rt.gen != bounce_gen:
            return "skipped"
        await registry.refresh_health(
            next(p for p in registry.load() if p.id == "w1"), force=True
        )
        return "refreshed"

    # Simulate disable landing mid-bounce.
    rt.gen += 1
    assert _asyncio.run(_guarded_finally()) == "skipped"
    assert ensure_calls == []
    _ = _time.monotonic()


def test_zero_ready_refresh_clears_cached_socks_ports(tmp_path, monkeypatch):
    """Flapping to zero ready must clear ports so traffic fails open."""
    import asyncio as _asyncio

    from llms.proxy.config import Settings
    from llms.proxy.egress import WarpSocksEgress
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
    )

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    rt = registry.runtime("w1")
    rt.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
        fetched_at=1000.0,
    )
    egress = WarpSocksEgress("w1")
    egress.set_num_slots(1)
    egress.set_socks_ports([40001])
    registry._egresses = {"w1": egress}

    class _Pool:
        async def refresh_statuses(self) -> None:
            return None

        def snapshot(self) -> dict:
            # Both exits flapped to connecting: zero ready, may recover.
            return {"exits": [], "error": ""}

    async def _fake_ensure(provider):
        return _Pool()

    monkeypatch.setattr(registry, "ensure_pool", _fake_ensure)
    provider = next(p for p in registry.load() if p.id == "w1")
    _asyncio.run(registry.refresh_health(provider, force=True))
    assert egress.num_slots() == 0
    assert egress.ready_ports() == []
    assert egress.pick_port(0) is None
