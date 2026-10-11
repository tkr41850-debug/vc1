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


def test_stale_cycling_cordon_ages_out_of_ring(tmp_path):
    """A cycling cordon older than the stall window must not steer routing.

    Live 2026-10-11 (citr): four ready-probation pools showed
    cycling:true + in_flight 0 with healthy exits — cordoned from the
    resolve() ring, zero requests logged. No bounce survives a restart
    and none runs longer than minutes, so a cordon older than the stall
    window is a strand, not a live bounce: the ring must include the
    pool again (and log loudly) instead of starving it forever.

    The strand predates the cycling_since stamp field, so the test
    sets a bare cycling=True (since 0.0): on a process older than the
    stall window that must age out too.
    """
    from llms.proxy.config import Settings
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
    )

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save(
        [
            Provider(id="w1", kind="warp", models=["gpt-*"], exits=1),
            Provider(id="w2", kind="warp", models=["gpt-*"], exits=1),
        ]
    )
    for pid in ("w1", "w2"):
        rt = registry.runtime(pid)
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.boot_epoch = 1.0
    rt1 = registry.runtime("w1")
    rt1.probation = True
    rt1.cycling = True
    assert rt1.cycling_since == 0.0
    egress = ProviderEgress(None, registry=registry)
    seen = {egress.resolve("gpt-5", bucket=b)[0] for b in range(60)}
    assert seen == {"w1", "w2"}, seen


def test_sweep_restores_concurrent_auto_cycle_cordon(tmp_path):
    """A sweep overlapping an auto-cycle bounce must not drop its cordon.

    Live 2026-10-11 (citr warp-1/2/3): ready-probation + cycling:true +
    in_flight 0, zero requests, fresh cycle_cooldown_remaining. The wedge:
    an auto-cycle bounce holds cycling=True across its per-exit bounce;
    a concurrent self-heal sweep (whole-pool _restart) save/restores the
    flag — was_cycling=True means the sweep's finally restores True, and
    when the auto-cycle bounce lands first its clear drops a cordon the
    sweep is still holding. The loser-to-finish leaves cycling=True with
    no bounce behind it, and the ring starves the pool (resolvable only
    by the 10-min stale age-out).

    Repro: auto-cycle arms the cordon, then a sweep _restart overlaps and
    finishes after the auto-cycle bounce lands. The pool must rejoin the
    ring (cycling False) once both complete.
    """
    import asyncio as _asyncio
    import time as _time

    from llms.proxy.config import Settings
    from llms.proxy.providers import Provider, ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    provider = next(p for p in registry.load() if p.id == "w1")
    rt = registry.runtime("w1")

    release_sweep = _asyncio.Event()
    sweep_entered = _asyncio.Event()
    cycle_done = _asyncio.Event()

    class _OverlapPool:
        async def reconnect(self, bounce_guard=None):
            # Sweep's whole-pool bounce: enters while the auto-cycle
            # bounce is in flight, lands after it.
            sweep_entered.set()
            await release_sweep.wait()
            return {"ok": True}

        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {"error": "", "exits": []}

    async def _fake_ensure(_provider):
        return _OverlapPool()

    async def _fake_refresh(_provider, force=False):
        return registry.runtime(_provider.id).health

    registry.ensure_pool = _fake_ensure  # type: ignore[assignment]
    registry.refresh_health = _fake_refresh  # type: ignore[assignment]

    async def _main() -> bool:
        # Auto-cycle arms the cordon (mirrors the pipeline set site).
        now = _time.monotonic()
        rt.cycling = True
        rt.cycling_since = now
        rt.last_auto_cycle = now
        # Sweep _restart overlaps: saves was_cycling=True, holds the
        # cordon across its own bounce window.
        sweep_task = _asyncio.create_task(
            registry._restart(provider, clear_backoff=False)
        )
        await sweep_entered.wait()
        # Auto-cycle bounce lands first: clears the cordon.
        rt.cycling = False
        rt.cycling_since = 0.0
        cycle_done.set()
        # Sweep lands after: must not resurrect the dropped cordon.
        release_sweep.set()
        await sweep_task
        return rt.cycling

    assert _asyncio.run(_main()) is False


def test_cancelled_restart_restores_cycling_cordon(tmp_path):
    """Cancelling _restart mid-bounce must not strand cycling=True.

    The restore after the reconnect try/except only runs on the
    success/Exception paths: CancelledError (BaseException) skips it
    and the ring cordons the pool forever — the citr 2026-10-11 strand
    (ready-probation + cycling:true + in_flight 0, zero requests).
    """
    import asyncio as _asyncio

    from llms.proxy.config import Settings
    from llms.proxy.providers import Provider, ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    provider = next(p for p in registry.load() if p.id == "w1")
    started = _asyncio.Event()

    async def _hang(_provider):
        started.set()
        await _asyncio.Event().wait()

    async def _fake_refresh(_provider, force=False):
        return registry.runtime(_provider.id).health

    registry.ensure_pool = _hang  # type: ignore[assignment]
    registry.refresh_health = _fake_refresh  # type: ignore[assignment]

    async def _main() -> bool:
        task = _asyncio.create_task(registry._restart(provider, clear_backoff=False))
        await started.wait()
        task.cancel()
        try:
            await task
        except _asyncio.CancelledError:
            pass
        return registry.runtime("w1").cycling

    assert _asyncio.run(_main()) is False
