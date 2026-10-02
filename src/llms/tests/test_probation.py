from __future__ import annotations


def test_probation_single_flight_then_promote(tmp_path):
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
            Provider(id="warp-1", kind="warp", models=["gpt-*"], exits=1),
            Provider(id="warp-2", kind="warp", models=["gpt-*"], exits=1),
        ]
    )
    # warp-1 just cooled off: probation, concurrency 1.
    rt1 = registry.runtime("warp-1")
    rt1.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
        fetched_at=1000.0,
    )
    rt1.probation = True
    rt1.in_flight = 1
    egress = ProviderEgress(None, registry=registry)
    # Second request must not ride the probing provider.
    picks = {egress.resolve("gpt-5", bucket=b)[0] for b in range(20)}
    assert picks == {"warp-2"}


def test_probation_promote_on_success_and_probe_backoff():
    """First success promotes; probe 429 sets 60s (or retry-after) backoff."""
    from llms.proxy.providers import ProviderRuntime

    rt = ProviderRuntime()
    rt.probation = True
    # Success path: promote.
    rt.probation = False
    assert rt.probation is False
    # Probe 429 with no retry-after: 60s.
    rt.probation = True
    rt.note_ratelimited(None, "probe limited")
    rt.probation = False
    assert 55.0 < rt.retry_in() <= 60.0
    # Probe 429 with retry-after: the given value.
    rt2 = ProviderRuntime()
    rt2.probation = True
    rt2.note_ratelimited(120.0, "probe limited")
    rt2.probation = False
    assert 115.0 < rt2.retry_in() <= 120.0


def test_from_retry_always_lands_in_probation(tmp_path):
    """Cooldown expiry enters ready-probation, never straight to ready."""
    import time as _time

    from llms.proxy.config import Settings
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
        derive_lifecycle,
    )

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    rt = registry.runtime("w1")
    rt.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
        fetched_at=_time.monotonic(),
    )
    rt.note_ratelimited(60.0, "limited")
    assert (
        derive_lifecycle(
            next(x for x in registry.load() if x.id == "w1"), rt, registry
        )[0]
        == "ratelimited"
    )
    # Cooldown expires: caller marks probation (see pipeline tail).
    rt.retry_until = 0.0
    rt.probation = True
    assert (
        derive_lifecycle(
            next(x for x in registry.load() if x.id == "w1"), rt, registry
        )[0]
        == "ready-probation"
    )


def test_lifecycle_matrix_covers_probation():
    import time as _time

    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRuntime,
        WarpExit,
        derive_lifecycle,
    )

    now = _time.monotonic()
    rt = ProviderRuntime()
    rt.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok")],
        fetched_at=now,
    )
    rt.probation = True
    p = Provider(id="w1", kind="warp", models=["gpt-*"], enabled=True, exits=1)
    assert derive_lifecycle(p, rt, now=now)[0] == "ready-probation"
    # Draining still beats probation.
    rt.in_flight = 2
    rt.drain_until = now + 300.0
    p2 = Provider(id="w1", kind="warp", models=["gpt-*"], enabled=False, exits=1)
    assert derive_lifecycle(p2, rt, now=now)[0] == "draining"
    # Ratelimited still beats probation.
    rt2 = ProviderRuntime()
    rt2.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok")],
        fetched_at=now,
    )
    rt2.probation = True
    rt2.retry_until = now + 60.0
    assert derive_lifecycle(p, rt2, now=now)[0] == "ratelimited"


def test_retry_expiry_gate_arms_probation(tmp_path):
    """Pre-track gate: expired backoff zeroes and arms; active/never untouched."""
    import time as _time

    from llms.proxy.config import Settings
    from llms.proxy.pipeline import _arm_probation_on_expiry
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
        derive_lifecycle,
    )

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    rt = registry.runtime("w1")
    rt.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
        fetched_at=_time.monotonic(),
    )
    # Expired backoff: gate zeroes and arms (never straight to ready).
    rt.retry_until = _time.monotonic() + 0.05
    rt.retry_reason = "limited"
    _time.sleep(0.06)
    assert _arm_probation_on_expiry(registry, "w1") is True
    assert rt.retry_until == 0.0
    assert rt.retry_reason == ""
    assert rt.probation is True
    assert (
        derive_lifecycle(
            next(x for x in registry.load() if x.id == "w1"), rt, registry
        )[0]
        == "ready-probation"
    )
    # No backoff recorded: gate is a no-op.
    rt.probation = False
    assert _arm_probation_on_expiry(registry, "w1") is False
    assert rt.probation is False
    # Active backoff: untouched (Task 4 discipline holds).
    rt.note_ratelimited(60.0, "limited")
    assert _arm_probation_on_expiry(registry, "w1") is False
    assert rt.retry_in() > 0
    assert rt.probation is False
    # Missing provider id: no-op, never raises.
    assert _arm_probation_on_expiry(registry, None) is False
    assert _arm_probation_on_expiry(None, "w1") is False


def test_reconnect_route_arms_probation(admin_client, monkeypatch):
    """POST reconnect clears backoff AND arms probation (never straight ready)."""
    import time as _time

    from llms.proxy.providers import ProviderHealth, WarpExit, derive_lifecycle

    class _Pool:
        async def reconnect(self):
            return {"ok": True}

        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {
                "error": "",
                "exits": [
                    {
                        "idx": 0,
                        "ready": True,
                        "status": "ok",
                        "socks": 40001,
                        "registered": True,
                        "error": "",
                    }
                ],
            }

    async def _ensure_pool(provider):
        return _Pool()

    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers",
        json={"id": "rc1", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    assert r.status_code == 201
    registry = tc.app.state.providers
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    rt = registry.runtime("rc1")
    rt.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
        fetched_at=_time.monotonic(),
    )
    rt.note_ratelimited(60.0, "limited")
    r = tc.post(
        "/api/admin/providers/rc1/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    assert rt.retry_until == 0.0
    assert rt.probation is True
    rc1 = next(x for x in registry.load() if x.id == "rc1")
    assert derive_lifecycle(rc1, rt, registry)[0] == "ready-probation"


def test_enable_arms_probation(admin_client, monkeypatch):
    """Re-enable arms probation; the ack snapshot reads ready-probation."""
    import time as _time

    from llms.proxy.providers import ProviderHealth, WarpExit

    async def _ensure_pool(provider):
        return None

    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers",
        json={
            "id": "rc2",
            "kind": "warp",
            "exits": 1,
            "models": ["gpt-*"],
            "enabled": False,
        },
    )
    assert r.status_code == 201
    registry = tc.app.state.providers
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    rt = registry.runtime("rc2")
    rt.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
        fetched_at=_time.monotonic(),
    )
    r = tc.put("/api/admin/providers/rc2", json={"enabled": True})
    assert r.status_code == 200
    assert rt.probation is True
    assert r.json()["lifecycle"] == "ready-probation"


def test_refresh_zero_to_ready_arms_probation(tmp_path, monkeypatch):
    """First ready snapshot (boot/recovery) enters probation, never ready."""
    import asyncio as _asyncio

    from llms.proxy.config import Settings
    from llms.proxy.providers import Provider, ProviderRegistry, derive_lifecycle

    class _Pool:
        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {
                "error": "",
                "exits": [
                    {
                        "idx": 0,
                        "ready": True,
                        "status": "ok",
                        "socks": 40001,
                        "registered": True,
                        "error": "",
                    }
                ],
            }

    async def _ensure_pool(provider):
        return _Pool()

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    provider = next(x for x in registry.load() if x.id == "w1")
    assert provider.kind == "warp"
    assert registry.runtime("w1").probation is False
    _asyncio.run(registry.refresh_health(provider, force=True))
    assert registry.runtime("w1").probation is True
    assert (
        derive_lifecycle(provider, registry.runtime("w1"), registry)[0]
        == "ready-probation"
    )
    # Steady-state refresh keeps the arm (no success yet) and never raises.
    _asyncio.run(registry.refresh_health(provider, force=True))
    assert registry.runtime("w1").probation is True


def test_lifespan_boot_stamps_boot_epoch(tmp_path, monkeypatch):
    """Lifespan boot stamps boot_epoch so restarts age into unhealthy."""
    from fastapi.testclient import TestClient

    from llms.proxy.main import create_app
    from llms.proxy.providers import Provider, ProviderRegistry
    from tests.conftest import make_settings

    async def _ensure_pool(self, provider):
        return None

    settings = make_settings(data_dir=str(tmp_path))
    seed = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    seed.save([Provider(id="boot1", kind="warp", models=["gpt-*"], exits=1)])
    monkeypatch.setattr(ProviderRegistry, "ensure_pool", _ensure_pool)
    app = create_app(settings)
    with TestClient(app):
        assert app.state.providers.runtime("boot1").boot_epoch > 0


def test_stream_wrap_settles_probation_on_exhaustion():
    """A fully-consumed stream probe promotes probation -> ready."""
    import asyncio as _asyncio
    import tempfile as _tf

    from fastapi.responses import StreamingResponse

    from llms.proxy.config import Settings
    from llms.proxy.pipeline import _inflight_track, _wrap_inflight
    from llms.proxy.providers import ProviderRegistry

    class _State:
        pass

    async def scenario(tmp):
        settings = Settings(data_dir=str(tmp))
        registry = ProviderRegistry(data_dir=str(tmp), settings=settings)
        state = _State()
        state.app = _State()
        state.app.state = _State()
        state.app.state.providers = registry
        state.app.state.admin_hub = None
        req = _State()
        req.app = state.app

        async def body():
            yield b"hello"

        rt = registry.runtime("w1")
        rt.probation = True
        _inflight_track(req, "w1")
        resp = _wrap_inflight(req, StreamingResponse(body()), "w1")
        chunks = [c async for c in resp.body_iterator]
        assert b"".join(chunks) == b"hello"
        assert registry.runtime("w1").in_flight == 0
        assert registry.runtime("w1").probation is False

    with _tf.TemporaryDirectory() as tmp:
        _asyncio.run(scenario(tmp))


def test_stream_wrap_disconnect_keeps_probation_armed():
    """Disconnect mid-body releases the slot but stays armed for re-probe."""
    import asyncio as _asyncio
    import tempfile as _tf

    from fastapi.responses import StreamingResponse

    from llms.proxy.config import Settings
    from llms.proxy.pipeline import _inflight_track, _wrap_inflight
    from llms.proxy.providers import ProviderRegistry

    class _State:
        pass

    async def scenario(tmp):
        settings = Settings(data_dir=str(tmp))
        registry = ProviderRegistry(data_dir=str(tmp), settings=settings)
        state = _State()
        state.app = _State()
        state.app.state = _State()
        state.app.state.providers = registry
        state.app.state.admin_hub = None
        req = _State()
        req.app = state.app

        async def body():
            yield b"part1"
            yield b"part2"

        rt = registry.runtime("w1")
        rt.probation = True
        _inflight_track(req, "w1")
        resp = _wrap_inflight(req, StreamingResponse(body()), "w1")
        first = await resp.body_iterator.__anext__()
        assert first == b"part1"
        await resp.body_iterator.aclose()
        assert registry.runtime("w1").in_flight == 0
        assert registry.runtime("w1").probation is True

    with _tf.TemporaryDirectory() as tmp:
        _asyncio.run(scenario(tmp))
