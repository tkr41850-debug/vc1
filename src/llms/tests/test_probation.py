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
    """Settle outcomes through the real function: 2xx/429 clear, 502 armed."""
    from fastapi.responses import JSONResponse

    from llms.proxy.pipeline import _settle_probation
    from llms.proxy.providers import ProviderRuntime

    class _State:
        pass

    def _settle(rt, status):
        state = _State()
        state.app = _State()
        state.app.state = _State()
        state.app.state.providers = None
        req = _State()
        req.app = state.app
        registry = _State()
        registry.runtime = lambda _pid, _rt=rt: _rt
        _settle_probation(
            req, registry, "w1", JSONResponse(status_code=status, content={})
        )

    # Success promotes.
    rt = ProviderRuntime()
    rt.probation = True
    _settle(rt, 200)
    assert rt.probation is False
    # Probe 429 clears the flag (backoff was applied by note_ratelimited
    # in the outcome block before settle runs).
    rt = ProviderRuntime()
    rt.probation = True
    rt.note_ratelimited(None, "probe limited")
    _settle(rt, 429)
    assert rt.probation is False
    assert 55.0 < rt.retry_in() <= 60.0
    # Probe 429 with retry-after: the given value.
    rt2 = ProviderRuntime()
    rt2.probation = True
    rt2.note_ratelimited(120.0, "probe limited")
    _settle(rt2, 429)
    assert rt2.probation is False
    assert 115.0 < rt2.retry_in() <= 120.0
    # Non-429 probe error stays armed for re-probe (Review Focus row 1:
    # a 502 must not promote, and must not let a joiner ride).
    rt3 = ProviderRuntime()
    rt3.probation = True
    _settle(rt3, 502)
    assert rt3.probation is True


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
        async def reconnect(self, bounce_guard=None):
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
    registry = tc.app.state.providers
    # Patch BEFORE create: POST create now boots + polls in the background
    # (spec §7), and refresh_health treats the pool snapshot as truth — a
    # real-pool refresh landing after the seed below would wipe the seeded
    # ready exit to zero-ready (preparing). The stub models this 1-exit
    # pool's ready exit, so any interleaving stays self-consistent.
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    r = tc.post(
        "/api/admin/providers",
        json={"id": "rc1", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    assert r.status_code == 201
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
    # Deferred clear (#4): backoff covers the background bounce window;
    # _restart clears it on completion. Pump until the bounce lands.
    deadline = _time.monotonic() + 10.0
    while rt.retry_in() > 0 and _time.monotonic() < deadline:
        _time.sleep(0.05)
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


def test_queue_signal_when_all_warp_at_quota(tmp_path):
    """Warp serves the model but every candidate is an at-quota probe: queued."""
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
    registry.save([Provider(id="w1", kind="warp", models=["gpt-*"], exits=1)])
    rt = registry.runtime("w1")
    rt.health = ProviderHealth(
        exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
        fetched_at=1000.0,
    )
    rt.probation = True
    rt.in_flight = 1
    egress = ProviderEgress(None, registry=registry)
    assert egress.resolve("gpt-5", bucket=0) == (None, "queued", None)


def test_queue_failover_to_next_warp_with_space(tmp_path):
    """Failover: next warp if space; else queue."""
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
        registry.runtime(pid).health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
    registry.runtime("w1").probation = True
    registry.runtime("w1").in_flight = 1
    egress = ProviderEgress(None, registry=registry)
    picks = {egress.resolve("gpt-5", bucket=b)[0] for b in range(20)}
    assert picks == {"w2"}


def test_no_warp_serving_model_fails_open_not_queued(tmp_path):
    """No warp serves the model: fail open to noproxy, never queue."""
    from llms.proxy.config import Settings
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="w1", kind="warp", models=["other-*"], exits=1)])
    egress = ProviderEgress(None, registry=registry)
    _provider_id, kind, _warp = egress.resolve("gpt-5", bucket=0)
    assert kind == "noproxy"


_QUEUED_MODEL = "deepseek-v4-flash-free"  # chat ingress == chat egress


def _queued_request(app, body: dict):
    """Minimal Starlette Request over a stub app (test_inflight_leaks pattern)."""
    import json as _json

    from starlette.requests import Request

    body_bytes = _json.dumps(body).encode()

    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "app": app,
    }
    return Request(scope, receive)


class _QueuedEgress:
    """Routes through the real ProviderEgress; stub client for fail-open."""

    def __init__(self, registry):
        import httpx as _httpx

        from llms.proxy.egress import ProviderEgress

        self._real = ProviderEgress(None, registry=registry)
        self._direct = _httpx.AsyncClient()

    def resolve(self, model, bucket=0):
        return self._real.resolve(model, bucket)

    def client_for(self, bucket, slot):
        return self._direct

    def sync_bucket_slots(self, table):
        return self._real.sync_bucket_slots(table)

    async def aclose(self):
        await self._direct.aclose()
        for egress in self._real._warp.values():
            await egress.aclose()


def _queued_app(registry, egress):
    from types import SimpleNamespace as _NS

    from llms.proxy.buckets import BucketTable

    return _NS(
        state=_NS(
            bucket_table=BucketTable(),
            egress=egress,
            providers=registry,
            usage=None,
            sessions=None,
            dedup=None,
            admin_hub=None,
        )
    )


def test_queued_hold_served_when_probe_releases(tmp_path, monkeypatch):
    """A queued waiter rides pipeline.run() through to warp once the probe clears."""
    import asyncio as _asyncio

    import llms.proxy.pipeline as _pipeline
    from llms.proxy.config import Settings
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
    )

    async def scenario():
        settings = Settings(
            data_dir=str(tmp_path), queue_keepalive_s=15.0, queue_wait_s=5.0
        )
        registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
        registry.save([Provider(id="w1", kind="warp", models=["deepseek-*"], exits=1)])
        rt = registry.runtime("w1")
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.probation = True
        rt.in_flight = 1
        egress = _QueuedEgress(registry)
        app = _queued_app(registry, egress)
        assert egress.resolve(_QUEUED_MODEL, bucket=0)[1] == "queued"

        async def _noop_refresh(provider):
            # Pool-less hermetic stand-in: health is already seeded above.
            return registry.runtime(provider.id).health

        monkeypatch.setattr(registry, "refresh_health", _noop_refresh)

        async def _ok_forward(*args, **kwargs):
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=200, content={"id": "chatcmpl-q", "choices": []}
            )

        monkeypatch.setattr(_pipeline, "forward", _ok_forward)

        async def release_soon():
            await _asyncio.sleep(0.05)
            registry.runtime("w1").in_flight = 0

        releaser = _asyncio.ensure_future(release_soon())
        try:
            body = {
                "model": _QUEUED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            }
            response = await _pipeline.run(_queued_request(app, body), settings, "chat")
        finally:
            await releaser
            await egress.aclose()
        return response, registry

    response, registry = _asyncio.run(scenario())
    assert response.status_code == 200
    assert registry.runtime("w1").in_flight == 0
    assert registry.runtime("w1").probation is False


def test_queued_degrades_past_deadline(tmp_path, monkeypatch):
    """Zero budget: pipeline.run() 429s with the actual budget, forward untouched."""
    import asyncio as _asyncio
    import json as _json

    import llms.proxy.pipeline as _pipeline
    from llms.proxy.config import Settings
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
    )

    async def scenario():
        settings = Settings(
            data_dir=str(tmp_path), queue_keepalive_s=0.5, queue_wait_s=0.0
        )
        registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
        registry.save([Provider(id="w1", kind="warp", models=["deepseek-*"], exits=1)])
        rt = registry.runtime("w1")
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.probation = True
        rt.in_flight = 1
        egress = _QueuedEgress(registry)
        app = _queued_app(registry, egress)
        assert egress.resolve(_QUEUED_MODEL, bucket=0)[1] == "queued"

        async def _must_not_run(*args, **kwargs):
            raise AssertionError("degraded waiter must not reach the upstream leg")

        monkeypatch.setattr(_pipeline, "forward", _must_not_run)
        try:
            body = {
                "model": _QUEUED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            }
            response = await _pipeline.run(_queued_request(app, body), settings, "chat")
        finally:
            await egress.aclose()
        return response, registry

    response, registry = _asyncio.run(scenario())
    assert response.status_code == 429
    # Degrade carries the actual budget, not a hard-coded 600 (an
    # operator QUEUE_WAIT_S=300 must not advertise 600).
    assert response.headers["retry-after"] == "0"
    assert _json.loads(response.body.decode())["error"]["type"] == "queue_timeout"
    # The waiter was never in-flight tracked, so nothing is owed on degrade.
    assert registry.runtime("w1").in_flight == 1


def test_queued_hold_aborts_on_disconnect(tmp_path, monkeypatch):
    """A waiter whose client disconnects aborts the hold (499, no forward)."""
    import asyncio as _asyncio

    import llms.proxy.pipeline as _pipeline
    from llms.proxy.config import Settings
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
    )

    async def scenario():
        settings = Settings(
            data_dir=str(tmp_path), queue_keepalive_s=0.01, queue_wait_s=60.0
        )
        registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
        registry.save([Provider(id="w1", kind="warp", models=["deepseek-*"], exits=1)])
        rt = registry.runtime("w1")
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.probation = True
        rt.in_flight = 1
        egress = _QueuedEgress(registry)
        app = _queued_app(registry, egress)

        async def _must_not_run(*args, **kwargs):
            raise AssertionError("disconnected waiter must not reach upstream")

        monkeypatch.setattr(_pipeline, "forward", _must_not_run)
        try:
            body = {
                "model": _QUEUED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            }
            req = _queued_request(app, body)

            async def _gone():
                return True

            req.is_disconnected = _gone  # type: ignore[method-assign]
            response = await _pipeline.run(req, settings, "chat")
        finally:
            await egress.aclose()
        return response, registry

    response, registry = _asyncio.run(scenario())
    assert response.status_code == 499
    assert registry.runtime("w1").in_flight == 1


def test_post_track_quota_recheck_bounces_joiner(tmp_path, monkeypatch):
    """A joiner racing the probe flight re-queues instead of tracking 2.

    Review Focus row 1: resolve→track spans two awaits, so a second
    arrival can resolve onto the same probation provider while the
    first is in flight. The post-track re-check releases the joiner's
    slot and holds it queued; in_flight never exceeds 1.
    """
    import asyncio as _asyncio

    import llms.proxy.pipeline as _pipeline
    from llms.proxy.config import Settings
    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRegistry,
        WarpExit,
    )

    async def scenario():
        settings = Settings(
            data_dir=str(tmp_path), queue_keepalive_s=0.01, queue_wait_s=0.0
        )
        registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
        registry.save([Provider(id="w1", kind="warp", models=["deepseek-*"], exits=1)])
        rt = registry.runtime("w1")
        rt.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=1000.0,
        )
        rt.probation = True
        # The probe flight is already tracked when the joiner resolves.
        rt.in_flight = 1
        egress = _QueuedEgress(registry)
        app = _queued_app(registry, egress)

        async def _flapping_forward(*args, **kwargs):
            raise AssertionError("bounced joiner must not reach upstream")

        monkeypatch.setattr(_pipeline, "forward", _flapping_forward)

        # refresh_health is a no-op: the race is in_flight moving under
        # resolve, not health.
        async def _noop_refresh(provider, force=False):
            return registry.runtime(provider.id).health

        monkeypatch.setattr(registry, "refresh_health", _noop_refresh)
        try:
            body = {
                "model": _QUEUED_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
            }
            response = await _pipeline.run(_queued_request(app, body), settings, "chat")
        finally:
            await egress.aclose()
        return response, registry

    response, registry = _asyncio.run(scenario())
    assert response.status_code == 429
    assert registry.runtime("w1").in_flight == 1


def test_stream_settle_stays_armed_on_parser_failure():
    """A fully-read stream with a failed parser outcome must not promote."""
    from fastapi.responses import StreamingResponse

    from llms.proxy.pipeline import _settle_probation
    from llms.proxy.providers import ProviderRuntime

    class _State:
        pass

    async def body():
        yield b"partial"

    for outcome in ("failed", "incomplete"):
        rt = ProviderRuntime()
        rt.probation = True
        req = _State()
        req.app = _State()
        req.state = _State()
        req.state.stream_outcome = outcome
        registry = _State()
        registry.runtime = lambda _pid, _rt=rt: _rt
        _settle_probation(req, registry, "w1", StreamingResponse(body()))
        assert rt.probation is True, outcome

    rt = ProviderRuntime()
    rt.probation = True
    req = _State()
    req.app = _State()
    req.state = _State()
    req.state.stream_outcome = "completed"
    registry = _State()
    registry.runtime = lambda _pid, _rt=rt: _rt
    _settle_probation(req, registry, "w1", StreamingResponse(body()))
    assert rt.probation is False


def test_lifespan_boot_forces_first_poll(tmp_path, monkeypatch):
    """Boot performs the first health poll: fetched_at moves without traffic."""
    from fastapi.testclient import TestClient

    from llms.proxy.main import create_app
    from llms.proxy.providers import Provider, ProviderRegistry
    from tests.conftest import make_settings

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
                    },
                ],
            }

    async def _ensure_pool(self, provider):
        return _Pool()

    settings = make_settings(data_dir=str(tmp_path))
    seed = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    seed.save([Provider(id="bootpoll", kind="warp", models=["gpt-*"], exits=1)])
    monkeypatch.setattr(ProviderRegistry, "ensure_pool", _ensure_pool)
    app = create_app(settings)
    with TestClient(app):
        rt = app.state.providers.runtime("bootpoll")
        assert rt.health.fetched_at > 0
        assert rt.probation is True


def test_stream_wrap_records_recent_on_exhaustion():
    """A fully-consumed stream probe lands in Last requests (live)."""
    import asyncio as _asyncio
    import tempfile as _tf
    import time as _time

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
        req.state = _State()

        async def body():
            yield b"hello"

        _inflight_track(req, "w1")
        resp = _wrap_inflight(
            req,
            StreamingResponse(body()),
            "w1",
            model="gpt-5",
            warp_idx=3,
            started=_time.monotonic(),
        )
        chunks = [c async for c in resp.body_iterator]
        assert b"".join(chunks) == b"hello"
        snap = registry.runtime("w1").recent_snapshot()
        assert len(snap) == 1
        assert snap[0]["model"] == "gpt-5"
        assert snap[0]["status"] == 200
        assert snap[0]["warp_idx"] == 3

    with _tf.TemporaryDirectory() as tmp:
        _asyncio.run(scenario(tmp))


def test_stream_wrap_disconnect_records_nothing():
    """A disconnected stream leaves no Last-requests entry (never completed)."""
    import asyncio as _asyncio
    import tempfile as _tf
    import time as _time

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
        req.state = _State()

        async def body():
            yield b"part1"
            yield b"part2"

        _inflight_track(req, "w1")
        resp = _wrap_inflight(
            req,
            StreamingResponse(body()),
            "w1",
            model="gpt-5",
            warp_idx=3,
            started=_time.monotonic(),
        )
        first = await resp.body_iterator.__anext__()
        assert first == b"part1"
        await resp.body_iterator.aclose()
        assert registry.runtime("w1").recent_snapshot() == []

    with _tf.TemporaryDirectory() as tmp:
        _asyncio.run(scenario(tmp))
