from __future__ import annotations

from llms.proxy.providers import Provider, ProviderRegistry


def test_seed_has_noproxy_serving_free_models(tmp_path):
    from llms.proxy.router import FREE_MODELS

    registry = ProviderRegistry(data_dir=tmp_path)
    providers = registry.load()
    assert [p.id for p in providers] == ["noproxy"]
    assert providers[0].kind == "noproxy"
    assert providers[0].enabled is True
    for m in FREE_MODELS:
        assert providers[0].serves(m) is True
    assert providers[0].serves("gpt-unknown-thing") is False


def test_provider_serves_patterns(tmp_path):
    p = Provider(id="w", kind="warp", models=["gpt-*", "claude-exact"])
    assert p.serves("GPT-4o") is True
    assert p.serves("claude-exact") is True
    assert p.serves("claude-other") is False
    q = Provider(id="a", kind="warp", models=["*"])
    assert q.serves("anything-at-all") is True


def test_registry_roundtrip_and_route(tmp_path):
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(
        registry.load()
        + [
            Provider(
                id="warp-1",
                label="Local warps",
                kind="warp",
                models=["gpt-*"],
                enabled=True,
                exits=4,
            )
        ]
    )
    routed = registry.route("gpt-5")
    assert routed is not None and routed.id == "warp-1"
    assert routed.exits == 4
    routed = registry.route("muse-spark-1.3-contributor-free")
    assert routed is not None and routed.id == "noproxy"


def test_route_skips_disabled_warp(tmp_path):
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(
        registry.load()
        + [Provider(id="warp-1", kind="warp", models=["*"], enabled=False)]
    )
    routed = registry.route("gpt-5")
    assert routed is None


def test_base_url_rejected_with_migration_error(tmp_path):
    import yaml

    from llms.proxy.store import StoreError

    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(registry.load())
    path = registry.path()
    raw = yaml.safe_load(path.read_text()) or []
    raw.append(
        {
            "id": "warp-1",
            "kind": "warp",
            "base_url": "http://pool:8080",
            "models": ["*"],
        }
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    try:
        registry.load()
    except StoreError as exc:
        assert "no longer supported" in str(exc)
    else:
        raise AssertionError("expected StoreError for base_url")


def test_retry_in_decays_and_records():
    registry = ProviderRegistry(data_dir="/tmp/does-not-matter")
    rt = registry.runtime("warp-1")
    assert rt.retry_in() == 0.0
    rt.note_ratelimited(None, "slow down")
    assert 55.0 < rt.retry_in() <= 60.0
    assert rt.retry_reason == "slow down"
    rt.retry_until = 0.0
    assert rt.retry_in() == 0.0


def test_recent_ring_caps_at_ten():
    import asyncio

    from llms.proxy.providers import RecentRequest

    registry = ProviderRegistry(data_dir="/tmp/does-not-matter")
    rt = registry.runtime("warp-1")

    async def fill():
        for i in range(15):
            await rt.record(RecentRequest(ts=float(i), model="m", status=200, ms=1.0))

    asyncio.run(fill())
    snap = rt.recent_snapshot()
    assert len(snap) == 10
    assert snap[0]["ts"] == 5.0
    assert snap[-1]["ts"] == 14.0


def test_provider_admin_crud(admin_client, tmp_path):
    from llms.proxy.store import Store

    tc, _ = admin_client
    r = tc.get("/api/admin/providers")
    assert r.status_code == 200
    assert [p["id"] for p in r.json()["providers"]] == ["noproxy"]

    r = tc.post(
        "/api/admin/providers",
        json={
            "id": "warp-1",
            "label": "Pool",
            "kind": "warp",
            "exits": 4,
            "models": ["gpt-*"],
            "enabled": True,
        },
    )
    assert r.status_code == 201
    assert (
        tc.post(
            "/api/admin/providers",
            json={"id": "warp-1", "kind": "warp", "exits": 2},
        ).status_code
        == 409
    )
    assert (
        tc.post(
            "/api/admin/providers", json={"id": "w2", "kind": "warp", "exits": 0}
        ).status_code
        == 400
    )
    assert (
        tc.post("/api/admin/providers", json={"id": "w2", "kind": "bogus"}).status_code
        == 400
    )

    r = tc.put("/api/admin/providers/warp-1", json={"enabled": False})
    assert r.status_code == 200 and r.json()["enabled"] is False
    assert tc.delete("/api/admin/providers/noproxy").status_code == 403
    assert tc.delete("/api/admin/providers/warp-1").status_code == 200
    assert tc.delete("/api/admin/providers/warp-1").status_code == 404
    assert Store(data_dir=tmp_path).load_keys() is not None


def test_provider_admin_updates_models(admin_client):
    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers",
        json={
            "id": "warp-models",
            "label": "M",
            "kind": "warp",
            "exits": 1,
            "models": ["gpt-*"],
            "enabled": True,
        },
    )
    assert r.status_code == 201
    r = tc.put(
        "/api/admin/providers/warp-models",
        json={"models": ["claude-*", "muse-spark-*"]},
    )
    assert r.status_code == 200
    providers = tc.get("/api/admin/providers").json()["providers"]
    entry = next(p for p in providers if p["id"] == "warp-models")
    assert entry["models"] == ["claude-*", "muse-spark-*"]
    assert tc.put("/api/admin/providers/nope", json={"models": []}).status_code == 404


def test_warp_egress_socks_and_retry_tracking(monkeypatch):
    import asyncio

    import httpx as _httpx

    from llms.proxy.egress import ProviderEgress, WarpSocksEgress
    from llms.proxy.main import create_app
    from llms.proxy.store import ApiKey, Store
    from tests.conftest import TEST_HEADERS, TEST_SECRET, make_settings

    def _zen_handler(request: _httpx.Request) -> _httpx.Response:
        assert str(request.url).endswith("/responses")
        # Anonymous responses leg requires stream:true upstream: the
        # synthesize path folds SSE, not a JSON body, so the mock must
        # speak realistic SSE (a bare JSON body parses as zero deltas
        # and the truncation rule now reads that as incomplete, not
        # completed).
        return _httpx.Response(
            200,
            content=(
                b'data: {"type": "response.created",'
                b' "response": {"id": "resp_1"}}\n\n'
                b'data: {"type": "response.completed",'
                b' "response": {"id": "resp_1", "status": "completed",'
                b' "usage": {"input_tokens": 1, "output_tokens": 1}}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    zen_client = _httpx.AsyncClient(transport=_httpx.MockTransport(_zen_handler))

    import pathlib
    import tempfile

    data_dir = pathlib.Path(tempfile.mkdtemp())
    Store(data_dir=data_dir).save_keys([ApiKey(key=TEST_SECRET)])
    registry = ProviderRegistry(data_dir=data_dir)
    registry.save(
        registry.load()
        + [
            Provider(
                id="warp-1",
                kind="warp",
                exits=2,
                models=["muse-spark*"],
            )
        ]
    )

    from fastapi.testclient import TestClient

    from llms.proxy.buckets import BucketTable
    from llms.proxy.egress import DirectEgress

    upstream = _httpx.AsyncClient(
        transport=_httpx.MockTransport(
            lambda req: _httpx.Response(200, json={"unexpected": True})
        )
    )
    app = create_app(make_settings(data_dir=str(data_dir)))
    app.state.providers = registry
    app.state.egress = ProviderEgress(DirectEgress(upstream), registry=registry)
    app.state.bucket_table = BucketTable(num_buckets=1024, num_slots=1)
    # lifespan would overwrite our ProviderEgress with a fresh DirectEgress
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    app.router.lifespan_context = _noop_lifespan

    async def _fake_health(provider, force=False):
        from llms.proxy.providers import ProviderHealth, WarpExit

        rt = registry.runtime("warp-1")
        rt.health = ProviderHealth(
            exits=[
                WarpExit(idx=1, ready=True, socks=40001, registered=True),
                WarpExit(idx=2, ready=True, socks=40002, registered=True),
            ],
            fetched_at=1.0,
        )
        return rt.health

    monkeypatch.setattr(registry, "refresh_health", _fake_health)
    monkeypatch.setattr(
        WarpSocksEgress, "client_for", lambda self, bucket, slot: zen_client
    )
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers=TEST_HEADERS,
        )
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "completed"
        assert r.headers.get("x-egress-provider") == "warp-1"
        # admin routes need the require_admin override; direct registry check:
        rt = registry.runtime("warp-1")
        snap = rt.recent_snapshot()
        assert len(snap) == 1
        assert snap[0]["model"] == "muse-spark-1.3-contributor-free"
        assert snap[0]["status"] == 200
    asyncio.run(registry.aclose())
    asyncio.run(zen_client.aclose())
    asyncio.run(upstream.aclose())


def test_provider_exits_capped(admin_client):
    tc, _ = admin_client
    r = tc.post(
        "/api/admin/providers",
        json={"id": "big", "kind": "warp", "exits": 500, "models": ["gpt-*"]},
    )
    assert r.status_code == 400
    r = tc.post(
        "/api/admin/providers",
        json={"id": "big", "kind": "warp", "exits": 4, "models": ["gpt-*"]},
    )
    assert r.status_code == 201
    r = tc.put("/api/admin/providers/big", json={"exits": 500})
    assert r.status_code == 200
    providers = tc.get("/api/admin/providers").json()["providers"]
    assert next(p for p in providers if p["id"] == "big")["exits"] == 32


def test_provider_stream_no_gap_between_snapshot_and_subscribe():
    import asyncio

    from llms.proxy.providers import ProviderRuntime, RecentRequest

    async def scenario():
        rt = ProviderRuntime()
        await rt.record(
            RecentRequest(ts=1.0, model="m", status=200, ms=1.0, warp_idx=None)
        )
        q, snap = await rt.subscribe_snapshot()
        # Record landing right after the atomic call: must be queued, and
        # must NOT already be in the snapshot (no duplicates either).
        await rt.record(
            RecentRequest(ts=2.0, model="m", status=200, ms=1.0, warp_idx=None)
        )
        assert [r["ts"] for r in snap] == [1.0]
        assert q.qsize() == 1
        assert (await q.get()).ts == 2.0
        await rt.unsubscribe(q)

    asyncio.run(scenario())


def test_lifecycle_precedence_matrix():
    import time as _time

    from llms.proxy.providers import (
        Provider,
        ProviderHealth,
        ProviderRuntime,
        WarpExit,
        derive_lifecycle,
    )

    # retry_until is a monotonic deadline: anchor "now" to the real clock
    # so retry_in() > 0 actually holds during the ratelimited cases.
    now = _time.monotonic()

    def rt(**kw):
        r = ProviderRuntime()
        for k, v in kw.items():
            setattr(r, k, v)
        return r

    def warp(enabled=True, ready=0, total=2, fetched=100.0, error=""):
        return Provider(
            id="w1", kind="warp", models=["gpt-*"], enabled=enabled, exits=total
        ), rt(
            health=ProviderHealth(
                exits=[
                    WarpExit(idx=i, ready=i < ready, status="ok") for i in range(total)
                ],
                fetched_at=fetched,
                error=error,
            )
        )

    # off: disabled, no pool, no in-flight
    p, r = warp(enabled=False)
    assert derive_lifecycle(p, r, now=now)[0] == "off"
    # draining: disabled + in-flight
    p, r = warp(enabled=False)
    r.in_flight = 3
    r.drain_until = now + 300.0
    lc, drain = derive_lifecycle(p, r, now=now)
    assert lc == "draining"
    # until_ms is remaining-ms (the UI renders it as a countdown), not
    # the absolute monotonic deadline: 300s out reads ~300000ms here
    # and converges to 0, independent of process uptime.
    assert drain == {"until_ms": int(300.0 * 1000), "forced": False}
    # draining beats ratelimited
    r.retry_until = now + 300.0
    assert derive_lifecycle(p, r, now=now)[0] == "draining"
    # ratelimited overlay on enabled
    p, r = warp(enabled=True, ready=1)
    r.retry_until = now + 300.0
    assert derive_lifecycle(p, r, now=now)[0] == "ratelimited"
    # preparing: never fetched
    p, r = warp(enabled=True, fetched=0.0)
    assert derive_lifecycle(p, r, now=now)[0] == "preparing"
    # preparing: within boot grace
    p, r = warp(enabled=True)
    r.boot_epoch = now - 10.0
    assert derive_lifecycle(p, r, now=now)[0] == "preparing"
    # unhealthy: past grace, still zero ready
    assert derive_lifecycle(p, r, now=now + 200.0)[0] == "unhealthy"
    # ready: enabled with a ready exit
    p, r = warp(enabled=True, ready=1)
    assert derive_lifecycle(p, r, now=now + 200.0)[0] == "ready"
    # ready: noproxy always serves when enabled
    n = Provider(id="noproxy", kind="noproxy", models=["*"], enabled=True)
    assert derive_lifecycle(n, rt(), now=now + 200.0)[0] == "ready"
    assert (
        derive_lifecycle(
            Provider(id="noproxy", kind="noproxy", models=["*"], enabled=False),
            rt(),
            now=now + 200.0,
        )[0]
        == "off"
    )


def test_inflight_wrap_releases_on_full_consume():
    import asyncio

    from fastapi.responses import StreamingResponse

    from llms.proxy.pipeline import _inflight_release, _inflight_track, _wrap_inflight
    from llms.proxy.providers import ProviderRegistry

    class _State:
        pass

    async def scenario(tmp_path):
        from llms.proxy.config import Settings

        settings = Settings(data_dir=str(tmp_path))
        registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
        state = _State()
        state.app = _State()
        state.app.state = _State()
        state.app.state.providers = registry
        state.app.state.admin_hub = None
        req = _State()
        req.app = state.app

        async def body():
            yield b"hello"

        _inflight_track(req, "w1")
        assert registry.runtime("w1").in_flight == 1
        resp = _wrap_inflight(req, StreamingResponse(body()), "w1")
        chunks = [c async for c in resp.body_iterator]
        assert b"".join(chunks) == b"hello"
        assert registry.runtime("w1").in_flight == 0
        # Guard never goes negative on double release.
        _inflight_release(req, "w1")
        assert registry.runtime("w1").in_flight == 0

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(scenario(tmp_path=__import__("pathlib").Path(tmp)))


def test_toggle_returns_transition_snapshot_fast(admin_client):
    import time as _time

    tc, _ = admin_client
    tc.post(
        "/api/admin/providers",
        json={"id": "flip", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    start = _time.monotonic()
    r = tc.put("/api/admin/providers/flip", json={"enabled": False})
    elapsed = _time.monotonic() - start
    assert r.status_code == 200
    # Ack timing (§2): no warp-cli await in the request path.
    assert elapsed < 1.0
    body = r.json()
    assert body["enabled"] is False
    # No pool exists in this hermetic client (lifespan skipped), so the
    # derived state is off; with a live pool it would read draining.
    assert body["lifecycle"] in ("draining", "off")
    assert body["drain"] is None or body["drain"]["forced"] is False
    r = tc.put("/api/admin/providers/flip", json={"enabled": True})
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["lifecycle"] == "preparing"


def test_generation_cancel_reenable_mid_drain(admin_client):
    import asyncio as _asyncio

    tc, _ = admin_client
    tc.post(
        "/api/admin/providers",
        json={"id": "flip2", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    registry = tc.app.state.providers
    tc.put("/api/admin/providers/flip2", json={"enabled": False})
    rt = registry.runtime("flip2")
    rt.in_flight = 5
    first_gen = rt.gen
    tc.put("/api/admin/providers/flip2", json={"enabled": True})
    assert rt.gen == first_gen + 1

    async def settle():
        await _asyncio.sleep(0.1)

    _asyncio.run(settle())


def test_publish_throttled_never_blocks_caller():
    import asyncio as _asyncio
    import time as _time

    from llms.proxy.admin_hub import AdminHub

    async def scenario():
        hub = AdminHub()
        q = await hub.subscribe("providers")
        for _ in range(3):
            t0 = _time.monotonic()
            await hub.publish_throttled("providers", delay_s=0.2)
            assert _time.monotonic() - t0 < 0.1
            await _asyncio.sleep(0.3)
        # Each call landed in a closed window (0.3 > 0.2 cooldown), so
        # each published immediately: 3 frames, no trailing.
        got = 0
        deadline = _time.monotonic() + 2.0
        while _time.monotonic() < deadline:
            try:
                await _asyncio.wait_for(q.get(), 0.3)
                got += 1
            except TimeoutError:
                break
        assert got == 3, got


def test_ipv4_transport_pins_local_address():
    # Live 2026-10-06: this box resolves opencode.ai to IPv6 first and
    # IPv6 egress is dead, so the default Happy-Eyeballs dial fails the
    # whole connection (downstream 502 `upstream unreachable`) while a
    # forced `-4` curl answers 200. The direct upstream transport must
    # bind 0.0.0.0 so it takes the IPv4 path.
    import httpx as _httpx

    from llms.proxy.egress import ipv4_transport

    transport = ipv4_transport()
    assert isinstance(transport, _httpx.AsyncHTTPTransport)
    # httpcore stashes the bind address on the pool: assert it took.
    assert transport._pool._local_address == "0.0.0.0"


def test_publish_throttled_coalesces_burst():
    import asyncio as _asyncio
    import time as _time

    from llms.proxy.admin_hub import AdminHub

    async def scenario():
        hub = AdminHub()
        q = await hub.subscribe("providers")
        # Tight burst inside one window: immediate + one trailing.
        for _ in range(5):
            t0 = _time.monotonic()
            await hub.publish_throttled("providers", delay_s=0.2)
            assert _time.monotonic() - t0 < 0.1
        got = 0
        deadline = _time.monotonic() + 2.0
        while _time.monotonic() < deadline:
            try:
                await _asyncio.wait_for(q.get(), 0.3)
                got += 1
            except TimeoutError:
                break
        assert got == 2, got

    _asyncio.run(scenario())

    _asyncio.run(scenario())


def test_inflight_changes_push_providers_frame(admin_client):
    """Data-plane in-flight moves push a providers frame (no CRUD needed).

    Regression for Busy stuck / stale state: previously only the request
    tail (success release) published, and the tail publish raced the
    response send — hermetically the frame never arrived before the
    response. Publishing at track time (before the upstream leg) plus at
    release guarantees the Busy 1->0 transition is observable on SSE.
    """
    import asyncio as _asyncio
    import threading as _threading
    import time as _time

    tc, _ = admin_client
    hub = tc.app.state.admin_hub
    registry = tc.app.state.providers
    from llms.proxy.buckets import BucketTable
    from llms.proxy.egress import ProviderEgress

    # build_app_client wires a bare DirectEgress (no resolve/provider id,
    # so the Busy tick attributes nowhere); wrap it as lifespan does.
    _inner = tc.app.state.egress
    tc.app.state.egress = ProviderEgress(_inner, registry=registry)
    tc.app.state.bucket_table = BucketTable(num_buckets=1024, num_slots=1)

    gate = _threading.Event()

    async def _scenario():
        q = await hub.subscribe("providers")
        # Drain any pending frames so the test starts quiet.
        while True:
            try:
                q.get_nowait()
            except _asyncio.QueueEmpty:
                break
        # Hold the upstream leg open so in_flight is observable mid-flight.
        import llms.proxy.pipeline as _pl

        _orig_forward = _pl.forward

        async def _gated_forward(*args, **kwargs):
            gate.wait(10)
            return await _orig_forward(*args, **kwargs)

        _pl.forward = _gated_forward
        try:
            t = _threading.Thread(
                target=lambda: tc.post(
                    "/v1/responses",
                    json={
                        "model": "muse-spark-1.3-contributor-free",
                        "input": "hi",
                    },
                    headers={
                        "Authorization": "Bearer sk-test",
                    },
                ),
                daemon=True,
            )
            t.start()
            # Busy tick up: track-time publish fires before the gated leg.
            saw_busy = False
            deadline = _time.monotonic() + 10.0
            while _time.monotonic() < deadline:
                try:
                    await _asyncio.wait_for(q.get(), 1.0)
                except TimeoutError:
                    continue
                rt = registry.runtime("noproxy")
                if rt.in_flight >= 1:
                    saw_busy = True
                    break
            assert saw_busy, "no track-time push while upstream held"
            gate.set()
            t.join(timeout=20)
            # Busy back to zero: release-time publish after the response.
            deadline = _time.monotonic() + 10.0
            while _time.monotonic() < deadline:
                try:
                    await _asyncio.wait_for(q.get(), 1.0)
                except TimeoutError:
                    continue
                if registry.runtime("noproxy").in_flight == 0:
                    break
            assert registry.runtime("noproxy").in_flight == 0
            await hub.unsubscribe("providers", q)
        finally:
            _pl.forward = _orig_forward
            gate.set()

    _asyncio.run(_scenario())


def test_create_provider_rejects_traversal_id(admin_client):
    """Provider ids are path segments: ../ and / must 400, never mkdir."""
    tc, _ = admin_client
    for bad in ("../evil", "..\\evil", "/abs", "a/b", "a..b/../c", ""):
        r = tc.post(
            "/api/admin/providers",
            json={"id": bad, "kind": "warp", "exits": 1, "models": ["gpt-*"]},
        )
        assert r.status_code == 400, (bad, r.status_code)
    r = tc.post(
        "/api/admin/providers",
        json={"id": "warp-ok_1", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    assert r.status_code == 201, r.text


def test_provider_id_routes_reject_traversal(admin_client):
    """Every {provider_id:path} route 400s traversal ids (backstop in _find).

    Note: "/../evil" style paths normalize at the HTTP layer before
    routing, so the directly reachable shapes are encoded dots and
    absolute ids — all must 400, never reach the pool.
    """
    tc, _ = admin_client
    for bad in ("%2e%2e%2fevil", "/abs"):
        assert tc.get(f"/api/admin/providers/{bad}/health").status_code == 400
        assert tc.post(f"/api/admin/providers/{bad}/reconnect").status_code == 400
        r = tc.delete(f"/api/admin/providers/{bad}")
        assert r.status_code == 400, (bad, r.status_code)


def test_create_warp_boots_and_polls_in_background(admin_client, monkeypatch):
    """POST create triggers background boot+poll: health moves without traffic."""
    import time as _time

    class _FakePool:
        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {"error": "", "exits": []}

    async def _ensure_pool(provider):
        return _FakePool()

    tc, _ = admin_client
    registry = tc.app.state.providers
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    r = tc.post(
        "/api/admin/providers",
        json={"id": "bootcreate", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    assert r.status_code == 201
    deadline = _time.monotonic() + 5.0
    while registry.runtime("bootcreate").health.fetched_at <= 0:
        assert _time.monotonic() < deadline, "background boot poll never ran"
        _time.sleep(0.05)
