from __future__ import annotations

import pytest


class FakePool:
    """Stand-in for the in-process WarpPool behind ProviderRegistry."""

    def __init__(self, ready: tuple[int, ...] = (1, 2)) -> None:
        self.ready = ready
        self.reconnects = 0

    async def reconnect(self, bounce_guard=None) -> dict:
        self.reconnects += 1
        return {
            "ok": True,
            "before": {"ready": 0, "exits": 2},
            "after": {"ready": 2, "exits": 2},
        }

    async def refresh_statuses(self) -> None:
        return None

    def snapshot(self) -> dict:
        return {
            "error": "",
            "exits": [
                {
                    "idx": i,
                    "ready": True,
                    "status": "Connected",
                    "reason": "",
                    "socks": 40000 + i,
                    "registered": True,
                    "error": "",
                }
                for i in self.ready
            ],
        }


def test_reconnect_bounces_pool_and_resyncs(admin_client, tmp_path, monkeypatch):
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderRegistry

    pool = FakePool()

    async def _ensure_pool(provider):
        return pool

    tc, _ = admin_client
    table = tc.app.state.bucket_table
    registry = ProviderRegistry(data_dir=tmp_path)
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    registry.save([Provider(id="pool1", kind="warp", exits=2, models=["muse-*"])])
    egress = ProviderEgress(tc.app.state.egress, registry=registry)
    assert egress.resolve("muse-spark")[0] == "pool1"
    tc.app.state.egress = egress
    tc.app.state.providers = registry
    assert table.num_slots == 1
    r = tc.post(
        "/api/admin/providers/pool1/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    body = r.json()
    # Ack (§2): the transition snapshot returns at once; the bounce
    # settles in the background and publishes over SSE.
    assert body["id"] == "pool1"
    assert body["reconnect"] == {
        "started": True,
        "before": {"ready": 0, "exits": 0, "error": ""},
    }
    assert body["lifecycle"] in ("preparing", "ready", "unhealthy", "ratelimited")


@pytest.mark.parametrize(
    ("provider_id", "expected"),
    [("noproxy", 400), ("nope", 404)],
    ids=["noproxy", "unknown"],
)
def test_reconnect_errors(admin_client, provider_id, expected):
    tc, _ = admin_client
    r = tc.post(
        f"/api/admin/providers/{provider_id}/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == expected


def test_local_reconnect_pool_uses_secret_key(app_client, tmp_path, monkeypatch):
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderRegistry

    pool = FakePool()

    async def _ensure_pool(provider):
        return pool

    tc, _ = app_client
    registry = ProviderRegistry(data_dir=tmp_path)
    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    registry.save([Provider(id="pool1", kind="warp", exits=2, models=["muse-*"])])
    tc.app.state.egress = ProviderEgress(tc.app.state.egress, registry=registry)
    tc.app.state.providers = registry
    r = tc.post(
        "/api/providers/pool1/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "pool1"
    assert body["reconnect"]["started"] is True
    assert body["lifecycle"] in ("preparing", "ready", "unhealthy", "ratelimited")


def test_local_reconnect_pool_requires_key(app_client):
    tc, _ = app_client
    assert tc.post("/api/providers/pool1/reconnect").status_code == 401


@pytest.mark.parametrize(
    ("provider_id", "expected"),
    [("noproxy", 400), ("nope", 404)],
    ids=["noproxy", "unknown"],
)
def test_local_reconnect_pool_errors(app_client, provider_id, expected):
    tc, _ = app_client
    r = tc.post(
        f"/api/providers/{provider_id}/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == expected


def test_reconnect_clears_retry(admin_client, tmp_path, monkeypatch):
    """Manual reconnect drops the 429 backoff (already the contract)."""
    import time

    from llms.proxy.providers import Provider, ProviderRegistry

    tc, _ = admin_client
    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save([Provider(id="pool1", kind="warp", exits=1, models=["*"])])

    class _Pool:
        async def reconnect(self, bounce_guard=None):
            return {"ok": True}

        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {"error": "", "exits": []}

    async def _ensure_pool(provider):
        return _Pool()

    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    tc.app.state.providers = registry
    rt = registry.runtime("pool1")
    rt.note_ratelimited(60.0, "slow down")
    assert rt.retry_until > time.monotonic()
    r = tc.post(
        "/api/admin/providers/pool1/reconnect",
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    # Deferred clear (#4): the ack arms probation but leaves the backoff
    # covering the bounce window — _restart clears it only after the
    # bounce completes. The TestClient portal runs the app loop in a
    # background thread, so pump wall-clock until the bounce lands.
    deadline = time.monotonic() + 10.0
    while rt.retry_in() > 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert rt.retry_until == 0.0
    assert rt.retry_reason == ""


@pytest.mark.parametrize("raiser", ["ensure_pool", "refresh_statuses"])
def test_boot_failure_still_moves_health_off_preparing(tmp_path, monkeypatch, raiser):
    """Boot-time pool failure must not orphan the lifecycle row.

    Live finding: warps sat in `preparing` for 4h+ until a manual Debug
    forced-refresh. Root cause: the lifespan boot block called
    ensure_pool() before stamping boot_epoch and before the inner
    refresh_health — a raise skipped all three, leaving fetched_at 0
    and boot_epoch 0, so derive_lifecycle's preparing arm (fetched_at
    <= 0, short-circuiting even the boot-grace expiry) held forever.
    The fix stamps the epoch and moves fetched_at on both failure
    paths; the row ages into `unhealthy` past the grace and the error
    surfaces in the snapshot instead of sticking on boot.
    """
    import asyncio as _asyncio
    import time as _time

    from llms.proxy.providers import (
        BOOT_GRACE_S,
        Provider,
        ProviderRegistry,
        derive_lifecycle,
    )

    registry = ProviderRegistry(data_dir=tmp_path)

    class _FailPool:
        async def refresh_statuses(self):
            if raiser == "refresh_statuses":
                raise RuntimeError("daemon bring-up timed out")

        def snapshot(self):
            return {"error": "", "exits": []}

    async def _ensure_pool(provider):
        if raiser == "ensure_pool":
            raise RuntimeError("registration failed: ratelimited")
        return _FailPool()

    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    provider = Provider(id="warp-1", kind="warp", exits=1, models=["*"])
    registry.save([provider])
    rt = registry.runtime("warp-1")
    # Lifespan boot ordering (main.py): stamp epoch even when the pool
    # fails, then still poll so fetched_at moves and the error lands.
    try:
        _asyncio.run(registry.ensure_pool(provider))
    except Exception:
        rt.boot_epoch = _time.monotonic()
    else:
        rt.boot_epoch = _time.monotonic()
    _asyncio.run(registry.refresh_health(provider, force=True))

    assert rt.health.fetched_at > 0
    assert "registration failed" in rt.health.error or "timed out" in rt.health.error
    lc, _ = derive_lifecycle(provider, rt, None, now=_time.monotonic())
    assert lc == "preparing"
    lc, _ = derive_lifecycle(
        provider, rt, None, now=_time.monotonic() + BOOT_GRACE_S + 1
    )
    assert lc == "unhealthy", lc


async def _await_task(coro):
    """Drive a maybe_self_heal decision coro to its sweep result."""

    task = await coro
    if task is None:
        return []
    return await task


def _await_sweep(coro):
    import asyncio as _asyncio

    return _asyncio.run(_await_task(coro))


def _self_heal_world(tmp_path, monkeypatch, states: dict[str, str]):
    """Registry with canned lifecycle states per provider id.

    states maps provider id -> one of ready / ready-probation /
    ratelimited / unhealthy / preparing. Ready states get a ready exit
    in health; ratelimited gets retry_until set; reconnects are counted.
    """
    import time as _time

    from llms.proxy.providers import Provider, ProviderRegistry, WarpExit

    registry = ProviderRegistry(data_dir=tmp_path)
    registry.save(
        [Provider(id=pid, kind="warp", exits=1, models=["*"]) for pid in states]
    )
    reconnects: list[str] = []

    class _Pool:
        async def reconnect(self, bounce_guard=None):
            return {"ok": True}

        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {"error": "", "exits": []}

    async def _ensure_pool(provider):
        return _Pool()

    async def _bounce(provider):
        # Production bounce() restarts exits WITHOUT clearing the
        # backoff — the provider stays ratelimited until the cooldown
        # elapses or a post-bounce success clears it. Record the bounce;
        # leave runtime state untouched. (The sweep reports bounce
        # success, not post-bounce lifecycle, so no state change is
        # needed for the outcome.)
        reconnects.append(provider.id)
        return {"ok": True}

    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    monkeypatch.setattr(registry, "bounce", _bounce)
    for pid, lc in states.items():
        rt = registry.runtime(pid)
        if lc in ("ready", "ready-probation"):
            from llms.proxy.providers import ProviderHealth

            rt.health = ProviderHealth(
                fetched_at=_time.monotonic(),
                exits=[WarpExit(idx=0, ready=True, socks=40001)],
            )
            rt.probation = lc == "ready-probation"
        elif lc == "ratelimited":
            from llms.proxy.providers import ProviderHealth

            rt.health = ProviderHealth(
                fetched_at=_time.monotonic(),
                exits=[WarpExit(idx=0, ready=False)],
            )
            rt.note_ratelimited(60.0, "limited")
    return registry, reconnects


def test_self_heal_restarts_limited_when_half_pool_dry(tmp_path, monkeypatch):
    """2/3 running providers ratelimited -> both restart, staggered."""
    import asyncio as _asyncio

    registry, reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ratelimited", "w3": "ready"},
    )
    sleeps: list[float] = []
    real_sleep = _asyncio.sleep

    async def _spy_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(_asyncio, "sleep", _spy_sleep)
    try:
        done = _await_sweep(registry.maybe_self_heal("*", "t1"))
    finally:
        monkeypatch.setattr(_asyncio, "sleep", real_sleep)
    assert sorted(done) == ["w1", "w2"]
    assert sorted(reconnects) == ["w1", "w2"]
    assert sleeps == [1.0]


def test_self_heal_quiet_below_half(tmp_path, monkeypatch):
    """1/3 ratelimited -> no restart."""

    registry, reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ready", "w3": "ready"},
    )
    done = _await_sweep(registry.maybe_self_heal("*", "t1"))
    assert done == []
    assert reconnects == []


def test_self_heal_skips_already_recovered(tmp_path, monkeypatch):
    """A provider that left Ratelimit before its turn is never bounced."""
    import asyncio as _asyncio
    import time as _time

    registry, reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ratelimited"},
    )
    real_sleep = _asyncio.sleep

    async def _sibling_recovers_w2(delay):
        # The 1s stagger before w2's turn: a sibling sweep restarts w2
        # first, so the pre-restart skip must fire and the bounce never
        # happens (reconnects stays ["w1"]).
        rt = registry.runtime("w2")
        rt.retry_until = 0.0
        rt.retry_reason = ""
        from llms.proxy.providers import ProviderHealth, WarpExit

        rt.health = ProviderHealth(
            fetched_at=_time.monotonic(),
            exits=[WarpExit(idx=0, ready=True, socks=40001)],
        )

    monkeypatch.setattr(_asyncio, "sleep", _sibling_recovers_w2)
    try:
        done = _await_sweep(registry.maybe_self_heal("*", "t1"))
    finally:
        monkeypatch.setattr(_asyncio, "sleep", real_sleep)
    assert done == ["w1"]
    assert reconnects == ["w1"]


def test_self_heal_serializes_overlapping_sweeps(tmp_path, monkeypatch):
    """Concurrency=1: a second sweep while one holds the lock is a no-op."""
    import asyncio as _asyncio

    registry, reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ready"},
    )

    async def scenario():
        async with registry._self_heal_lock:
            task = await registry.maybe_self_heal("*", "t1")
            assert task is None
        return reconnects

    assert _asyncio.run(scenario()) == []


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ("ready", "ratelimited", True),
        ("ready-probation", "unhealthy", True),
        ("ready", "ready-probation", False),
        ("ready-probation", "ready", False),
        ("ready", "ready", False),
        ("unhealthy", "ratelimited", False),
        (None, "ratelimited", False),
        ("ready", None, False),
    ],
)
def test_left_ready_trigger_predicate(before, after, expected):
    from llms.proxy.pipeline import _left_ready

    assert _left_ready(before, after) is expected


def test_distinct_sessions_hash_distinct_buckets():
    """Same key + same model + distinct sessions must spread buckets."""
    from llms.proxy.affinity import bucket_for

    buckets = {
        bucket_for(None, "muse-spark-1.3-contributor-free", 6, "sk-x", f"ses_{i:026d}")
        for i in range(10)
    }
    assert len(buckets) > 1
    # Same session still pins (prompt cache stays warm).
    assert bucket_for(None, "m", 1024, "sk-x", "ses_same") == bucket_for(
        None, "m", 1024, "sk-x", "ses_same"
    )


def test_self_heal_skips_recently_bounced(tmp_path, monkeypatch):
    """A sweep inside the re-bounce cooldown never re-bounces.

    Regression for the live re-bounce churn: stale in-flight 429s
    re-fired the sweep ~15s after the last bounce and the same 3-4
    warps bounced for 7+ minutes without recovering. A provider
    bounced within SELF_HEAL_COOLDOWN_S is skipped (stays counted in
    the ratio, so the gate still fires for its siblings).
    """

    registry, reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ratelimited", "w3": "ready"},
    )
    first = _await_sweep(registry.maybe_self_heal("*", "t1"))
    assert sorted(first) == ["w1", "w2"]
    assert sorted(reconnects) == ["w1", "w2"]
    # Immediate second sweep: gate still fires (2/3 limited) but both
    # providers are inside the cooldown — no re-bounce.
    second = _await_sweep(registry.maybe_self_heal("*", "t2"))
    assert second == []
    assert sorted(reconnects) == ["w1", "w2"]


def test_self_heal_rebounces_after_cooldown(tmp_path, monkeypatch):
    """A still-limited provider is bounced again past the cooldown."""
    import time as _time

    from llms.proxy.providers import SELF_HEAL_COOLDOWN_S

    registry, reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ready"},
    )
    assert _await_sweep(registry.maybe_self_heal("*", "t1")) == ["w1"]
    # Age the bounce past the cooldown; the provider never recovered
    # (backoff re-armed by traffic), so the next sweep re-bounces it.
    registry.runtime("w1").last_self_heal -= SELF_HEAL_COOLDOWN_S + 1.0
    registry.runtime("w1").note_ratelimited(60.0, "still limited")
    assert _await_sweep(registry.maybe_self_heal("*", "t2")) == ["w1"]
    assert reconnects == ["w1", "w1"]
    _ = _time.monotonic()


def test_self_heal_drains_in_flight_before_bounce(tmp_path, monkeypatch):
    """A queued restart waits for in-flight to clear before bouncing.

    The 429 only cordons (new traffic steers away); the tunnel stays
    open with live flights, and reconnect() disconnects ALL exits —
    bouncing immediately kills healthy sibling-exit requests. The
    sweep must let in-flight drain (target <= 1: the triggering 429's
    own request is still counted) before firing.
    """
    import asyncio as _asyncio

    registry, _reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ready"},
    )
    rt = registry.runtime("w1")
    rt.in_flight = 3
    order: list[str] = []
    orig_bounce = registry.bounce

    async def _spy_bounce(provider):
        order.append(f"bounce@{rt.in_flight}")
        return await orig_bounce(provider)

    async def _drain():
        await _asyncio.sleep(0.5)
        rt.in_flight = 1  # trigger's own flight remains
        order.append("drained")

    monkeypatch.setattr(registry, "bounce", _spy_bounce)

    async def scenario():
        drainer = _asyncio.create_task(_drain())
        sweep = await registry.maybe_self_heal("*", "t1")
        assert sweep is not None
        done = await sweep
        await drainer
        return done

    assert _asyncio.run(scenario()) == ["w1"]
    assert order == ["drained", "bounce@1"]


def test_self_heal_drain_expiry_bounces_anyway(tmp_path, monkeypatch):
    """A wedged body must not pin the sweep: expiry bounces."""

    from llms.proxy.providers import SELF_HEAL_DRAIN_S

    assert SELF_HEAL_DRAIN_S <= 30, "drain bound must stay well under tunnel bring-up"
    registry, _reconnects = _self_heal_world(
        tmp_path,
        monkeypatch,
        {"w1": "ratelimited", "w2": "ready"},
    )
    registry.runtime("w1").in_flight = 5  # never drains
    assert _await_sweep(registry.maybe_self_heal("*", "t1")) == ["w1"]
    assert _reconnects == ["w1"]
