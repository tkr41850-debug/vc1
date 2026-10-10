from __future__ import annotations

import pytest


class FakePool:
    """Stand-in for the in-process WarpPool behind ProviderRegistry."""

    def __init__(self, ready: tuple[int, ...] = (1, 2)) -> None:
        self.ready = ready
        self.reconnects = 0

    async def reconnect(self) -> dict:
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
        async def reconnect(self):
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
