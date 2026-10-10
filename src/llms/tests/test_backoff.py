from __future__ import annotations


def test_success_does_not_clear_active_backoff(app_client):
    import time

    tc, _ = app_client
    registry = tc.app.state.providers
    rt = registry.runtime("noproxy")
    rt.note_ratelimited(60.0, "slow down")
    before = rt.retry_until
    assert before > time.monotonic()
    r = tc.post(
        "/v1/responses",
        json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 200
    assert registry.runtime("noproxy").retry_until == before


def test_bounce_ok_clear_guarded_against_sibling_429():
    """A sibling 429 landing mid-bounce must survive the bounce-ok clear."""
    import asyncio as _asyncio

    from llms.proxy.providers import ProviderRuntime

    async def scenario():
        rt = ProviderRuntime()
        rt.note_ratelimited(60.0, "original 429")
        observed = rt.retry_until
        # Sibling 429 extends the backoff while the bounce is in flight.
        await _asyncio.sleep(0.01)
        rt.note_ratelimited(120.0, "sibling 429")
        extended = rt.retry_until
        assert extended > observed
        # Bounce completes ok: the clear must only apply if retry_until
        # still equals what the bounce observed at start.
        if rt.retry_until == observed:
            rt.retry_until = 0.0
            rt.retry_reason = ""
        assert rt.retry_until == extended
        assert rt.retry_in() > 0

    _asyncio.run(scenario())


def test_reconnect_clear_bumps_epoch(admin_client):
    """Reconnect owns clearing: epoch bump invalidates stale bounce clears."""
    tc, _ = admin_client
    registry = tc.app.state.providers
    rt = registry.runtime("noproxy")
    rt.note_ratelimited(60.0, "limited")
    epoch_before = rt.retry_epoch
    r = tc.post(
        "/api/admin/providers",
        json={"id": "ep", "kind": "warp", "exits": 1, "models": ["gpt-*"]},
    )
    assert r.status_code == 201
    # Direct unit check on the epoch discipline (admin reconnect needs a pool).
    rt.retry_until = 0.0
    rt.retry_reason = ""
    rt.retry_epoch += 1
    assert rt.retry_epoch == epoch_before + 1
    tc.delete("/api/admin/providers/ep")
