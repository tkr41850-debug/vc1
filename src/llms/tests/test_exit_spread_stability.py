"""RED-5: spread churn + attribution shift when the spread shrinks.

Two related divergences vs N singles:

A. Spread churn: slots pin to the slot index, and pick_port(slot) =
ports[slot % len(spread)]. When one exit is excluded, len changes and
(almost) every slot remaps to a different exit. With N singles, a
member's 429 only moves that member's 1/N share; everyone else's
buckets keep their exit. Measure: over the 1024-bucket table, how many
(bucket -> port) mappings change when the spread shrinks 3 -> 2?

B. warp_idx attribution shift: warp_idx is the position in the CURRENT
spread. After an exclusion, the same physical exit (same port) reports
a different warp_idx (e.g. port 44003 shifts 1 -> 0). Consecutive
recent[] entries for one exit then carry different warp_idx values,
breaking per-exit attribution over time.

Per TDD: watch fail, then fix minimal.
"""
from __future__ import annotations

NUM_BUCKETS = 1024


def _health3():
    from llms.proxy.providers import ProviderHealth, WarpExit

    return ProviderHealth(
        exits=[
            WarpExit(idx=0, ready=True, status="ok", socks=44001),
            WarpExit(idx=1, ready=True, status="ok", socks=44002),
            WarpExit(idx=2, ready=True, status="ok", socks=44003),
        ],
        fetched_at=1000.0,
    )


def _mk(tmp_path):
    from llms.proxy.config import Settings
    from llms.proxy.providers import Provider, ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="m", kind="warp", models=["gpt-*"], exits=3)])
    registry.runtime("m").boot_epoch = 1.0
    registry.runtime("m").health = _health3()
    return registry


def _mapping(egress, model="gpt-5"):
    from llms.proxy.buckets import BucketTable

    table = BucketTable(num_buckets=NUM_BUCKETS, num_slots=3)
    out = {}
    for b in range(NUM_BUCKETS):
        _, _, w = egress.resolve(model, bucket=b)
        out[b] = w.pick_port(table.slot_for(b))
    return out


def test_spread_shrink_churn_bounded(tmp_path):
    """Excluding 1 of 3 exits must move ~1/3 of buckets, not ~all."""
    from llms.proxy.egress import ProviderEgress

    reg = _mk(tmp_path)
    egress = ProviderEgress(None, registry=reg)
    before = _mapping(egress)
    reg.runtime("m").note_exit_ratelimited(1, 120.0, "exit 1: limited")
    after = _mapping(egress)
    moved = sum(1 for b in before if before[b] != after[b])
    # N-singles equivalent moves only the 429'd member's share (~1/3).
    # Full remap (slot % len) moves ~2/3+; bound it at 1/2 with margin.
    assert moved <= NUM_BUCKETS // 2, f"{moved}/{NUM_BUCKETS} buckets remapped"


def test_warp_idx_stable_per_physical_exit(tmp_path):
    """Same port must report the same warp_idx before/after an exclusion."""
    from llms.proxy.egress import ProviderEgress

    reg = _mk(tmp_path)
    egress = ProviderEgress(None, registry=reg)
    _, _, w = egress.resolve("gpt-5", bucket=0)
    before = {p: i for i, p in enumerate(w.ready_ports()) if p is not None}
    reg.runtime("m").note_exit_ratelimited(1, 120.0, "exit 1: limited")
    _, _, w2 = egress.resolve("gpt-5", bucket=0)
    after = {p: i for i, p in enumerate(w2.ready_ports()) if p is not None}
    for port in after:
        assert after[port] == before[port], (
            f"port {port}: warp_idx {before[port]} -> {after[port]}"
        )


def test_health_refresh_reapplies_live_holes(tmp_path, monkeypatch):
    """A health refresh must not silently un-exclude a still-limited exit."""
    import asyncio as _asyncio

    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider

    reg = _mk(tmp_path)
    provider = next(p for p in reg.load() if p.id == "m")

    class _Pool:
        async def refresh_statuses(self):
            return None

        def snapshot(self):
            h = _health3()
            return {
                "error": "",
                "exits": [
                    {
                        "idx": w.idx,
                        "ready": w.ready,
                        "status": w.status,
                        "reason": w.reason,
                        "socks": w.socks,
                        "registered": True,
                        "error": "",
                    }
                    for w in h.exits
                ],
            }

    async def _ensure_pool(p):
        return _Pool()

    monkeypatch.setattr(reg, "ensure_pool", _ensure_pool)
    egress = ProviderEgress(None, registry=reg)
    _, _, w = egress.resolve("gpt-5", bucket=0)
    assert [p for p in w.ready_ports() if p is not None] == [44001, 44002, 44003]

    reg.runtime("m").note_exit_ratelimited(1, 120.0, "exit 1: limited")
    _, _, w2 = egress.resolve("gpt-5", bucket=0)
    assert w2.pick_port(1) == 44003, w2.ready_ports()

    # Same ready set, new snapshot object (as refresh_health builds).
    reg.runtime("m").health = _health3()
    _asyncio.run(reg.refresh_health(provider, force=True))
    _, _, w3 = egress.resolve("gpt-5", bucket=0)
    assert w3.pick_port(1) == 44003, w3.ready_ports()
