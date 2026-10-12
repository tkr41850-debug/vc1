"""RED-4: per-exit record must key on POOL exit idx, not spread position.

pipeline computes warp_idx as the request slot's POSITION in the ready
spread (ports.index(pick_port(slot))). note_exit_ratelimited(idx) and
the resolve() spread filter compare against WarpExit.idx (pool idx).
When the ready set is non-contiguous (e.g. idx 0 down, idx 1+2 ready),
spread position 0 is pool idx 1 — recording the position mis-steers:
the bad exit stays in the spread and an unrelated idx is excluded.

Per TDD: watch this fail, then fix the pipeline to map socks_port ->
pool idx via the health snapshot before recording.
"""
from __future__ import annotations


def _noncontig_health():
    from llms.proxy.providers import ProviderHealth, WarpExit

    return ProviderHealth(
        exits=[
            WarpExit(idx=0, ready=False, status="down", socks=44001),
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
    registry.runtime("m").health = _noncontig_health()
    return registry


def test_exit_record_keys_on_pool_idx_not_spread_position(tmp_path):
    """429 on spread position 0 (pool idx 1) must steer around pool idx 1."""
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.pipeline import _exit_idx_for_port

    reg = _mk(tmp_path)
    rt = reg.runtime("m")
    egress = ProviderEgress(None, registry=reg)
    _, _, w = egress.resolve("gpt-5", bucket=0)
    assert sorted(w.ready_ports()) == [44002, 44003]

    # The request rode spread position 0 = port 44002 = pool idx 1.
    port = w.pick_port(0)
    assert port == 44002
    pool_idx = _exit_idx_for_port(rt, port)
    assert pool_idx == 1, pool_idx

    rt.note_exit_ratelimited(pool_idx, 120.0, "exit 1: limited")
    assert rt.limited_exits() == {1}
    _, _, w2 = egress.resolve("gpt-5", bucket=0)
    assert w2.ready_ports() == [44003], w2.ready_ports()
