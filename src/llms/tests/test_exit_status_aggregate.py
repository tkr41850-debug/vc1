"""RED-6 (skeptic refutation a): one exit's 429 flips the WHOLE provider
to lifecycle 'ratelimited' despite a healthy majority.

derive_lifecycle checks provider retry_until before ready: a single
exit's 429 arms the aggregate, so a 1x3 with 2 healthy exits reads
'ratelimited' — while the Nx1 twin only flips the 429'd member.
The admin status dot goes yellow for capacity that is 2/3 healthy.

Per TDD: watch fail, then fix minimal (derive must consider healthy
un-limited siblings before the aggregate arm).
"""
from __future__ import annotations


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


def _mk(tmp_path, providers):
    from llms.proxy.config import Settings
    from llms.proxy.providers import ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save(providers)
    for p in providers:
        registry.runtime(p.id).boot_epoch = 1.0
    return registry


def test_one_exit_429_keeps_provider_ready_with_healthy_majority(tmp_path):
    """A lone exit 429 must not flip a 2/3-healthy provider to ratelimited."""
    from llms.proxy.providers import Provider, derive_lifecycle

    reg = _mk(tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)])
    rt = reg.runtime("m")
    rt.health = _health3()
    rt.note_exit_ratelimited(1, 120.0, "exit 1: limited")
    assert rt.retry_in() > 0
    provider = next(p for p in reg.load() if p.id == "m")
    lc, _ = derive_lifecycle(provider, rt, reg)
    assert lc == "ready", lc


def test_all_exits_limited_stays_ratelimited(tmp_path):
    """When every servable exit is limited, the aggregate still reads limited."""
    from llms.proxy.providers import Provider, derive_lifecycle

    reg = _mk(tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)])
    rt = reg.runtime("m")
    rt.health = _health3()
    for i in range(3):
        rt.note_exit_ratelimited(i, 120.0, f"exit {i}: limited")
    provider = next(p for p in reg.load() if p.id == "m")
    lc, _ = derive_lifecycle(provider, rt, reg)
    assert lc == "ratelimited", lc


def test_legacy_aggregate_without_exit_record_stays_ratelimited(tmp_path):
    """Legacy note_ratelimited (unknown exit) keeps the old status behavior."""
    from llms.proxy.providers import Provider, derive_lifecycle

    reg = _mk(tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)])
    rt = reg.runtime("m")
    rt.health = _health3()
    rt.note_ratelimited(120.0, "limited")
    provider = next(p for p in reg.load() if p.id == "m")
    lc, _ = derive_lifecycle(provider, rt, reg)
    assert lc == "ratelimited", lc
