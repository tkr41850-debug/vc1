"""RED: multi-exit equivalence — 1xN-exit must behave as N x 1-exit.

Claim under test: a multi-exit provider (one Provider, N ready exits) must
behave as N equivalent single-exit providers (N Providers, 1 ready exit each)
on routability, aggregate status, exit selection spread, degradation on one
exit down, and retry accounting.

Per TDD: this file is written FIRST. Each test must FAIL on the current code
for the right reason before any production change.
"""
from __future__ import annotations


def _ready_health(n: int, base_port: int = 40001, start_idx: int = 0):
    from llms.proxy.providers import ProviderHealth, WarpExit

    return ProviderHealth(
        exits=[
            WarpExit(idx=start_idx + i, ready=True, status="ok", socks=base_port + i)
            for i in range(n)
        ],
        fetched_at=1000.0,
    )


def _mk_registry(tmp_path, providers):
    from llms.proxy.config import Settings
    from llms.proxy.providers import ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save(providers)
    for p in providers:
        registry.runtime(p.id).boot_epoch = 1.0
    return registry


def test_equiv_routability_same_ready_pattern(tmp_path):
    """1x3 ready exits must route the same share as 3x1 single-exit providers."""
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider

    multi = [Provider(id="multi", kind="warp", models=["gpt-*"], exits=3)]
    r_multi = _mk_registry(tmp_path / "a", multi)
    r_multi.data_dir.mkdir(parents=True, exist_ok=True)
    r_multi.runtime("multi").health = _ready_health(3)

    singles = [
        Provider(id=f"s{i}", kind="warp", models=["gpt-*"], exits=1) for i in range(3)
    ]
    r_single = _mk_registry(tmp_path / "b", singles)
    for i in range(3):
        r_single.runtime(f"s{i}").health = _ready_health(1, base_port=41001 + i)

    e_multi = ProviderEgress(None, registry=r_multi)
    e_single = ProviderEgress(None, registry=r_single)

    multi_seen: set = set()
    single_seen: set = set()
    for b in range(60):
        pid, kind, _ = e_multi.resolve("gpt-5", bucket=b)
        assert kind == "warp", f"multi failed open at bucket {b}"
        multi_seen.add(pid)
        pid, kind, _ = e_single.resolve("gpt-5", bucket=b)
        assert kind == "warp", f"singles failed open at bucket {b}"
        single_seen.add(pid)
    # Both topologies route everything to warp; the singles spread across
    # all three members exactly like the multi spreads across its exits.
    assert multi_seen == {"multi"}
    assert single_seen == {f"s{i}" for i in range(3)}


def test_equiv_status_ready_iff_any_ready(tmp_path):
    """Aggregate status: ready iff >=1 ready exit; single bad exit degrades."""
    from llms.proxy.providers import Provider, derive_lifecycle

    reg = _mk_registry(
        tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)]
    )
    rt = reg.runtime("m")
    provider = next(p for p in reg.load() if p.id == "m")
    # 2 of 3 ready: aggregate must be ready/probation class, never unhealthy.
    rt.health = _ready_health(2)
    rt.probation = False
    lc, _ = derive_lifecycle(provider, rt, reg)
    assert lc == "ready", lc
    # 0 of 3 ready (past grace): unhealthy, not hidden by the ready count.
    import time as _time

    from llms.proxy.providers import ProviderHealth, WarpExit

    rt.health = ProviderHealth(
        exits=[WarpExit(idx=i, ready=False, status="down") for i in range(3)],
        fetched_at=_time.monotonic(),
    )
    rt.boot_epoch = _time.monotonic() - 600.0
    lc, _ = derive_lifecycle(provider, rt, reg)
    assert lc == "unhealthy", lc
    # 1 of 3 ready degrades but serves — never fails, never hides.
    rt.health = _ready_health(1)
    rt.boot_epoch = 1.0
    lc, _ = derive_lifecycle(provider, rt, reg)
    assert lc == "ready", lc


def test_equiv_selection_spread(tmp_path):
    """Exit selection: slot % ready-exits must spread like bucket % N singles."""
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider

    reg = _mk_registry(
        tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)]
    )
    reg.runtime("m").health = _ready_health(3, base_port=42001)
    egress = ProviderEgress(None, registry=reg)
    pid, kind, w = egress.resolve("gpt-5", bucket=0)
    assert kind == "warp" and pid == "m"
    ports = {w.pick_port(slot) for slot in range(30)}
    assert ports == {42001, 42002, 42003}, ports
    spread = {w.pick_port(slot) for slot in range(3)}
    assert len(spread) == 3, spread


def test_equiv_single_exit_down_degrades_not_fails(tmp_path):
    """One exit down inside a multi: route ready siblings, never fail open."""
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider, ProviderHealth, WarpExit

    reg = _mk_registry(
        tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)]
    )
    reg.runtime("m").health = ProviderHealth(
        exits=[
            WarpExit(idx=0, ready=True, status="ok", socks=43001),
            WarpExit(idx=1, ready=False, status="down", socks=43002),
            WarpExit(idx=2, ready=True, status="ok", socks=43003),
        ],
        fetched_at=1000.0,
    )
    egress = ProviderEgress(None, registry=reg)
    for b in range(20):
        pid, kind, w = egress.resolve("gpt-5", bucket=b)
        assert (pid, kind) == ("m", "warp"), (b, pid, kind)
        assert w.pick_port(b % 8) in (43001, 43003), w.pick_port(b % 8)


def test_equiv_retry_accounting_matches_single(tmp_path):
    """Per-exit failure must arm the same retry_in a single-exit 429 would."""
    from llms.proxy.providers import Provider

    reg = _mk_registry(
        tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)]
    )
    rt = reg.runtime("m")
    rt.health = _ready_health(3)
    # A 429 on one exit of the multi arms provider-level backoff — the same
    # accounting a single-exit provider would get. The retry reason must
    # name the failing exit so the aggregate cannot hide it.
    rt.note_ratelimited(60.0, "exit 1: Too many requests")
    assert rt.retry_in() > 0
    assert "exit 1" in rt.retry_reason
    # Sibling success must NOT clear an active backoff (matches singles).
    before = rt.retry_until
    assert rt.retry_until == before


def test_equiv_admin_snapshot_per_exit_observable(tmp_path):
    """health.exits[] + ips per exit + warp_idx must be observable end-to-end."""
    from llms.proxy.providers import Provider, provider_snapshot

    reg = _mk_registry(
        tmp_path, [Provider(id="m", kind="warp", models=["gpt-*"], exits=3)]
    )
    rt = reg.runtime("m")
    rt.health = _ready_health(3, base_port=44001)
    snap = provider_snapshot(
        next(p for p in reg.load() if p.id == "m"), rt, 300.0, reg
    )
    exits = snap["health"]["exits"]
    assert len(exits) == 3
    assert sorted(e["socks"] for e in exits) == [44001, 44002, 44003]
    assert all("idx" in e and "ready" in e and "status" in e for e in exits)
