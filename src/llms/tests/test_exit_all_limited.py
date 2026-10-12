"""RED-7 (skeptic wart): all-exits-limited resolve() returns a warp
tuple whose spread is all holes — pick_port returns None and the
pipeline fail-opens to direct via a RuntimeError catch.

Trace: spread = [ready not in limited] or ready. With every servable
exit limited, the filter empties and `or ready` restores the FULL
limited spread — set_excluded_ports then holes every position, and
pick_port walks the ring finding no live port -> None -> client_for
raises -> pipeline catches and fails open to direct.

Fail-open is the correct outcome, but the path is a wasted warp
attempt (log line + exception + direct retry) on EVERY request for the
whole cooldown, where N singles would least-wait-fallback to the same
direct outcome without the exception round-trip. Worse: warp_idx
computes as None and the recent[] entry lands on the warp provider
with warp_idx None, attributing a direct-served request to warp.

Minimal fix: when every servable exit is limited, resolve() should
still return the provider (least-wait parity) but the spread must keep
ONE live port (least-wait exit serves, rest holed) instead of holing
all and failing open. That matches N singles where the least-wait
member serves while its siblings sit out.

Per TDD: watch fail, then fix minimal.
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


def _mk(tmp_path):
    from llms.proxy.config import Settings
    from llms.proxy.providers import Provider, ProviderRegistry

    settings = Settings(data_dir=str(tmp_path))
    registry = ProviderRegistry(data_dir=str(tmp_path), settings=settings)
    registry.save([Provider(id="m", kind="warp", models=["gpt-*"], exits=3)])
    registry.runtime("m").boot_epoch = 1.0
    registry.runtime("m").health = _health3()
    return registry


def test_all_limited_keeps_one_live_port_no_fail_open(tmp_path):
    """All exits limited: spread serves via one live port, no exception."""
    from llms.proxy.egress import ProviderEgress

    reg = _mk(tmp_path)
    rt = reg.runtime("m")
    for i in range(3):
        rt.note_exit_ratelimited(i, 120.0, f"exit {i}: limited")
    egress = ProviderEgress(None, registry=reg)
    pid, kind, w = egress.resolve("gpt-5", bucket=0)
    assert (pid, kind) == ("m", "warp"), (pid, kind)
    # The spread must serve (least-wait parity with N singles), not hole
    # every position and force a fail-open exception round-trip.
    port = w.pick_port(0)
    assert port is not None, w.ready_ports()
    live = [p for p in w.ready_ports() if p is not None]
    assert len(live) >= 1, w.ready_ports()
