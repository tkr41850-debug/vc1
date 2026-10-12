"""RED-3 (honest): per-exit 429 isolation on a multi-exit provider.

Verified current behavior (probe_div.py, since removed):
- 1x3 multi, note_ratelimited(120) on the PROVIDER -> resolve() still
  returns ('m','warp') for bucket 0. Rationale: single-candidate ring
  has no alternative, so fresh=[...] is empty and the least-wait
  fallback returns the limited provider anyway.
- N x singles, one 429'd -> the other two serve (failover exists).
- Real divergence needs >=2 providers serving the model: the multi's
  429 deprioritizes it against its SIBLING provider, while singles only
  lose one member.

So: multi('m', 3 exits) + single('s', 1 exit) serving the same model.
A 429 observed on ONE exit of 'm' must not push traffic to 's' — 'm'
still has 2 healthy exits. Only when ALL of m's exits are limited may
resolve() prefer 's'.

Per TDD: watch this fail first, then fix minimal.
"""
from __future__ import annotations


def _ready(n: int, base: int = 40001, ready_idx=None):
    from llms.proxy.providers import ProviderHealth, WarpExit

    ris = set(range(n)) if ready_idx is None else set(ready_idx)
    return ProviderHealth(
        exits=[
            WarpExit(
                idx=i,
                ready=(i in ris),
                status="ok" if i in ris else "down",
                socks=base + i,
            )
            for i in range(n)
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


def test_one_exit_429_does_not_yield_ring_share_to_sibling(tmp_path):
    """A 429 on 1 of 3 exits must not cede ring share to a sibling provider."""
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import Provider

    reg = _mk(
        tmp_path,
        [
            Provider(id="m", kind="warp", models=["gpt-*"], exits=3),
            Provider(id="s", kind="warp", models=["gpt-*"], exits=1),
        ],
    )
    reg.runtime("m").health = _ready(3, base=40001)
    reg.runtime("s").health = _ready(1, base=41001)
    egress = ProviderEgress(None, registry=reg)

    before = {egress.resolve("gpt-5", bucket=b)[0] for b in range(60)}
    assert before == {"m", "s"}, before

    # One exit's 429, recorded against the exit that served it. The
    # provider still has 2 healthy exits: ring share must be unchanged.
    reg.runtime("m").note_exit_ratelimited(1, 120.0, "exit 1: Too many requests")
    # The aggregate still records the limit (self-heal/shed/hint readers),
    # but the provider keeps its ring share via healthy-exit steering.
    assert reg.runtime("m").retry_in() > 0
    assert reg.runtime("m").limited_exits() == {1}
    after = {egress.resolve("gpt-5", bucket=b)[0] for b in range(60)}
    assert after == before, f"ring share ceded after one exit 429: {after}"
    # ... while the limited exit itself is steered around.
    _, _, w = egress.resolve("gpt-5", bucket=0)
    assert sorted(w.ready_ports()) == [40001, 40003], w.ready_ports()
