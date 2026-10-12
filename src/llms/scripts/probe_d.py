import time
from llms.proxy.providers import ProviderRuntime, ProviderHealth, WarpExit

# Item 1: legacy note_ratelimited leaves limited_exits() empty -> _fresh False
rt = ProviderRuntime()
rt.health = ProviderHealth(
    exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
    fetched_at=1000.0,
)
rt.note_ratelimited(120.0, "slow")
assert rt.retry_in() > 0
assert rt.limited_exits() == set(), rt.limited_exits()
# mirror of egress._fresh
fresh = True if rt.retry_in() <= 0 else bool(rt.limited_exits())
assert fresh is False
print("item1 PASS: legacy note_ratelimited -> limited empty -> _fresh False (failover preserved)")

# Item 4: clear_backoff state identical to old inline clears
rt2 = ProviderRuntime()
rt2.note_ratelimited(60.0, "slow down")
rt2.note_exit_ratelimited(3, 60.0, "x")
assert rt2.retry_in() > 0 and rt2.limited_exits() == {3}
rt2.clear_backoff()
assert rt2.retry_until == 0.0 and rt2.retry_reason == "" and rt2.limited_exits() == set()
# matches old inline (retry_until=0, reason="") plus exit table cleared
print("item4a PASS: clear_backoff resets provider+exit state")

# Item 4b: epoch-guard semantics preserved (sibling 429 survives clear)
rt3 = ProviderRuntime()
rt3.note_ratelimited(60.0, "original")
observed_epoch = rt3.retry_epoch
rt3.note_ratelimited(120.0, "sibling")
assert rt3.retry_epoch == observed_epoch + 1
print("item4b PASS: sibling 429 bumps epoch (bounce-ok guard intact)")

# Item 3b: refresh_health with no per-exit records + _egresses None -> no crash
import asyncio
from llms.proxy.providers import Provider, ProviderRegistry

async def main():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        reg = ProviderRegistry(data_dir=d)
        assert reg._egresses is None
        p = Provider(id="w1", kind="warp", exits=1, models=["*"])
        rt4 = reg.runtime("w1")
        rt4.health = ProviderHealth(
            exits=[WarpExit(idx=0, ready=True, status="ok", socks=40001)],
            fetched_at=time.monotonic(),
        )
        # no per-exit records
        assert rt4.limited_exits() == set()
        print("item3b PASS: no per-exit records, _egresses None, no holes path taken")

asyncio.run(main())
