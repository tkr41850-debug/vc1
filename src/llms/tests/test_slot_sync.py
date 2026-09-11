from __future__ import annotations

from llms.proxy.buckets import BucketTable
from llms.proxy.egress import DirectEgress, ProviderEgress, WarpPoolEgress
from llms.proxy.providers import Provider, ProviderRegistry


def test_table_resize_reshuffles():
    table = BucketTable(num_buckets=8, num_slots=1)
    assert table.slot_for(5) == 0
    assert table.set_num_slots(4) is True
    assert table.slot_for(5) == 1
    assert table.set_num_slots(4) is False
    assert table.set_num_slots(0) is True
    assert table.num_slots == 1


def _registry_with_warp(tmp_path, **overrides) -> ProviderRegistry:
    registry = ProviderRegistry(data_dir=tmp_path)
    params = {
        "id": "pool1",
        "label": "pool",
        "kind": "warp",
        "base_url": "http://pool:8080",
        "models": ["muse-*"],
        "enabled": True,
    }
    params.update(overrides)
    registry.save([Provider(**params)])
    return registry


def test_sync_new_pool_starts_at_zero_slots(tmp_path):
    import httpx

    table = BucketTable(num_buckets=16, num_slots=1)
    registry = _registry_with_warp(tmp_path)
    egress = ProviderEgress(DirectEgress(httpx.AsyncClient()), registry=registry)
    assert egress.sync_bucket_slots(table) is False
    assert table.num_slots == 1
    assert egress.resolve("muse-spark")[1] == "warp"


def test_sync_grows_as_exits_come_up(tmp_path):
    import httpx

    table = BucketTable(num_buckets=16, num_slots=1)
    registry = _registry_with_warp(tmp_path)
    egress = ProviderEgress(DirectEgress(httpx.AsyncClient()), registry=registry)
    egress.resolve("muse-spark")
    egress._warp["pool1"].set_num_slots(3)
    assert egress.sync_bucket_slots(table) is True
    assert table.num_slots == 3


def test_resolve_skips_known_empty_pool(tmp_path):
    import time

    import httpx

    from llms.proxy.providers import ProviderHealth

    registry = _registry_with_warp(tmp_path)
    egress = ProviderEgress(DirectEgress(httpx.AsyncClient()), registry=registry)
    assert egress.resolve("muse-spark")[1] == "warp"
    rt = registry.runtime("pool1")
    rt.health = ProviderHealth(fetched_at=time.monotonic(), exits=[])
    assert egress.resolve("muse-spark")[1] == "noproxy"


def test_sync_tracks_ready_exits_and_disable(tmp_path):
    import httpx

    table = BucketTable(num_buckets=16, num_slots=1)
    registry = _registry_with_warp(tmp_path)
    egress = ProviderEgress(DirectEgress(httpx.AsyncClient()), registry=registry)
    egress._warp["pool1"] = WarpPoolEgress("http://pool:8080", num_slots=3)
    assert egress.sync_bucket_slots(table) is True
    assert table.num_slots == 3
    registry.save(
        [Provider(id="pool1", kind="warp", base_url="http://pool:8080", enabled=False)]
    )
    assert egress.sync_bucket_slots(table) is True
    assert table.num_slots == 1


def test_sync_direct_only_stays_single_slot(tmp_path):
    import httpx

    table = BucketTable(num_buckets=16, num_slots=1)
    registry = ProviderRegistry(data_dir=tmp_path)
    egress = ProviderEgress(DirectEgress(httpx.AsyncClient()), registry=registry)
    assert egress.sync_bucket_slots(table) is False
    assert table.num_slots == 1


def test_create_provider_resizes_table(admin_client, tmp_path):
    from llms.proxy.egress import ProviderEgress
    from llms.proxy.providers import ProviderRegistry

    tc, _ = admin_client
    table = tc.app.state.bucket_table
    egress = ProviderEgress(
        tc.app.state.egress, registry=ProviderRegistry(data_dir=tmp_path)
    )
    tc.app.state.egress = egress
    assert table.num_slots == 1
    r = tc.post(
        "/api/admin/providers",
        json={
            "id": "pool1",
            "kind": "warp",
            "base_url": "http://pool:8080",
            "models": ["muse-*"],
        },
        headers={"Authorization": "Bearer sk-test"},
    )
    assert r.status_code == 201
    assert (tmp_path / "warps" / "pool1").is_dir()
    assert table.num_slots == 1
    egress.resolve("muse-spark")
    egress._warp["pool1"].set_num_slots(4)
    r = tc.put("/api/admin/providers/pool1", json={"label": "p1"})
    assert r.status_code == 200
    assert table.num_slots == 4
    r = tc.delete("/api/admin/providers/pool1")
    assert r.status_code == 200
    assert not (tmp_path / "warps" / "pool1").exists()
    assert table.num_slots == 1


def test_warp_status_persists_ready_exits(tmp_path):
    from llms.proxy.providers import ProviderHealth, ProviderRegistry, WarpExit

    registry = ProviderRegistry(data_dir=tmp_path)
    assert registry.ready_exits("pool1") is None
    registry.save_warp_status(
        "pool1",
        ProviderHealth(
            active=1,
            exits=[WarpExit(idx=1, ready=True), WarpExit(idx=2, ready=False)],
            fetched_at=1.0,
        ),
    )
    assert registry.ready_exits("pool1") == 1
    assert (tmp_path / "warps" / "pool1" / "status.json").exists()


def test_resolve_seeds_slots_from_disk(tmp_path):
    import httpx

    from llms.proxy.providers import ProviderHealth, WarpExit

    registry = _registry_with_warp(tmp_path)
    registry.save_warp_status(
        "pool1",
        ProviderHealth(exits=[WarpExit(idx=1, ready=True)], fetched_at=1.0),
    )
    egress = ProviderEgress(DirectEgress(httpx.AsyncClient()), registry=registry)
    pid, kind, warp = egress.resolve("muse-spark")
    assert (pid, kind) == ("pool1", "warp")
    assert warp is not None and warp.num_slots() == 1
