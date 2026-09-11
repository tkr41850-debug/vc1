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


def test_sync_grows_table_for_new_pool(tmp_path):
    import httpx

    table = BucketTable(num_buckets=16, num_slots=1)
    registry = _registry_with_warp(tmp_path)
    egress = ProviderEgress(DirectEgress(httpx.AsyncClient()), registry=registry)
    assert egress.sync_bucket_slots(table) is True
    assert table.num_slots == 8


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
    tc.app.state.egress = ProviderEgress(
        tc.app.state.egress, registry=ProviderRegistry(data_dir=tmp_path)
    )
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
    assert table.num_slots == 8
