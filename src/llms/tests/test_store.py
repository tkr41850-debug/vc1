from __future__ import annotations

from pathlib import Path

from llms.proxy.store import ApiKey, ModelEntry, Store


def test_missing_keys_file_reads_empty(tmp_path: Path):
    store = Store(data_dir=tmp_path)
    assert store.load_keys() == []


def test_key_roundtrip(tmp_path: Path):
    store = Store(data_dir=tmp_path)
    store.save_keys(
        [ApiKey(key="ak-team1", label="Team 1"), ApiKey(key="ak-off", enabled=False)]
    )
    keys = store.load_keys()
    assert [k.key for k in keys] == ["ak-team1", "ak-off"]
    assert keys[0].label == "Team 1"
    assert keys[0].enabled is True
    assert keys[1].enabled is False


def test_key_allowed(tmp_path: Path):
    store = Store(data_dir=tmp_path)
    store.save_keys([ApiKey(key="ak-on"), ApiKey(key="ak-off", enabled=False)])
    assert store.key_allowed("ak-on") is True
    assert store.key_allowed("ak-off") is False
    assert store.key_allowed("ak-missing") is False


def test_missing_models_file_seeds_free_models(tmp_path: Path):
    from llms.proxy.router import FREE_MODELS

    store = Store(data_dir=tmp_path)
    assert [m.id for m in store.load_models()] == list(FREE_MODELS)


def test_model_roundtrip_and_enabled_ids(tmp_path: Path):
    store = Store(data_dir=tmp_path)
    store.save_models(
        [
            ModelEntry(id="m-a", label="A"),
            ModelEntry(id="m-b", enabled=False),
        ]
    )
    models = store.load_models()
    assert [m.id for m in models] == ["m-a", "m-b"]
    assert models[0].label == "A"
    assert store.enabled_model_ids() == ["m-a"]
