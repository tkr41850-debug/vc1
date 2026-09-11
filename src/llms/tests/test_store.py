from __future__ import annotations

from pathlib import Path

from llms.proxy.store import ApiKey, ModelEntry, Store, StoreError, is_secret_key


def test_is_secret_key():
    assert is_secret_key("sk-abc") is True
    assert is_secret_key("ak-abc") is False
    assert is_secret_key("Bearer sk-abc") is False
    assert is_secret_key("") is False


def test_missing_keys_file_reads_empty(tmp_path: Path):
    store = Store(data_dir=tmp_path)
    assert store.load_keys() == []


def test_key_roundtrip(tmp_path: Path):
    store = Store(data_dir=tmp_path)
    store.save_keys(
        [ApiKey(key="sk-team1", label="Team 1"), ApiKey(key="sk-off", enabled=False)]
    )
    keys = store.load_keys()
    assert [k.key for k in keys] == ["sk-team1", "sk-off"]
    assert keys[0].label == "Team 1"
    assert keys[0].enabled is True
    assert keys[1].enabled is False


def test_key_allowed(tmp_path: Path):
    store = Store(data_dir=tmp_path)
    store.save_keys([ApiKey(key="sk-on"), ApiKey(key="sk-off", enabled=False)])
    assert store.key_allowed("sk-on") is True
    assert store.key_allowed("sk-off") is False
    assert store.key_allowed("sk-missing") is False


def test_corrupt_keys_file_raises_store_error(tmp_path: Path):
    import pytest

    (tmp_path / "keys.yaml").write_text("{unclosed: [bracket\n  - nope")
    with pytest.raises(StoreError):
        Store(data_dir=tmp_path).load_keys()


def test_non_list_keys_file_raises_store_error(tmp_path: Path):
    import pytest

    (tmp_path / "keys.yaml").write_text("key: sk-x\n")
    with pytest.raises(StoreError):
        Store(data_dir=tmp_path).load_keys()


def test_corrupt_models_file_raises_store_error(tmp_path: Path):
    import pytest

    (tmp_path / "models.yaml").write_text("{unclosed: [bracket\n  - nope")
    with pytest.raises(StoreError):
        Store(data_dir=tmp_path).load_models()


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
