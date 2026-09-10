from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

KEYS_FILE = "keys.yaml"
MODELS_FILE = "models.yaml"


SECRET_PREFIX = "sk-"


class StoreError(Exception):
    """Raised when a YAML store file exists but cannot be parsed."""


def is_secret_key(value: str) -> bool:
    return value.startswith(SECRET_PREFIX)


@dataclass
class ApiKey:
    key: str
    label: str = ""
    enabled: bool = True


@dataclass
class ModelEntry:
    id: str
    label: str = ""
    enabled: bool = True


@dataclass
class Store:
    data_dir: Path
    _keys: list[ApiKey] = field(default_factory=list, init=False, repr=False)
    _models: list[ModelEntry] = field(default_factory=list, init=False, repr=False)

    def keys_path(self) -> Path:
        return self.data_dir / KEYS_FILE

    def models_path(self) -> Path:
        return self.data_dir / MODELS_FILE

    # -- keys -----------------------------------------------------------

    def load_keys(self) -> list[ApiKey]:
        path = self.keys_path()
        if not path.exists():
            return []
        try:
            raw = yaml.safe_load(path.read_text()) or []
        except yaml.YAMLError as exc:
            raise StoreError(f"unparsable {KEYS_FILE}: {exc}") from exc
        if not isinstance(raw, list):
            raise StoreError(f"unparsable {KEYS_FILE}: expected a list")
        out = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("key"):
                continue
            out.append(
                ApiKey(
                    key=str(item["key"]),
                    label=str(item.get("label", "")),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return out

    def save_keys(self, keys: list[ApiKey]) -> None:
        path = self.keys_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(
                [{"key": k.key, "label": k.label, "enabled": k.enabled} for k in keys],
                sort_keys=False,
            )
        )
        tmp.replace(path)

    def find_key(self, key: str) -> ApiKey | None:
        for k in self.load_keys():
            if k.key == key:
                return k
        return None

    def key_allowed(self, key: str) -> bool:
        found = self.find_key(key)
        return found is not None and found.enabled

    # -- models ---------------------------------------------------------

    def load_models(self) -> list[ModelEntry]:
        path = self.models_path()
        if not path.exists():
            from llms.proxy.router import FREE_MODELS

            return [ModelEntry(id=m) for m in FREE_MODELS]
        try:
            raw = yaml.safe_load(path.read_text()) or []
        except yaml.YAMLError as exc:
            raise StoreError(f"unparsable {MODELS_FILE}: {exc}") from exc
        if not isinstance(raw, list):
            raise StoreError(f"unparsable {MODELS_FILE}: expected a list")
        out = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            out.append(
                ModelEntry(
                    id=str(item["id"]),
                    label=str(item.get("label", "")),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return out

    def save_models(self, models: list[ModelEntry]) -> None:
        path = self.models_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            yaml.safe_dump(
                [{"id": m.id, "label": m.label, "enabled": m.enabled} for m in models],
                sort_keys=False,
            )
        )
        tmp.replace(path)

    def enabled_model_ids(self) -> list[str]:
        return [m.id for m in self.load_models() if m.enabled]
