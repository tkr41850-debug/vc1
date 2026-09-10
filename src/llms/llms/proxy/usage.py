from __future__ import annotations

import copy
import json
from pathlib import Path

USAGE_FILE = "usage.json"


class UsageTracker:
    def __init__(self) -> None:
        self._keys: dict[str, dict] = {}

    def _entry(self, key: str, model: str) -> dict:
        key_entry = self._keys.setdefault(
            key,
            {"requests": 0, "input_tokens": 0, "output_tokens": 0, "models": {}},
        )
        model_entry = key_entry["models"].setdefault(
            model, {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        )
        return key_entry, model_entry

    def record(
        self,
        key: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        key_entry, model_entry = self._entry(key, model)
        key_entry["requests"] += 1
        model_entry["requests"] += 1
        if input_tokens is not None:
            key_entry["input_tokens"] += input_tokens
            model_entry["input_tokens"] += input_tokens
        if output_tokens is not None:
            key_entry["output_tokens"] += output_tokens
            model_entry["output_tokens"] += output_tokens

    def snapshot(self) -> dict:
        return {"keys": copy.deepcopy(self._keys)}

    def load(self, data: dict) -> None:
        if isinstance(data, dict) and isinstance(data.get("keys"), dict):
            self._keys = copy.deepcopy(data["keys"])

    def load_file(self, data_dir: str | Path) -> None:
        path = Path(data_dir) / USAGE_FILE
        if not path.exists():
            return
        try:
            self.load(json.loads(path.read_text()))
        except Exception:
            pass

    def save_file(self, data_dir: str | Path) -> None:
        path = Path(data_dir) / USAGE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot()))


def extract_usage(ingress: str, payload: dict) -> tuple[int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None
    if ingress == "chat":
        return usage.get("prompt_tokens"), usage.get("completion_tokens")
    return usage.get("input_tokens"), usage.get("output_tokens")
