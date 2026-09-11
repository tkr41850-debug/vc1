from __future__ import annotations

import copy
import json
from pathlib import Path

USAGE_FILE = "usage.json"


def _blank_entry() -> dict:
    return {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "models": {},
    }


def _migrate_entry(entry: dict) -> dict:
    entry.setdefault("cached_tokens", 0)
    entry.setdefault("reasoning_tokens", 0)
    for model_entry in entry.get("models", {}).values():
        if isinstance(model_entry, dict):
            model_entry.setdefault("cached_tokens", 0)
            model_entry.setdefault("reasoning_tokens", 0)
    return entry


class UsageTracker:
    def __init__(self) -> None:
        self._keys: dict[str, dict] = {}

    def _entry(self, key: str, model: str) -> dict:
        key_entry = self._keys.setdefault(key, _blank_entry())
        model_entry = key_entry["models"].setdefault(
            model,
            {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
            },
        )
        return key_entry, model_entry

    def record(
        self,
        key: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_tokens: int | None = None,
        reasoning_tokens: int | None = None,
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
        if cached_tokens is not None:
            key_entry["cached_tokens"] += cached_tokens
            model_entry["cached_tokens"] += cached_tokens
        if reasoning_tokens is not None:
            key_entry["reasoning_tokens"] += reasoning_tokens
            model_entry["reasoning_tokens"] += reasoning_tokens

    def rekey(self, old_key: str, new_key: str) -> None:
        """Move attribution from old_key to new_key (rotation).

        Merges counters when new_key already has traffic; drops the old key
        entry so the admin list shows one live key.
        """
        if old_key == new_key or old_key not in self._keys:
            return
        old = self._keys.pop(old_key)
        if new_key not in self._keys:
            self._keys[new_key] = old
            return
        new = self._keys[new_key]
        new["requests"] += old.get("requests", 0)
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "reasoning_tokens",
        ):
            new[field] = new.get(field, 0) + old.get(field, 0)
        for model, m_old in (old.get("models") or {}).items():
            if not isinstance(m_old, dict):
                continue
            m_new = new.setdefault("models", {}).setdefault(model, _blank_entry())
            m_new.pop("models", None)
            m_new["requests"] += m_old.get("requests", 0)
            for field in (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "reasoning_tokens",
            ):
                m_new[field] = m_new.get(field, 0) + m_old.get(field, 0)

    def snapshot(self) -> dict:
        return {"keys": copy.deepcopy(self._keys)}

    def load(self, data: dict) -> None:
        if isinstance(data, dict) and isinstance(data.get("keys"), dict):
            self._keys = {
                k: _migrate_entry(copy.deepcopy(v))
                for k, v in data["keys"].items()
                if isinstance(v, dict)
            }

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
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot()))
        tmp.replace(path)


def _to_int(value) -> int | None:
    """Best-effort int coercion; malformed upstream usage never fails requests."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_usage(
    ingress: str, payload: dict
) -> tuple[int | None, int | None, int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None, None, None
    if ingress == "chat":
        in_tok = _to_int(usage.get("prompt_tokens"))
        out_tok = _to_int(usage.get("completion_tokens"))
        in_details = usage.get("prompt_tokens_details", {})
        out_details = usage.get("completion_tokens_details", {})
    elif ingress == "messages":
        return (
            _to_int(usage.get("input_tokens")),
            _to_int(usage.get("output_tokens")),
            _to_int(usage.get("cache_read_input_tokens")),
            None,
        )
    else:
        in_tok = _to_int(usage.get("input_tokens"))
        out_tok = _to_int(usage.get("output_tokens"))
        in_details = usage.get("input_tokens_details", {})
        out_details = usage.get("output_tokens_details", {})
    if not isinstance(in_details, dict):
        in_details = {}
    if not isinstance(out_details, dict):
        out_details = {}
    return (
        in_tok,
        out_tok,
        _to_int(in_details.get("cached_tokens")),
        _to_int(out_details.get("reasoning_tokens")),
    )
