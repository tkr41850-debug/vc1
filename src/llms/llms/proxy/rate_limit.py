from __future__ import annotations

import re

RATE_LIMIT_PATTERN = re.compile(
    r"rate.?limit|too many|quota|freeusagelimit|FreeUsageLimit", re.IGNORECASE
)


def _payload_text(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            parts = [
                str(v)
                for v in (
                    error.get("type"),
                    error.get("message"),
                    error.get("code"),
                    # FastAPI-style error bodies carry the signal here
                    # ({"detail": "Rate limit exceeded"}) rather than under
                    # "error"; without it classify() misses them as "ok".
                    error.get("detail", payload.get("detail")),
                )
                if v
            ]
            if parts:
                return " ".join(parts)
            return str(error)
        return str(error)
    return str(payload)


def classify(
    status: int, payload=None, headers: dict | None = None
) -> tuple[str, float | None]:
    retry_after: float | None = None
    if headers:
        for key, value in headers.items():
            if key.lower() == "retry-after":
                try:
                    retry_after = float(str(value).split(",")[0].strip())
                except ValueError:
                    retry_after = None
                break
    if status == 429 or RATE_LIMIT_PATTERN.search(_payload_text(payload)):
        return "ratelimited", retry_after
    if status >= 500:
        return "error", None
    return "ok", None
