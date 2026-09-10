from __future__ import annotations

import json
import logging
import uuid

logger = logging.getLogger("zen_proxy")

_configured = False


def setup_logging(level: str = "INFO") -> logging.Logger:
    global _configured
    if not _configured:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        _configured = True
    return logger


def new_trace_id() -> str:
    return uuid.uuid4().hex[:12]


def redact_headers(headers: dict) -> dict:
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk == "authorization" and isinstance(v, str) and len(v) > 12:
            out[k] = v[:12] + "...<redacted>"
        elif lk in {"x-opencode-session", "x-opencode-request"}:
            out[k] = v
        else:
            out[k] = v
    return out


def log_ingress(trace_id: str, path: str, body: dict) -> None:
    try:
        rendered = json.dumps(body)[:4000]
    except Exception:
        rendered = "<unserializable>"
    logger.info(
        "[%s] ingress %s model=%s stream=%s body=%s",
        trace_id,
        path,
        body.get("model"),
        body.get("stream"),
        rendered,
    )


def log_upstream(trace_id: str, url: str, headers: dict, body: dict) -> None:
    try:
        rendered = json.dumps(body)[:4000]
    except Exception:
        rendered = "<unserializable>"
    logger.info(
        "[%s] upstream POST %s headers=%s body=%s",
        trace_id,
        url,
        redact_headers(headers),
        rendered,
    )


def log_response(trace_id: str, status: int, payload_size: int) -> None:
    logger.info("[%s] upstream status=%s bytes=%s", trace_id, status, payload_size)
