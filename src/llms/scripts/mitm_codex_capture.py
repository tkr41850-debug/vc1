"""mitmdump addon: record codex->proxy->zen flows for offline inspection.

Sits between codex and the gateway (or between gateway and zen) and dumps
full request/response bodies to a JSONL capture file, so a Codex tool-call
repro can be eyeballed frame-by-frame after the run instead of tailing
live logs.

Upstream of this proxy the bodies are plain HTTP (gateway speaks plain
http:// to zen; codex speaks plain http:// to the gateway), so no TLS
keylog or CA install is needed in either position. Do NOT point it at
https://opencode.ai directly — that leg is TLS and out of scope.

Usage (codex side — recommended: sees exactly what codex sends):
    mitmdump -p 8080 -s scripts/mitm_codex_capture.py --set capture_path=/tmp/codex-flow.jsonl
    HTTPS_PROXY=http://127.0.0.1:8080 HTTP_PROXY=http://127.0.0.1:8080 \\
        codex exec --skip-git-repo-check "Use the shell to run echo hi"

Usage (zen side — sees exactly what the gateway forwards upstream):
    # point the gateway at the proxy: ZEN_BASE_URL=http://127.0.0.1:8081
    mitmdump -p 8081 -s scripts/mitm_codex_capture.py --set capture_path=/tmp/zen-flow.jsonl

Each line: {"ts", "flow": "request"|"response", "method", "url",
"status", "headers" (auth redacted), "body" (truncated at 64k)}.
"""

from __future__ import annotations

import json
import time

from mitmproxy import ctx

BODY_LIMIT = 64 * 1024


def _redact(headers) -> dict:
    out = {}
    for k, v in headers.items():
        lk = k.lower()
        if lk == "authorization" and len(v) > 12:
            out[k] = v[:12] + "...<redacted>"
        else:
            out[k] = v
    return out


def _emit(flow_kind: str, method: str, url: str, status: int, headers: dict, body: bytes) -> None:
    path = ctx.options.capture_path or "/tmp/codex-flow.jsonl"
    try:
        text = bytes(body or b"").decode("utf-8", errors="replace")[:BODY_LIMIT]
    except Exception:
        text = "<undecodable>"
    rec = {
        "ts": time.time(),
        "flow": flow_kind,
        "method": method,
        "url": url,
        "status": status,
        "headers": _redact(headers),
        "body": text,
    }
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def request(flow) -> None:
    r = flow.request
    _emit("request", r.method, r.pretty_url, 0, dict(r.headers), r.raw_content)


def response(flow) -> None:
    r = flow.request
    resp = flow.response
    _emit(
        "response",
        r.method,
        r.pretty_url,
        resp.status_code,
        dict(resp.headers),
        resp.raw_content,
    )


def load(loader) -> None:
    loader.add_option(
        "capture_path", str, "/tmp/codex-flow.jsonl", "JSONL capture file path"
    )
