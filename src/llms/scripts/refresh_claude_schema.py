"""Regenerate tests/fixtures/claude-schema.jsonl (zero quota).

Runs the installed claude CLI against a local stub server and saves the
POST /v1/messages request body — which carries the CLI's live tool
declaration — as the hermetic fixture that test_claude_capture.py and
test_compat_table.py pin the claude rename rows against.

The stub answers a minimal end_turn so the CLI exits 0; nothing leaves
the machine (no API key, no quota). Re-run when the CLI version changes
the declared toolset — the tests assert the count, so a drift fails
LOUDLY here instead of silently testing a stale declaration.

Usage:
    python scripts/refresh_claude_schema.py
"""

from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "claude-schema.jsonl"
)

CAPTURED: dict = {}


class _Stub(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _reply(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if isinstance(body, dict) and body.get("tools") and "tools" not in CAPTURED:
            CAPTURED["tools"] = body["tools"]
            CAPTURED["model"] = body.get("model")
        self._reply(
            {
                "id": "msg_stub",
                "type": "message",
                "role": "assistant",
                "model": "stub",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    def do_GET(self) -> None:
        self._reply({"data": []})


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        env = {
            "PATH": "/usr/bin:/bin:/home/uqmm/.local/bin",
            "HOME": "/tmp/claude-schema-refresh",
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
            "ANTHROPIC_API_KEY": "stub-key",
            "ANTHROPIC_AUTH_TOKEN": "",
        }
        Path(env["HOME"]).mkdir(exist_ok=True)
        proc = subprocess.run(
            ["claude", "-p", "Reply with exactly: hi."],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        print(proc.stdout[-500:])
        if not CAPTURED.get("tools"):
            print(f"no tools captured (rc={proc.returncode}): {proc.stderr[-1500:]}")
            return 1
        tools = CAPTURED["tools"]
        print(f"captured {len(tools)} tools, model={CAPTURED.get('model')}")
        FIXTURE.write_text(
            json.dumps({"body": {"tools": tools, "model": CAPTURED.get("model")}})
            + "\n"
        )
        print(f"wrote {FIXTURE}")
    finally:
        server.shutdown()
        thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
