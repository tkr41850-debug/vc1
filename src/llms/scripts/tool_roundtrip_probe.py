from __future__ import annotations

import os

import httpx

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def main() -> int:
    try:
        with (
            probe_dirs("tool-probe"),
            running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/tool-probe-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/tool-probe-data"),
            ),
        ):
            body = {
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": "run it"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "fc-repro1",
                                "name": "bash",
                                "input": {"command": "echo hi"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "fc-repro1",
                                "content": "hi",
                            }
                        ],
                    },
                    {"role": "user", "content": "reply with exactly: tools-ok"},
                ],
                "tools": [
                    {
                        "name": "bash",
                        "description": "run",
                        "input_schema": {"type": "object"},
                    }
                ],
                "max_tokens": 512,
            }
            r = httpx.post(f"{BASE_URL}/ak-probe/v1/messages", json=body, timeout=120.0)
            print(f"status={r.status_code} stop={r.json().get('stop_reason')}")
            if r.status_code != 200:
                return fail(f"tool roundtrip failed: {r.text[:500]}")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
