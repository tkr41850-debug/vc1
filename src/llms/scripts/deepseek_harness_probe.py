from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = os.getenv("PROBE_BASE_URL", f"http://127.0.0.1:{PORT}/v1")
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def wait_for_health(port: int, timeout_s: int = 30) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/healthz", timeout=2
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(1)
    return False


def main() -> int:
    from openai import OpenAI

    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "proxy.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd="/home/uqmm/vc1/src/llms",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_health(PORT):
            print("probe: proxy did not become healthy", file=sys.stderr)
            return 1
        client = OpenAI(api_key="test-key", base_url=BASE_URL)
        response = client.responses.create(
            model=MODEL,
            input="reply with exactly: harness-ok",
            max_output_tokens=512,
        )
        print(
            json.dumps(
                {
                    "id": response.id,
                    "model": response.model,
                    "status": response.status,
                    "output_text": response.output_text,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:
        print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        proc.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
