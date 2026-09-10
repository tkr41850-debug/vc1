from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
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
    from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

    root = Path("/tmp/dsh-probe")
    home = root / "home"
    workspace = root / "workspace"
    home.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)

    with open(os.getenv("PROXY_LOG", "/tmp/dsh-probe-proxy.log"), "ab") as proxy_log:
        proxy = subprocess.Popen(
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
            stdout=proxy_log,
            stderr=subprocess.STDOUT,
        )
        try:
            if not wait_for_health(PORT):
                print("probe: proxy did not become healthy", file=sys.stderr)
                return 1
            config = DeepSeekHarnessConfig(
                provider="deepseek-official",
                model=MODEL,
                max_tokens=512,
                cwd=str(workspace),
                dsh_home=str(home),
                profile="sdk-minimal",
                base_url=BASE_URL,
                api_key="dummy",
            )
            with DeepSeekHarness(config) as harness:
                result = harness.run(
                    "Reply with exactly: dsh-ok.",
                    session_id=f"probe-{uuid.uuid4().hex[:8]}",
                )
            print(f"finish_reason={result.finish_reason}")
            print(f"final_response={result.final_response!r}")
            if result.finish_reason == "error":
                print(f"notifications={result.notifications!r}")
                print(f"events={result.events!r}")
            return 0
        except Exception as exc:
            print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        finally:
            proxy.terminate()
            proxy.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
