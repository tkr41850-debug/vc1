from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
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
    root = Path("/tmp/codex-probe")
    home = root / "codex-home"
    workspace = root / "workspace"
    home.mkdir(parents=True, exist_ok=True)
    workspace.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(
        f'model = "{MODEL}"\n'
        'model_provider = "zen-proxy"\n'
        "\n"
        "[model_providers.zen-proxy]\n"
        'name = "zen-proxy"\n'
        f'base_url = "{BASE_URL}"\n'
        'wire_api = "responses"\n'
    )

    with open(os.getenv("PROXY_LOG", "/tmp/codex-probe-proxy.log"), "ab") as proxy_log:
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
            env = dict(
                os.environ,
                CODEX_HOME=str(home),
                OPENAI_API_KEY="dummy",
            )
            completed = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--skip-git-repo-check",
                    "Reply with exactly: codex-ok.",
                ],
                cwd=str(workspace),
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            print(completed.stdout[-2000:])
            if completed.returncode != 0:
                print(completed.stderr[-2000:], file=sys.stderr)
                return 1
            return 0
        except Exception as exc:
            print(f"probe failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        finally:
            proxy.terminate()
            proxy.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
