from __future__ import annotations

import os
import subprocess

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")
PROBE_SECRET = os.getenv("PROBE_SECRET", "sk-probe")


def main() -> int:
    try:
        with (
            probe_dirs("claude-probe") as (_home, workspace),
            running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/claude-probe-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/claude-probe-data"),
            ),
        ):
            env = dict(
                os.environ,
                ANTHROPIC_BASE_URL=BASE_URL,
                ANTHROPIC_API_KEY=PROBE_SECRET,
                ANTHROPIC_MODEL=MODEL,
            )

            def run_claude(*args: str):
                return subprocess.run(
                    ["claude", "-p", *args, "--model", MODEL, "--allowedTools", "Bash"],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )

            first = run_claude("Reply with exactly: turn-one.")
            print(first.stdout[-500:])
            if first.returncode != 0:
                return fail(first.stderr[-2000:])
            second = run_claude("--continue", "Reply with exactly: turn-two.")
            print(second.stdout[-500:])
            if second.returncode != 0:
                return fail(second.stderr[-2000:])
            third = run_claude(
                "--continue",
                "Use Bash to run echo tool-ok, then reply with exactly its output.",
            )
            print(third.stdout[-500:])
            if third.returncode != 0 or "tool-ok" not in third.stdout:
                return fail(
                    f"tool turn failed: {third.returncode} {third.stderr[-2000:]}"
                )
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
