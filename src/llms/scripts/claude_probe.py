from __future__ import annotations

import os
import subprocess

from llms.probe.dirs import probe_dirs
from llms.probe.guards import require_messages_live
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = os.getenv("PROBE_MODEL", "claude-haiku-4-5")


def main() -> int:
    require_messages_live()
    try:
        with (
            probe_dirs("claude-probe") as (_home, workspace),
            running_proxy(PORT, os.getenv("PROXY_LOG", "/tmp/claude-probe-proxy.log")),
        ):
            env = dict(
                os.environ,
                ANTHROPIC_BASE_URL=BASE_URL,
                ANTHROPIC_API_KEY="dummy",
            )
            completed = subprocess.run(
                [
                    "claude",
                    "-p",
                    "Reply with exactly: claude-ok.",
                    "--model",
                    MODEL,
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
                return fail(completed.stderr[-2000:])
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
