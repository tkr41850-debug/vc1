from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


@contextmanager
def running_proxy(port: int, log_path: str | None = None):
    from llms.probe.health import wait_for_health

    with open(log_path or os.devnull, "ab") as proxy_log:
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "uvicorn",
                "llms.proxy.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd="/home/uqmm/vc1/src/llms",
            stdout=proxy_log,
            stderr=subprocess.STDOUT,
        )
        try:
            if not wait_for_health(port):
                raise RuntimeError("proxy did not become healthy")
            yield proc
        finally:
            proc.terminate()
            proc.wait(timeout=15)
