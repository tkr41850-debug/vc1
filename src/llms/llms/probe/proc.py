from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


@contextmanager
def running_proxy(port: int, log_path: str | None = None, data_dir: str | None = None):
    from llms.probe.health import wait_for_health

    if data_dir is not None:
        import yaml

        keys_path = os.path.join(data_dir, "keys.yaml")
        keys = []
        if os.path.exists(keys_path):
            with open(keys_path) as f:
                keys = yaml.safe_load(f) or []
        if not any(
            isinstance(k, dict)
            and k.get("key") == "ak-probe"
            and k.get("enabled", True)
            for k in keys
        ):
            keys.append({"key": "ak-probe", "label": "probe", "enabled": True})
            with open(keys_path, "w") as f:
                yaml.safe_dump(keys, f, sort_keys=False)

    with open(log_path or os.devnull, "ab") as proxy_log:
        env = dict(os.environ)
        if data_dir is not None:
            env["DATA_DIR"] = data_dir
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
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
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
