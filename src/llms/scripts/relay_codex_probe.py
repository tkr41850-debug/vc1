"""Codex redirect leg via the main local gateway (relay mode).

The main gateway (`just up`, port 8789) already runs with UPSTREAM_RELAY_*
from .env, so this just needs a local client key and a workspace — no
scratch gateway. The CLI does the real work through the relay's LB.

The local allowlisted key lives in /tmp/battery_local_sk (see the
battery-key-files memory); KEY_FILE overrides.

Usage:
  uv run python scripts/relay_codex_probe.py
"""

from __future__ import annotations

import os
import subprocess
import sys

PORT = int(os.getenv("ZEN_GATEWAY_PORT", "8789"))
BASE = f"http://127.0.0.1:{PORT}"
CODEX_BIN = os.getenv("CODEX_BIN", "/home/uqmm/.nvm/versions/node/v24.15.0/bin/codex")
SPARK = os.getenv("PROBE_MODEL", "glm-5.3-flash-free")
LOG_PATH = os.getenv("PROBE_LOG", "/tmp/llms.log")


def main() -> int:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from battery import battery_dirs, codex_env

    key_file = os.getenv("KEY_FILE", "/tmp/battery_local_sk")
    try:
        with open(key_file) as f:
            secret = f.read().strip().split()[0]
    except OSError as exc:
        print(f"key file {key_file} unreadable: {exc}", file=sys.stderr)
        return 1
    home, ws = battery_dirs("cx-relay-redir")
    env = codex_env(home, BASE, "ak-relay-cx01", secret, SPARK)
    try:
        c = subprocess.run(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Run shell command printf REDIRECT-OK > redir.txt "
                    "via the shell tool, then reply with exactly DONE-REDIR-1."
                ),
            ],
            cwd=str(ws),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        target = ws / "redir.txt"
        blob = target.read_bytes() if target.exists() else b"<missing>"
        print(f"TIMEOUT: redir.txt={blob!r}")
        print((exc.stderr or "")[-2000:])
        return 2
    print(f"returncode={c.returncode}")
    print((c.stdout or "")[-2000:])
    target = ws / "redir.txt"
    blob = target.read_bytes() if target.exists() else b"<missing>"
    print(f"redir.txt={blob!r}")
    if c.returncode != 0:
        print((c.stderr or "")[-2000:])
        return 1
    if blob != b"REDIRECT-OK" or "DONE-REDIR-1" not in (c.stdout or ""):
        return 1
    print("RELAY CODEX REDIRECT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
