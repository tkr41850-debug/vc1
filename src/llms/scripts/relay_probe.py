"""Live relay probe: local gateway -> upstream gateway as opencode traffic.

Spins a scratch local gateway (running_proxy convention) with
UPSTREAM_RELAY_URL/UPSTREAM_RELAY_KEY in env, sends one cheap
/v1/responses leg, and asserts:
  1. downstream 200 with non-empty text, and
  2. the scratch proxy log shows "upstream relay via <url>" (the local
     hop went to the relay, not zen).

Usage (values from server env / key file — never the repo):
  UPSTREAM_RELAY_URL=https://<upstream-host> \
  RELAY_KEY_FILE=/tmp/upstream_sk \
  uv run python scripts/relay_probe.py

RELAY_KEY_FILE defaults to /tmp/upstream_sk (same convention as
BATTERY_KEY_FILE); UPSTREAM_RELAY_KEY may be set directly instead.
The key is never printed.

Probes-vs-tests split: this needs two live gateways, so it lives in
scripts/, not tests/.
"""

from __future__ import annotations

import os
import sys

import httpx

PORT = int(os.getenv("PROBE_PORT", "8794"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    from llms.probe.dirs import probe_dirs
    from llms.probe.proc import running_proxy

    relay_url = os.getenv("UPSTREAM_RELAY_URL", "").strip()
    if not relay_url:
        return fail("UPSTREAM_RELAY_URL is not set")
    relay_key = os.getenv("UPSTREAM_RELAY_KEY", "").strip()
    if not relay_key:
        key_file = os.getenv("RELAY_KEY_FILE", "/tmp/upstream_sk")
        try:
            with open(key_file) as f:
                relay_key = f.read().strip().split()[0]
        except OSError as exc:
            return fail(f"no relay key: UPSTREAM_RELAY_KEY unset: {exc}")
    # The scratch proxy inherits this process's env (see
    # llms.probe.proc.running_proxy), so exporting here configures it.
    os.environ["UPSTREAM_RELAY_URL"] = relay_url
    os.environ["UPSTREAM_RELAY_KEY"] = relay_key

    log_path = os.getenv("PROBE_LOG", "/tmp/relay-probe-proxy.log")
    data_dir = os.getenv("PROBE_DATA_DIR", "/tmp/relay-probe-data")
    try:
        with (
            probe_dirs("relay-probe"),
            running_proxy(
                PORT,
                log_path,
                data_dir=data_dir,
                probe_secret=os.getenv("PROBE_SECRET"),
            ) as (_, secret),
        ):
            r = httpx.post(
                f"{BASE_URL}/v1/responses",
                json={"model": MODEL, "input": "reply with exactly: relay-ok"},
                headers={"Authorization": f"Bearer {secret}"},
                timeout=180.0,
            )
            print(f"status={r.status_code}")
            if r.status_code != 200:
                return fail(f"relay leg failed: {r.text[:500]}")
            try:
                text = r.json()["output"][0]["content"][0]["text"]
            except (KeyError, IndexError, ValueError):
                return fail(f"unexpected downstream shape: {r.text[:500]}")
            print(f"downstream text: {text[:120]!r}")
            if not text.strip():
                return fail("downstream text is empty")
        with open(log_path) as f:
            log = f.read()
        marker = f"upstream relay via {relay_url.rstrip('/')}/responses"
        if marker not in log:
            return fail(
                "scratch proxy never logged the relay hop "
                f"(missing {marker!r}; tail: {log[-800:]})"
            )
        print(f"hop verified: {marker}")
        print("RELAY PROBE PASS")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
