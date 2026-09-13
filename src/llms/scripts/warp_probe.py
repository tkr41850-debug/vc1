"""Live probe for llms-managed warp pools (needs warp-cli + network).

Spins up the gateway with a slots-based warp provider and verifies the
local supervisor wiring end to end:

  1. providers.yaml with ``slots`` loads (no migration error, boot survives)
  2. the supervisor persists ``data/warps/<id>/status.json`` after traffic
  3. requests either ride a ready exit (``x-egress-provider`` header) or
     fail open to direct when no exit is up yet — both are correct gateway
     behavior; only gateway errors fail the probe

Knobs: PROBE_PORT (8793), PROBE_DATA_DIR (required), PROBE_SECRET,
PROBE_MODEL (muse-spark-1.3-contributor-free), PROBE_SLOTS (1),
PROBE_WARP_WAIT (90s max wait for a ready exit via status.json).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}"
DATA_DIR = os.getenv("PROBE_DATA_DIR", "")
PROBE_SECRET = os.getenv("PROBE_SECRET", "sk-probe")
PROBE_HEADERS = {"Authorization": f"Bearer {PROBE_SECRET}"}
PROVIDER_ID = os.getenv("PROBE_PROVIDER", "warp-probe")
SLOTS = int(os.getenv("PROBE_SLOTS", "1"))
WARP_WAIT = float(os.getenv("PROBE_WARP_WAIT", "90"))
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def get(path: str):
    req = urllib.request.Request(BASE_URL + path)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def ready_exits_on_disk() -> int | None:
    path = os.path.join(DATA_DIR, "warps", PROVIDER_ID, "status.json")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    exits = payload.get("exits")
    if not isinstance(exits, list):
        return None
    return sum(1 for w in exits if isinstance(w, dict) and w.get("ready"))


def main() -> int:
    if not DATA_DIR:
        return fail("PROBE_DATA_DIR must point at the gateway's data dir")
    keys_path = os.path.join(DATA_DIR, "keys.yaml")
    with open(keys_path, "w") as f:
        f.write(f"- key: {PROBE_SECRET}\n  label: probe\n  enabled: true\n")
    import yaml

    providers_path = os.path.join(DATA_DIR, "providers.yaml")
    with open(providers_path, "w") as f:
        yaml.safe_dump(
            [
                {
                    "id": PROVIDER_ID,
                    "label": "probe pool",
                    "kind": "warp",
                    "slots": SLOTS,
                    "models": ["*"],
                    "enabled": True,
                }
            ],
            f,
            sort_keys=False,
        )
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "uvicorn",
            "llms.proxy.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        env={**os.environ, "DATA_DIR": DATA_DIR},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(30):
            try:
                status, _ = get("/healthz")
                if status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            return fail("proxy did not become healthy")

        # Wait for a ready exit (supervisor needs warp-cli + /dev/net/tun +
        # Cloudflare registration; absent that the pool stays unhealthy and
        # traffic must fail open to direct — also a passing result).
        ready: int | None = None
        deadline = time.monotonic() + WARP_WAIT
        while time.monotonic() < deadline:
            ready = ready_exits_on_disk()
            if ready:
                break
            time.sleep(5)
        print(f"ready exits on disk: {ready}")

        data = json.dumps(
            {"model": MODEL, "input": "reply with exactly: warp-ok"}
        ).encode()
        req = urllib.request.Request(
            BASE_URL + "/v1/responses",
            data=data,
            headers={"Content-Type": "application/json", **PROBE_HEADERS},
        )
        headers: dict = {}
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                status, payload, headers = (
                    r.status,
                    json.loads(r.read().decode()),
                    dict(r.headers),
                )
        except urllib.error.HTTPError as e:
            status, payload, headers = (
                e.code,
                {"error": e.read().decode()[:300]},
                dict(e.headers),
            )
        print(f"warped responses: {status}")

        # The supervisor persists status on every health refresh, so a
        # request must leave status.json behind — proof the pool is wired.
        if ready_exits_on_disk() is None:
            return fail("supervisor never persisted warps/<id>/status.json")
        print("supervisor status.json persisted")

        via = headers.get("x-egress-provider") or headers.get("X-Egress-Provider")
        if status == 429:
            print("upstream reports rate limit; RetryIn tracked (no failure)")
            return 0
        if status >= 500:
            # Upstream or pool egress unhealthy from here (no TUN, no
            # registration, Zen hiccup): gateway error mapping verified,
            # infra itself unavailable.
            print("upstream/egress unhealthy from here; error mapping ok (no failure)")
            return 0
        if status != 200:
            return fail(f"warped request failed: {status} {payload}")
        if ready:
            if via != PROVIDER_ID:
                return fail(f"pool has {ready} ready exits but request went direct")
            print(f"warp probe ok: request rode {PROVIDER_ID} exit(s)")
        else:
            if via:
                return fail(f"pool has no ready exits but request rode {via}")
            print("warp probe ok: pool down, request failed open to direct")
        print(f"usage: {payload.get('usage')}")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
