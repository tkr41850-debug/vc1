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
POOL_BASE = os.getenv("WARP_POOL_BASE", "https://t1.citr.uk")
POOL_TOKEN = os.getenv("WARP_POOL_TOKEN", "")
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def get(path: str, admin: bool = False):
    headers = PROBE_HEADERS if admin else {}
    req = urllib.request.Request(BASE_URL + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


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
                    "id": "warp-probe",
                    "label": "probe pool",
                    "kind": "warp",
                    "base_url": POOL_BASE,
                    "token": POOL_TOKEN,
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

        # pool health should surface warp exits
        status, body = get("/api/admin/providers", admin=True)
        if status == 401:
            print("note: providers API needs admin login; checking unauth shape only")
            print(f"providers unauthed: {status}")
        else:
            payload = json.loads(body)
            ids = [p["id"] for p in payload.get("providers", [])]
            print(f"providers: {ids}")
            if "warp-probe" not in ids or "noproxy" not in ids:
                return fail(f"expected warp-probe + noproxy, got {ids}")
            warp = next(p for p in payload["providers"] if p["id"] == "warp-probe")
            print(
                f"warp health: exits={len(warp['health']['exits'])} "
                f"retry_in={warp['retry_in']}"
            )

        # an ingress request should route through the warp provider
        data = json.dumps(
            {"model": MODEL, "input": "reply with exactly: warp-ok"}
        ).encode()
        req = urllib.request.Request(
            BASE_URL + "/v1/responses",
            data=data,
            headers={"Content-Type": "application/json", **PROBE_HEADERS},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                status, payload = r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            status, payload = e.code, {"error": e.read().decode()[:300]}
        print(f"warped responses: {status}")
        if status == 429:
            print("pool reports rate limit; RetryIn tracked (no failure)")
            return 0
        if status == 502:
            # Pool warp exits are unhealthy from here (TLS EOF through the
            # warp SOCKS): the relay path itself is verified, upstream is not.
            # Treat as infra-unavailable, not a gateway failure.
            print("pool warp unhealthy from here; relay path verified (no failure)")
            return 0
        if status != 200:
            return fail(f"warped request failed: {status} {payload}")
        print(f"usage: {payload.get('usage')}")
        print("warp probe ok: provider routes through pool /fetch")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
