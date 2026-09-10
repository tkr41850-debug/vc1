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
PROBE_KEY = os.getenv("PROBE_KEY", "ak-probe")
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def get(path: str, follow: bool = True):
    handlers = [] if follow else [NoRedirect()]
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(BASE_URL + path, timeout=10) as r:
            headers = {k.lower(): v for k, v in dict(r.headers).items()}
            return r.status, r.read().decode(), headers
    except urllib.error.HTTPError as e:
        headers = {k.lower(): v for k, v in dict(e.headers).items()}
        return e.code, e.read().decode(), headers


def post(path: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def main() -> int:
    if not DATA_DIR:
        return fail("PROBE_DATA_DIR must point at the gateway's data dir")
    keys_path = os.path.join(DATA_DIR, "keys.yaml")
    with open(keys_path, "w") as f:
        f.write(f"- key: {PROBE_KEY}\n  label: probe\n  enabled: true\n")
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
                status, _, _ = get("/healthz")
                if status == 200:
                    break
            except Exception:
                time.sleep(1)
        else:
            return fail("proxy did not become healthy")

        # 1. keyless ingress -> 401
        status, body = post("/v1/responses", {"model": MODEL, "input": "hi"})
        print(f"keyless: {status} {body[:120]}")
        if status != 401:
            return fail(f"keyless should 401, got {status}")

        # 2. unknown key -> 401
        status, body = post("/ak-nope/v1/responses", {"model": MODEL, "input": "hi"})
        print(f"unknown key: {status} {body[:120]}")
        if status != 401:
            return fail(f"unknown key should 401, got {status}")

        # 3. seeded key -> 200, token usage recorded
        status, body = post(
            f"/{PROBE_KEY}/v1/responses", {"model": MODEL, "input": "hi"}
        )
        print(f"keyed: {status} {body[:200]}")
        if status != 200:
            return fail(f"keyed request should 200, got {status}: {body[:500]}")
        payload = json.loads(body)
        usage = payload.get("usage", {})
        print(f"usage: in={usage.get('input_tokens')} out={usage.get('output_tokens')}")
        if usage.get("input_tokens") is None:
            return fail("expected token usage in response")

        # 4. keyed /v1/models -> 200
        status, _, _ = get(f"/{PROBE_KEY}/v1/models")
        print(f"keyed models: {status}")
        if status != 200:
            return fail(f"keyed models should 200, got {status}")

        # 5. unauthed UI -> 302 to login; unauthed admin API -> 401
        status, _, headers = get("/", follow=False)
        loc = headers.get("location", "")
        print(f"ui unauthed: {status} location={loc}")
        if status != 302 or loc != "/api/admin/login":
            return fail(f"ui should 302 to login, got {status} {loc}")
        status, body, _ = get("/api/admin/keys")
        print(f"admin unauthed: {status} {body[:120]}")
        if status != 401:
            return fail(f"admin api should 401, got {status}")

        print("admin probe ok: keyless/unknown 401, keyed 200 + usage, / -> 302 login")
        return 0
    finally:
        proc.terminate()
        proc.wait(timeout=15)


if __name__ == "__main__":
    raise SystemExit(main())
