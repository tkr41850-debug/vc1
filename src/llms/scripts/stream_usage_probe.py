from __future__ import annotations

import json
import os

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def main() -> int:
    """Exercise streamed usage end-to-end through a real harness.

    Sends one streamed chat turn via raw SSE (the same wire shape a
    streaming harness uses), then stops the gateway — shutdown flushes
    usage.json — and asserts the probe secret shows token counts rather
    than the request-only fallback.
    """
    import httpx

    data_dir = os.getenv("PROBE_DATA_DIR", "/tmp/stream-usage-data")
    try:
        with (
            probe_dirs("stream-usage-probe"),
            running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/stream-usage-proxy.log"),
                data_dir=data_dir,
                probe_secret=os.getenv("PROBE_SECRET"),
            ) as (proc, secret),
        ):
            body = {
                "model": "mimo-v2.5-free",
                "messages": [
                    {"role": "user", "content": "Reply with exactly: stream-ok."}
                ],
                "stream": True,
                "max_tokens": 64,
            }
            chunks = []
            with httpx.stream(
                "POST",
                f"{BASE_URL}/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {secret}"},
                timeout=180.0,
            ) as r:
                print(f"status={r.status_code} ctype={r.headers.get('content-type')}")
                if r.status_code != 200:
                    return fail(f"stream failed: {r.read().decode()[:500]}")
                text = ""
                for chunk in r.iter_bytes():
                    chunks.append(chunk)
                    text += chunk.decode(errors="replace")
                print(
                    f"streamed {len(chunks)} chunks, done={text.strip().endswith('[DONE]')}"
                )
                if "[DONE]" not in text:
                    return fail("stream missing terminal [DONE]")
            # Stop the gateway to flush usage.json, then check attribution.
            proc.terminate()
            proc.wait(timeout=15)
            usage_path = os.path.join(data_dir, "usage.json")
            if not os.path.exists(usage_path):
                return fail("usage.json missing after gateway shutdown")
            with open(usage_path) as f:
                snapshot = json.load(f)
            entry = snapshot.get("keys", {}).get(secret)
            print(f"usage snapshot: {json.dumps(entry)[:400]}")
            if not entry or entry.get("requests", 0) < 1:
                return fail("no usage attributed to probe secret")
            if entry.get("input_tokens", 0) <= 0:
                return fail(
                    "stream recorded request-only (no input tokens); "
                    "expected streamed token attribution"
                )
            print(
                f"stream usage probe ok: requests={entry['requests']} "
                f"in={entry['input_tokens']} out={entry.get('output_tokens')} "
                f"cached={entry.get('cached_tokens')}"
            )
            return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
