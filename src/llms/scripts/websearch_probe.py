from __future__ import annotations

import os

import httpx

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")
PROBE_SECRET = os.getenv("PROBE_SECRET", "sk-probe")


def main() -> int:
    try:
        with (
            probe_dirs("websearch-probe"),
            running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/websearch-probe-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/websearch-probe-data"),
            ),
        ):
            body = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": "Use web search to find what Opencode Zen is, then reply briefly.",
                    }
                ],
                "tools": [
                    {"type": "web_search_20260205", "name": "web_search", "max_uses": 3}
                ],
                "max_tokens": 2048,
            }
            r = httpx.post(
                f"{BASE_URL}/v1/messages",
                json=body,
                headers={"Authorization": f"Bearer {PROBE_SECRET}"},
                timeout=180.0,
            )
            if r.status_code != 200:
                return fail(f"web search failed: {r.text[:500]}")
            payload = r.json()
            texts = [
                b.get("text", "")
                for b in payload.get("content", [])
                if b.get("type") == "text"
            ]
            print(f"stop={payload.get('stop_reason')} text={''.join(texts)[:300]!r}")
            if payload.get("stop_reason") not in (
                "end_turn",
                "tool_use",
                "stop_sequence",
            ):
                return fail(f"unexpected stop: {payload.get('stop_reason')}")
            if not any("zen" in t.lower() for t in texts):
                return fail("answer shows no sign of search grounding")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
