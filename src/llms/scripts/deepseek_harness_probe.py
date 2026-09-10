from __future__ import annotations

import json
import os

from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = os.getenv("PROBE_BASE_URL", f"http://127.0.0.1:{PORT}/v1")
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def main() -> int:
    from openai import OpenAI

    try:
        with running_proxy(PORT):
            client = OpenAI(api_key="test-key", base_url=BASE_URL)
            response = client.responses.create(
                model=MODEL,
                input="reply with exactly: harness-ok",
                max_output_tokens=512,
            )
            print(
                json.dumps(
                    {
                        "id": response.id,
                        "model": response.model,
                        "status": response.status,
                        "output_text": response.output_text,
                        "usage": {
                            "input_tokens": response.usage.input_tokens,
                            "output_tokens": response.usage.output_tokens,
                        },
                    },
                    indent=2,
                )
            )
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
