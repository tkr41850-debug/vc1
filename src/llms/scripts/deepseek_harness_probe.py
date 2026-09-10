from __future__ import annotations

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
            first = client.responses.create(
                model=MODEL,
                input="reply with exactly: turn-one",
                max_output_tokens=512,
            )
            print(f"turn1: {first.status} {first.output_text!r}")
            second = client.responses.create(
                model=MODEL,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "reply with exactly: turn-one",
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": first.output_text}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "reply with exactly: turn-two",
                            }
                        ],
                    },
                ],
                max_output_tokens=512,
            )
            print(f"turn2: {second.status} {second.output_text!r}")
            if second.status != "completed":
                return fail(f"turn2 did not complete: {second.status}")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
