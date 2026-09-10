from __future__ import annotations

import os

from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = os.getenv("PROBE_BASE_URL", f"http://127.0.0.1:{PORT}/v1")
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")
PROBE_SECRET = os.getenv("PROBE_SECRET", "sk-probe")


def main() -> int:
    from openai import OpenAI

    try:
        with running_proxy(
            PORT, data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/dsh-probe-data")
        ):
            client = OpenAI(api_key=PROBE_SECRET, base_url=BASE_URL)
            first = client.responses.create(
                model=MODEL,
                input="reply with exactly: turn-one",
                max_output_tokens=1024,
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
                max_output_tokens=1024,
            )
            print(f"turn2: {second.status} {second.output_text!r}")
            if second.status != "completed":
                return fail(f"turn2 did not complete: {second.status}")
            tools = [
                {
                    "type": "function",
                    "name": "bash",
                    "description": "run a shell command",
                    "parameters": {"type": "object"},
                }
            ]
            call = client.responses.create(
                model=MODEL,
                input="use bash to run echo tool-ok, then reply with exactly its output",
                tools=tools,
                max_output_tokens=1024,
            )
            calls = [i for i in call.output if i.type == "function_call"]
            if not calls:
                print(f"tools: model answered directly {call.output_text!r}")
            else:
                print(f"tools: function_call {calls[0].name} {calls[0].arguments!r}")
                done = client.responses.create(
                    model=MODEL,
                    input=[
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": "use bash"}],
                        },
                        {
                            "type": "function_call",
                            "call_id": calls[0].call_id,
                            "name": calls[0].name,
                            "arguments": calls[0].arguments,
                        },
                        {
                            "type": "function_call_output",
                            "call_id": calls[0].call_id,
                            "output": "tool-ok",
                        },
                    ],
                    tools=tools,
                    max_output_tokens=1024,
                )
                print(f"tools: {done.status} {done.output_text!r}")
                if done.status != "completed":
                    return fail(f"tool turn failed: {done.status} {done.output_text!r}")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
