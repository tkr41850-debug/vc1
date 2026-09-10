from __future__ import annotations

import os
import uuid

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}/ak-probe/v1"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def main() -> int:
    from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

    try:
        with (
            probe_dirs("dsh-probe") as (home, workspace),
            running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/dsh-probe-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/dsh-probe-data"),
            ),
        ):
            config = DeepSeekHarnessConfig(
                provider="deepseek-official",
                model=MODEL,
                max_tokens=2048,
                cwd=str(workspace),
                dsh_home=str(home),
                profile="sdk-minimal",
                base_url=BASE_URL,
                api_key="dummy",
            )
            with DeepSeekHarness(config) as harness:
                session = f"probe-{uuid.uuid4().hex[:8]}"
                first = harness.run("Reply with exactly: turn-one.", session_id=session)
                print(f"turn1: {first.finish_reason} {first.final_response!r}")
                second = harness.run(
                    "Reply with exactly: turn-two.", session_id=session
                )
                print(f"turn2: {second.finish_reason} {second.final_response!r}")
                if second.finish_reason != "completed":
                    print(f"notifications={second.notifications!r}")
                    return fail(f"turn2 did not complete: {second.finish_reason}")
                third = harness.run(
                    "Use bash to run echo tool-ok, then reply with exactly its output.",
                    session_id=session,
                )
                print(f"tools: {third.finish_reason} {third.final_response!r}")
                if (
                    third.finish_reason != "completed"
                    or "tool-ok" not in third.final_response
                ):
                    print(f"notifications={third.notifications!r}")
                    return fail(
                        f"tool turn failed: {third.finish_reason} {third.final_response!r}"
                    )
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
