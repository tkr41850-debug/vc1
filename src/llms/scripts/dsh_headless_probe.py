from __future__ import annotations

import os
import uuid

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def main() -> int:
    from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

    try:
        with (
            probe_dirs("dsh-probe") as (home, workspace),
            running_proxy(PORT, os.getenv("PROXY_LOG", "/tmp/dsh-probe-proxy.log")),
        ):
            config = DeepSeekHarnessConfig(
                provider="deepseek-official",
                model=MODEL,
                max_tokens=512,
                cwd=str(workspace),
                dsh_home=str(home),
                profile="sdk-minimal",
                base_url=BASE_URL,
                api_key="dummy",
            )
            with DeepSeekHarness(config) as harness:
                result = harness.run(
                    "Reply with exactly: dsh-ok.",
                    session_id=f"probe-{uuid.uuid4().hex[:8]}",
                )
            print(f"finish_reason={result.finish_reason}")
            print(f"final_response={result.final_response!r}")
            if result.finish_reason == "error":
                print(f"notifications={result.notifications!r}")
                print(f"events={result.events!r}")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
