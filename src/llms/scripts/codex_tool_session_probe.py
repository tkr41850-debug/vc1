from __future__ import annotations

import os
import subprocess

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")


def main() -> int:
    """Longer multi-turn codex tool-calling session through the gateway.

    One `codex exec` task with dependent steps, forcing several
    function_call/function_call_output round trips through the translate
    + steer + synthesize paths: write files, run shell over them, read
    the result back, then answer. Verifies the final answer (proves
    every intermediate tool call round-tripped intact).
    """
    try:
        with probe_dirs("codex-tool-session-probe") as (home, workspace):
            (home / "config.toml").write_text(
                f'model = "{MODEL}"\n'
                'model_provider = "zen-proxy"\n'
                "\n"
                "[model_providers.zen-proxy]\n"
                'name = "zen-proxy"\n'
                f'base_url = "{BASE_URL}"\n'
                'wire_api = "responses"\n'
                'env_key = "OPENAI_API_KEY"\n'
            )
            with running_proxy(
                PORT,
                os.getenv("PROXY_LOG", "/tmp/codex-tool-session-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/codex-tool-session-data"),
                probe_secret=os.getenv("PROBE_SECRET"),
            ) as (_, secret):
                env = dict(
                    os.environ,
                    CODEX_HOME=str(home),
                    OPENAI_API_KEY=secret,
                )
                task = (
                    "Do these steps in order using your tools. "
                    "1. Create a file named numbers.txt containing the integers 1 through 10, one per line. "
                    "2. Use the shell to compute their sum and write it to sum.txt. "
                    "3. Read sum.txt back. "
                    "4. Create doubled.txt containing exactly twice that sum as a single integer. "
                    "5. Reply with exactly: session-sum=<value of doubled.txt>."
                )
                completed = subprocess.run(
                    ["codex", "exec", "--skip-git-repo-check", task],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )
                print(completed.stdout[-2000:])
                if completed.returncode != 0:
                    return fail(completed.stderr[-2000:])
                if "session-sum=110" not in completed.stdout:
                    return fail(
                        "wrong session answer (expected session-sum=110): "
                        f"{completed.stdout[-500:]}"
                    )
                # Independent evidence the intermediate tools ran server-side.
                for name, want in (("numbers.txt", "1\n"), ("sum.txt", "55")):
                    path = workspace / name
                    if not path.exists():
                        return fail(f"missing intermediate file {name}")
                    if want not in path.read_text():
                        return fail(f"{name} has unexpected content")
                print("codex tool-session probe ok: 4-step dependent session answered 110")
        return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
