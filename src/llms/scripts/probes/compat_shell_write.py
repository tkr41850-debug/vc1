"""Verify the spark-leg file-write via shell redirect (no exec channel).

The codex-plain (spark) leg declares only the `exec_command` shell
runner — there is no `exec` channel, so `apply_patch`-as-such cannot
run there (mechanism 12). But the runner executed 11/11 live
emissions including compound commands (`pwd; ls -la; cat`), so a
file-write via shell redirect may work through the already-proven
path. Drives the spark leg with a natural file-creation prompt,
then reads back the file and asserts byte-exact content:

    python scripts/probes/compat_shell_write.py

Prints PASS/FAIL with the harness's verbatim output on failure.
Probes live in scripts/ (externals-dependent); hermetic tests in
tests/ (see probes_vs_tests memory).
"""

from __future__ import annotations

import os
import subprocess

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MODEL = os.getenv("PROBE_MODEL", "muse-spark-1.3-contributor-free")

FILENAME = "shell-write.txt"
EXPECTED = "hello-shell\n"


def main() -> int:
    try:
        with probe_dirs("compat-shell-write") as (home, workspace):
            target = workspace / FILENAME
            if target.exists():
                target.unlink()
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
                os.getenv("PROXY_LOG", "/tmp/compat-shell-write-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/compat-shell-write-data"),
                probe_secret=os.getenv("PROBE_SECRET"),
            ) as (_, secret):
                env = dict(os.environ, CODEX_HOME=str(home), OPENAI_API_KEY=secret)
                prompt = (
                    f"Create a file named {FILENAME} in the current directory "
                    f"with exactly this content: {EXPECTED!r}. "
                    "Use whatever tool you have available. "
                    "Then reply with ONLY the word DONE, nothing else."
                )
                completed = subprocess.run(
                    [
                        "codex",
                        "exec",
                        "--skip-git-repo-check",
                        "-s",
                        "workspace-write",
                        prompt,
                    ],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                print(completed.stdout[-1500:])
                if completed.returncode != 0:
                    return fail(
                        f"codex failed: {completed.returncode} {completed.stderr[-2000:]}"
                    )
                try:
                    content = target.read_text()
                except FileNotFoundError:
                    return fail(f"FAIL shell-write: {target} not created")
                if content != EXPECTED:
                    return fail(
                        f"FAIL shell-write: content {content!r} != {EXPECTED!r}"
                    )
                print(f"PASS shell-write: byte-exact {content!r}")
                return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
