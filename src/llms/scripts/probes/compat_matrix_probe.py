"""Task 7 live matrix probe (QUOTA-GATED — run only when the gate is open).

Drives one natural file-write per leg through a fresh probe proxy and
asserts byte-exact workspace files, so each compat row gets a live
verdict alongside its hermetic pin:

  codex-plain (spark): `write` -> `exec_command` printf redirect
  luna: natural file-write (arm-detector proof: expect a bare-patch
        turn re-wrapped via `__exec_rewrite__`, or shell fallback)
  claude (messages): Bash echo `tool-ok` round-trip (first live tool
        translation on the claude leg; file-write after a claude
        compat row ships)

Gate: run only when >50min elapsed since hi-check.json mtime:
  test $(( $(date +%s) - $(stat -c %Y hi-check.json) )) -gt 3000
Usage:
  PROBE_PORT=8799 python scripts/probes/compat_matrix_probe.py [spark|luna|claude|all]
"""

from __future__ import annotations

import os
import subprocess
import sys

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8794"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MSG_URL = f"http://127.0.0.1:{PORT}"

EXPECTED = "hello-matrix\n"


def _run_codex(home, workspace, secret, model: str, prompt: str, filename: str) -> int:
    (home / "config.toml").write_text(
        f'model = "{model}"\nmodel_provider = "zen-proxy"\n\n'
        f'[model_providers.zen-proxy]\nname = "zen-proxy"\n'
        f'base_url = "{BASE_URL}"\nwire_api = "responses"\n'
        'env_key = "OPENAI_API_KEY"\n'
    )
    target = workspace / filename
    if target.exists():
        target.unlink()
    env = dict(os.environ, CODEX_HOME=str(home), OPENAI_API_KEY=secret)
    c = subprocess.run(
        ["codex", "exec", "--skip-git-repo-check", "-s", "workspace-write", prompt],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        stdin=subprocess.DEVNULL,
    )
    print(c.stdout[-1500:])
    if c.returncode != 0:
        return fail(f"codex failed: {c.returncode} {c.stderr[-2000:]}")
    try:
        content = target.read_text()
    except FileNotFoundError:
        return fail(f"FAIL {filename}: not created")
    if content != EXPECTED:
        return fail(f"FAIL {filename}: {content!r} != {EXPECTED!r}")
    print(f"PASS {filename}: byte-exact {content!r}")
    return 0


def _prompt(filename: str) -> str:
    return (
        f"Create a file named {filename} in the current directory "
        f"with exactly this content: {EXPECTED!r}. Then reply with "
        "ONLY the word DONE, nothing else."
    )


def cell_spark(home, workspace, secret: str) -> int:
    return _run_codex(
        home,
        workspace,
        secret,
        "muse-spark-1.3-contributor-free",
        _prompt("matrix-spark.txt"),
        "matrix-spark.txt",
    )


def cell_luna(home, workspace, secret: str) -> int:
    return _run_codex(
        home,
        workspace,
        secret,
        "gpt-5.6-luna",
        _prompt("matrix-luna.txt"),
        "matrix-luna.txt",
    )


def cell_claude(workspace, secret: str) -> int:
    # The CLI validates its model against Claude-native ids, so scrub
    # any shell ANTHROPIC_MODEL (the operator shell pins a Zen id for
    # its own session) and pass the CLI-native default explicitly —
    # the proxy maps the messages leg onto its default upstream
    # model (default_messages_model) regardless.
    env = dict(os.environ)
    env.pop("ANTHROPIC_MODEL", None)
    env.update(
        ANTHROPIC_BASE_URL=MSG_URL,
        ANTHROPIC_API_KEY=secret,
        ANTHROPIC_AUTH_TOKEN="",
    )
    c = subprocess.run(
        [
            "claude",
            "-p",
            "Use Bash to run echo tool-ok, then reply with exactly its output.",
            "--model",
            "haiku",
            "--allowedTools",
            "Bash",
        ],
        cwd=str(workspace),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    print(c.stdout[-800:])
    if c.returncode != 0 or "tool-ok" not in c.stdout:
        return fail(f"claude tool turn failed: {c.returncode} {c.stderr[-2000:]}")
    print("PASS claude: Bash tool-ok round-trip")
    return 0


def _proxy_kwargs(tag: str):
    return {
        "log_path": os.getenv("PROXY_LOG", f"/tmp/compat-matrix-{tag}-proxy.log"),
        "data_dir": os.getenv("PROBE_DATA_DIR", f"/tmp/compat-matrix-{tag}-data"),
        "probe_secret": os.getenv("PROBE_SECRET"),
    }


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("spark", "luna", "claude", "all"):
        return fail(f"usage: {sys.argv[0]} [spark|luna|claude|all]")
    rc = 0
    for name in [which] if which != "all" else ["spark", "luna", "claude"]:
        with (
            probe_dirs(f"compat-matrix-{name}") as (home, workspace),
            running_proxy(PORT, **_proxy_kwargs(name)) as (_, secret),
        ):
            if name == "spark":
                rc = cell_spark(home, workspace, secret) or rc
            elif name == "luna":
                rc = cell_luna(home, workspace, secret) or rc
            else:
                rc = cell_claude(workspace, secret) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
