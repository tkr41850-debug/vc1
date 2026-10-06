"""Verify the luna apply_patch marker grammar per candidate operation.

Drives the luna leg with an explicit `custom_tool_call exec` carrying
`await tools.apply_patch("<candidate patch>")`, then reads back the file
and asserts byte-exact content. One candidate per run:

    python scripts/probes/compat_patch_grammar.py add|update|delete

Prints PASS/FAIL with the harness's verbatim error on failure. Task 3 of
the compat plan: no write->apply_patch table entry ships until the
grammar here is verified live.
"""
from __future__ import annotations

import os
import subprocess
import sys

from llms.probe.dirs import probe_dirs
from llms.probe.proc import fail, running_proxy

PORT = int(os.getenv("PROBE_PORT", "8793"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
MODEL = os.getenv("PROBE_MODEL", "gpt-5.6-luna")

ADD_FILE = "*** Begin Patch\n*** Add File: grammar-add.txt\n+hello-grammar\n*** End Patch"
UPDATE_FILE = (
    "*** Begin Patch\n*** Update File: grammar-update.txt\n"
    "@@\n-old-line\n+new-line\n*** End Patch"
)
DELETE_FILE = "*** Begin Patch\n*** Delete File: grammar-delete.txt\n*** End Patch"

CANDIDATES = {"add": ADD_FILE, "update": UPDATE_FILE, "delete": DELETE_FILE}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in CANDIDATES:
        return fail(f"usage: {sys.argv[0]} add|update|delete")
    op = sys.argv[1]
    patch = CANDIDATES[op]
    try:
        with probe_dirs("compat-grammar") as (home, workspace):
            target = workspace / f"grammar-{op}.txt"
            if op in ("update",):
                target.write_text("old-line\n")
            if op == "delete":
                target.write_text("bye\n")
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
                os.getenv("PROXY_LOG", "/tmp/compat-grammar-proxy.log"),
                data_dir=os.getenv("PROBE_DATA_DIR", "/tmp/compat-grammar-data"),
                probe_secret=os.getenv("PROBE_SECRET"),
            ) as (_, secret):
                env = dict(os.environ, CODEX_HOME=str(home), OPENAI_API_KEY=secret)
                prompt = (
                    "Run this exact JavaScript via the exec custom tool with "
                    f"input exactly: await tools.apply_patch({patch!r}). "
                    "Then reply with ONLY the word DONE, nothing else."
                )
                completed = subprocess.run(
                    ["codex", "exec", "--skip-git-repo-check",
                     "-s", "workspace-write", prompt],
                    cwd=str(workspace),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
                print(completed.stdout[-1500:])
                if completed.returncode != 0:
                    return fail(f"codex failed: {completed.returncode} {completed.stderr[-2000:]}")
                if op == "delete":
                    if target.exists():
                        return fail(f"FAIL delete: {target} still exists")
                    print(f"PASS delete: {target.name} removed")
                    return 0
                try:
                    content = target.read_text()
                except FileNotFoundError:
                    return fail(f"FAIL {op}: {target} not created")
                expected = "hello-grammar\n" if op == "add" else "new-line\n"
                if content != expected:
                    return fail(f"FAIL {op}: content {content!r} != {expected!r}")
                print(f"PASS {op}: byte-exact {content!r}")
                return 0
    except Exception as exc:
        return fail(f"probe failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
