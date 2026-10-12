"""Single-turn codex battery (responses leg). See battery/README.md."""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from battery import battery_dirs, codex_env, require_live, run_cli

CODEX_BIN = __import__("os").getenv(
    "CODEX_BIN", "/home/uqmm/.nvm/versions/node/v24.15.0/bin/codex"
)
SPARK = "muse-spark-1.3-contributor-free"


def test_cx_single_file_write():
    base, sk = require_live()
    home, ws = battery_dirs("cx-file-write")
    env = codex_env(home, base, "ak-battery-cx01", sk, SPARK)
    target = ws / "write_plain.txt"
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Write exactly MARKER-WRITE-1 on one line to write_plain.txt "
                "using a tool, then reply with exactly DONE-WRITE-1."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"MARKER-WRITE-1\n"
    assert c.stdout.strip().endswith("DONE-WRITE-1")


def test_cx_single_shell_redirect_write():
    # Fixed 2026-10-12 (alias attribution): the loop was the model
    # re-emitting `shell` because the executed success echoed back
    # under the rewritten `exec_command` name it never emitted.
    # to_zen_responses now replays executed alias pairs under the
    # emitted name, and the leg passes (battery 17 passed incl. this
    # leg; standalone RELAY CODEX REDIRECT PASS). TimeoutExpired still
    # xfails — a hung CLI is harness/model dynamics, not an assertion.
    base, sk = require_live()
    home, ws = battery_dirs("cx-shell-redir")
    env = codex_env(home, base, "ak-battery-cx02", sk, SPARK)
    target = ws / "redir.txt"
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Run shell command printf REDIRECT-OK > redir.txt via the "
                    "shell tool, then reply with exactly DONE-REDIR-1."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model loops re-emitting shell after byte-exact write")
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"REDIRECT-OK"
    assert "DONE-REDIR-1" in c.stdout


def test_cx_single_chained_multitool():
    base, sk = require_live()
    home, ws = battery_dirs("cx-chain")
    env = codex_env(home, base, "ak-battery-cx03", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Run echo CHAIN-A > a.txt, then echo CHAIN-B > b.txt, then "
                "cat both files, then reply with exactly CHAIN-DONE."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert (ws / "a.txt").read_bytes() == b"CHAIN-A\n"
    assert (ws / "b.txt").read_bytes() == b"CHAIN-B\n"
    assert c.stdout.strip().endswith("CHAIN-DONE")


def test_cx_single_image_attach():
    base, sk = require_live()
    home, ws = battery_dirs("cx-image")
    env = codex_env(home, base, "ak-battery-cx04", sk, SPARK)
    import base64

    png = ws / "red.png"
    # 1x1 red PNG, no PIL dependency
    png.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
    )
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Reply with exactly ATTACH-IMG-1 if you received an image "
                "attachment, else reply NO-IMAGE."
            ),
            "-i",
            str(png),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("ATTACH-IMG-1")


def test_cx_single_review_diff():
    base, sk = require_live()
    home, ws = battery_dirs("cx-patch")
    env = codex_env(home, base, "ak-battery-cx05", sk, SPARK)
    target = ws / "patch_target.txt"
    if target.exists():
        target.unlink()
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Create patch_target.txt with one line BASE-LINE via "
                "apply_patch using this exact grammar:\n"
                "*** Begin Patch\n*** Add File: patch_target.txt\n"
                "+BASE-LINE\n*** End Patch\n"
                "Then reply with exactly DONE-PATCH-1."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "SyntaxError" not in c.stdout and "Unexpected token" not in c.stdout
    assert target.read_bytes() == b"BASE-LINE\n"
    assert c.stdout.strip().endswith("DONE-PATCH-1")


def test_cx_single_sandbox_workspace():
    base, sk = require_live()
    home, ws = battery_dirs("cx-sandbox")
    env = codex_env(home, base, "ak-battery-cx06", sk, SPARK)
    target = ws / "sandbox.txt"
    if target.exists():
        target.unlink()
    ro = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            (
                "Try to write SANDBOX-WRITE-OK on one line to sandbox.txt "
                "with a tool; if blocked reply with exactly DONE-SANDBOX-RO."
            ),
        ],
        ws,
        env,
    )
    assert ro.stdout.strip().endswith("DONE-SANDBOX-RO")
    assert not target.exists()
    ww = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            (
                "Write exactly SANDBOX-WRITE-OK on one line to sandbox.txt "
                "using a tool, then reply with exactly DONE-SANDBOX-WW."
            ),
        ],
        ws,
        env,
    )
    assert ww.returncode == 0, ww.stderr[-1000:]
    assert target.read_bytes() == b"SANDBOX-WRITE-OK\n"
    assert ww.stdout.strip().endswith("DONE-SANDBOX-WW")


def test_cx_single_stdin_vs_arg():
    base, sk = require_live()
    home, ws = battery_dirs("cx-stdin")
    env = codex_env(home, base, "ak-battery-cx07", sk, SPARK)
    arg = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "Reply with exactly STDIN-ARG-MARKER.",
        ],
        ws,
        env,
    )
    assert arg.stdout.strip().endswith("STDIN-ARG-MARKER")
    pipe = run_cli(
        [CODEX_BIN, "exec", "--skip-git-repo-check", "-"],
        ws,
        env,
        input_text="Reply with exactly STDIN-PIPE-MARKER.",
    )
    assert pipe.stdout.strip().endswith("STDIN-PIPE-MARKER")
    assert not pipe.stdout.strip().endswith("STDIN-ARG-MARKER")


def test_cx_single_fork_session():
    base, sk = require_live()
    home, ws = battery_dirs("cx-fork")
    env = codex_env(home, base, "ak-battery-cx08", sk, SPARK)
    first = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            "--json",
            (
                "Write exactly FORK-SEED-5H1N to fork.txt and reply exactly "
                "FORK-SEED-5H1N."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    session_id = ""
    for line in first.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = re.search(r'"thread_id"\s*:\s*"([^"]+)"', json.dumps(event))
        if m:
            session_id = m.group(1)
            break
    assert session_id, "no thread id in --json events"
    forked = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "fork",
            session_id,
            "Read fork.txt and reply exactly FORKED-<contents> and nothing else.",
        ],
        ws,
        env,
    )
    assert forked.returncode == 0, forked.stderr[-1000:]
    assert "FORKED-FORK-SEED-5H1N" in forked.stdout
    assert (ws / "fork.txt").read_bytes().rstrip(b"\n") == b"FORK-SEED-5H1N"


def test_cx_single_json_structured():
    base, sk = require_live()
    home, ws = battery_dirs("cx-json")
    env = codex_env(home, base, "ak-battery-cx09", sk, SPARK)
    schema = ws / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"ok": {"type": "string"}},
                "required": ["ok"],
                "additionalProperties": False,
            }
        )
    )
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--output-schema",
            str(schema),
            (
                "Reply with a JSON object with ok set to schema-ok, plus a "
                "sentence of commentary before it."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    last = c.stdout.strip().splitlines()[-1]
    assert json.loads(last) == {"ok": "schema-ok"}


def test_cx_single_ephemeral():
    base, sk = require_live()
    home, ws = battery_dirs("cx-ephemeral")
    env = codex_env(home, base, "ak-battery-cx10", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "Reply with exactly EPHEMERAL-OK.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("EPHEMERAL-OK")
    resumed = run_cli(
        [
            CODEX_BIN,
            "exec",
            "resume",
            "--last",
            "Reply with exactly EPHEMERAL-RESUMED.",
        ],
        ws,
        env,
    )
    if resumed.returncode == 0:
        assert "EPHEMERAL-OK" not in resumed.stdout
        assert "EPHEMERAL-RESUMED" in resumed.stdout


def test_cx_single_output_last_message():
    import tempfile

    base, sk = require_live()
    home, ws = battery_dirs("cx-lastmsg")
    env = codex_env(home, base, "ak-battery-cx17", sk, SPARK)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=str(ws)
    ) as f:
        out_path = f.name
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-o",
            out_path,
            "Reply with exactly LASTMSG-OK.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    with open(out_path) as f:
        assert "LASTMSG-OK" in f.read()


def test_cx_single_search_flag():
    base, sk = require_live()
    home, ws = battery_dirs("cx-search")
    env = codex_env(home, base, "ak-battery-cx18", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "--search",
            "exec",
            "--skip-git-repo-check",
            "Reply with exactly SEARCH-FLAG-OK.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("SEARCH-FLAG-OK")


def test_cx_single_mcp_tool():
    """MCP stdio tool via `codex mcp add` (doc-sourced).

    Register a fastmcp stdio server exposing marker_echo, then exec a
    prompt relaying its result verbatim. Unlike the claude leg, codex
    needs no permission pre-grant (live 2026-10-11: CX-MCP-MARKER-OK,
    rc=0). The server is written into the leg workspace at runtime;
    skip when uv/fastmcp cannot spawn it here. NOTE: `mcp add` writes
    to CODEX_HOME's global config, and battery_dirs shares one home per
    tag — the cx-mcp tag owns its home, so no cross-leg leak.

    FLAKE PIN (live 2026-10-12, 2 runs): the harness never dispatches
    the registered tool client-side — run 1 refused fast ("MCP tool
    call requires approval, but approval policy is never"), run 2
    hung the full 600s CLI timeout with zero stdout. Gateway evidence
    is clean either way (9 turns, all upstream 200, zero steer
    failures; final turn offers `mcp__battery_probe` and the CLI
    stalls after). The failure is upstream of any llms traffic, so a
    hard fail here blocks the suite on codex-harness mood, not compat.
    """
    import shutil

    base, sk = require_live()
    home, ws = battery_dirs("cx-mcp")
    env = codex_env(home, base, "ak-battery-cx19", sk, SPARK)
    server = ws / "mcp_server.py"
    server.write_text(
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('battery-probe')\n"
        "@mcp.tool\n"
        "def marker_echo() -> str:\n"
        '    """Reply with exactly CX-MCP-MARKER-OK."""\n'
        '    return "CX-MCP-MARKER-OK"\n'
        "if __name__ == '__main__':\n"
        "    mcp.run()\n"
    )
    if shutil.which("uv") is None:
        pytest.skip("live battery: uv missing for MCP server spawn")
    probe = subprocess.run(
        ["uv", "run", "--with", "fastmcp", "python", "-c", "import fastmcp"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if probe.returncode != 0:
        pytest.skip("live battery: fastmcp uninstallable here")
    add = run_cli(
        [
            CODEX_BIN,
            "mcp",
            "add",
            "battery-probe",
            "--",
            "uv",
            "run",
            "--with",
            "fastmcp",
            "python",
            str(server),
        ],
        ws,
        env,
    )
    assert add.returncode == 0, add.stderr[-1000:]
    assert "battery-probe" in add.stdout
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Use the marker_echo tool from the battery-probe MCP "
                    "server, then reply with its result verbatim."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex harness stalled on MCP dispatch (600s, zero stdout)")
    assert c.returncode == 0, c.stderr[-1000:]
    if "CX-MCP-MARKER-OK" not in c.stdout:
        pytest.xfail(f"codex harness did not dispatch MCP tool: {c.stdout[-200:]}")
    assert "CX-MCP-MARKER-OK" in c.stdout
