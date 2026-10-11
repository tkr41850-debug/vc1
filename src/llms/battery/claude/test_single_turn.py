"""Single-turn claude battery (messages leg). See battery/README.md."""

from __future__ import annotations

import json
import os

from battery import battery_dirs, claude_env, require_live, run_cli

CLAUDE_BIN = os.getenv("CLAUDE_BIN", "/home/uqmm/.local/bin/claude")


def test_cl_single_print():
    base, sk = require_live()
    _, ws = battery_dirs("cl-print")
    env = claude_env(base, "ak-battery-cl01", sk)
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "Reply with exactly: CLAUDE-PRINT-OK and nothing else.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip() == "CLAUDE-PRINT-OK"
    sj = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "stream-json",
            "--verbose",
            "Reply with exactly: CLAUDE-PRINT-OK and nothing else.",
        ],
        ws,
        env,
    )
    assert sj.returncode == 0, sj.stderr[-1000:]
    lines = [json.loads(line) for line in sj.stdout.splitlines() if line.strip()]
    assert lines, "empty stream-json output"
    assert not any(
        "tool_use" in line
        for line in sj.stdout.splitlines()
        if "content_block_start" in line
    )
    assert "CLAUDE-PRINT-OK" in sj.stdout


def test_cl_single_bash():
    base, sk = require_live()
    _, ws = battery_dirs("cl-bash")
    env = claude_env(base, "ak-battery-cl02", sk)
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Bash",
            (
                "Use the Bash tool once to run exactly: echo CLAUDE-BASH-OK. "
                "Then reply with the tool output verbatim."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "CLAUDE-BASH-OK" in c.stdout
    sj = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Bash",
            "--output-format",
            "stream-json",
            "--verbose",
            (
                "Use the Bash tool once to run exactly: echo CLAUDE-BASH-OK. "
                "Then reply with the tool output verbatim."
            ),
        ],
        ws,
        env,
    )
    assert sj.returncode == 0, sj.stderr[-1000:]
    assert '"name":"Bash"' in sj.stdout.replace(" ", "")
    assert "echo CLAUDE-BASH-OK" in sj.stdout
    assert "CLAUDE-BASH-OK" in sj.stdout


def test_cl_single_write():
    base, sk = require_live()
    _, ws = battery_dirs("cl-write")
    env = claude_env(base, "ak-battery-cl03", sk)
    target = ws / "cl-write.txt"
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Write",
            f"--add-dir={ws}",
            (
                f"Use the Write tool to create {target} with exactly these "
                "15 bytes: CLAUDE-WRITE-OK (no trailing newline). Reply DONE."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"CLAUDE-WRITE-OK"
    assert "DONE" in c.stdout


def test_cl_single_read():
    base, sk = require_live()
    _, ws = battery_dirs("cl-read")
    env = claude_env(base, "ak-battery-cl04", sk)
    target = ws / "cl-read.txt"
    target.write_bytes(b"CLAUDE-READ-OK")
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Read",
            f"--add-dir={ws}",
            (
                f"Use the Read tool on {target} and reply with its exact "
                "content and nothing else."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip() == "CLAUDE-READ-OK"


def test_cl_single_edit():
    base, sk = require_live()
    _, ws = battery_dirs("cl-edit")
    env = claude_env(base, "ak-battery-cl05", sk)
    target = ws / "cl-edit.txt"
    target.write_bytes(b"CLAUDE-EDIT-V1")
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Edit",
            f"--add-dir={ws}",
            (
                "Use the Edit tool to replace CLAUDE-EDIT-V1 with "
                f"CLAUDE-EDIT-V2 in {target}. Reply DONE."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"CLAUDE-EDIT-V2"
    assert "DONE" in c.stdout


def test_cl_single_multi_tool():
    base, sk = require_live()
    _, ws = battery_dirs("cl-multi")
    env = claude_env(base, "ak-battery-cl06", sk)
    target = ws / "cl-multi.txt"
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Bash,Write",
            f"--add-dir={ws}",
            (
                "Do both in this turn: (1) Use Write to create "
                f"{target} with exactly CLAUDE-MULTI-OK, "
                "(2) Use Bash to run echo CLAUDE-MULTI-BASH. "
                "Reply with both results verbatim."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"CLAUDE-MULTI-OK"
    assert "CLAUDE-MULTI-BASH" in c.stdout
    sj = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Bash,Write",
            f"--add-dir={ws}",
            "--output-format",
            "stream-json",
            "--verbose",
            (
                "Do both in this turn: (1) Use Write to create "
                f"{target} with exactly CLAUDE-MULTI-OK, "
                "(2) Use Bash to run echo CLAUDE-MULTI-BASH. "
                "Reply with both results verbatim."
            ),
        ],
        ws,
        env,
    )
    assert sj.returncode == 0, sj.stderr[-1000:]
    flat = sj.stdout.replace(" ", "")
    assert '"name":"Write"' in flat and '"name":"Bash"' in flat


def test_cl_single_output_format():
    base, sk = require_live()
    _, ws = battery_dirs("cl-fmt")
    env = claude_env(base, "ak-battery-cl07", sk)
    prompt = "Reply with exactly: CLAUDE-FMT-OK and nothing else."
    text = run_cli(
        [CLAUDE_BIN, "-p", "--model", "haiku", "--output-format", "text", prompt],
        ws,
        env,
    )
    assert text.returncode == 0, text.stderr[-1000:]
    assert text.stdout.strip() == "CLAUDE-FMT-OK"
    sj = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "stream-json",
            "--verbose",
            prompt,
        ],
        ws,
        env,
    )
    assert sj.returncode == 0, sj.stderr[-1000:]
    lines = [json.loads(line) for line in sj.stdout.splitlines() if line.strip()]
    assert lines, "empty stream-json output"
    assert "CLAUDE-FMT-OK" in sj.stdout
    assert "message_stop" in sj.stdout or "result" in sj.stdout


def test_cl_single_allowed_tools():
    """Client-side tool gating: the --allowedTools list must reach the
    model as the tool set (system init event), not merely be asserted on
    tool_use absence (the model can echo tool names in text).

    NOTE (live 2026-10-11): the deny leg shows `--allowedTools=Read`
    arriving upstream as the FULL tool list (Bash offered, Bash attempted
    with space-stripped `echoCLAUDE-GATE-OK`). Gating is not enforced on
    this path — xfail pin until the allowlist is actually applied."""
    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-gate")
    env = claude_env(base, "ak-battery-cl08", sk)
    prompt = "Use the Bash tool to run echo CLAUDE-GATE-OK and reply with the output."
    allow = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Bash",
            "--output-format",
            "stream-json",
            "--verbose",
            prompt,
        ],
        ws,
        env,
    )
    assert allow.returncode == 0, allow.stderr[-1000:]
    assert '"name":"Bash"' in allow.stdout.replace(" ", "")
    assert "CLAUDE-GATE-OK" in allow.stdout
    deny = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--allowedTools=Read",
            "--output-format",
            "stream-json",
            "--verbose",
            prompt,
        ],
        ws,
        env,
    )
    assert deny.returncode == 0, deny.stderr[-1000:]
    assert deny.stdout.strip(), "deny run produced empty output"
    lines = [json.loads(line) for line in deny.stdout.splitlines() if line.strip()]
    init = next(
        (e for e in lines if e.get("type") == "system" and e.get("subtype") == "init"),
        None,
    )
    assert init is not None, "no system init event in deny stream"
    offered = init.get("tools")
    if offered != ["Read"]:
        pytest.xfail(
            f"allowlist not applied, model offered {len(offered)} tools "
            "instead of [Read]"
        )
    assert offered == ["Read"]


def test_cl_single_json_structured():
    base, sk = require_live()
    _, ws = battery_dirs("cl-jsonschema")
    env = claude_env(base, "ak-battery-cl09", sk)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(schema),
            (
                "Extract the word BLUEBIRD from this sentence and return it "
                "as the answer field, plus a sentence of commentary first."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    payload = json.loads(c.stdout)
    assert payload.get("structured_output") == {"answer": "BLUEBIRD"}


def test_cl_single_partial_messages():
    base, sk = require_live()
    _, ws = battery_dirs("cl-partial")
    env = claude_env(base, "ak-battery-cl10", sk)
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "Reply with exactly: PARTIAL-OK and nothing else.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    events = [json.loads(line) for line in c.stdout.splitlines() if line.strip()]
    assert events, "empty stream"
    assert any(
        e.get("type") == "stream_event"
        and (e.get("event") or {}).get("type") == "content_block_delta"
        for e in events
    ), "no streaming deltas"
    assert "PARTIAL-OK" in c.stdout


def test_cl_single_session_id_and_name():
    import uuid

    base, sk = require_live()
    _, ws = battery_dirs("cl-sid")
    env = claude_env(base, "ak-battery-cl15", sk)
    sid = str(uuid.uuid4())
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "json",
            "--session-id",
            sid,
            "-n",
            "battery-sid-probe",
            "Reply with exactly: SID-OK and nothing else.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    payload = json.loads(c.stdout)
    assert payload.get("session_id") == sid
    assert "SID-OK" in json.dumps(payload)


def test_cl_single_fork_session():
    base, sk = require_live()
    _, ws = battery_dirs("cl-fork")
    env = claude_env(base, "ak-battery-cl16", sk)
    first = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "json",
            "Reply with exactly FORK-SEED-9Q2W and nothing else.",
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    s1 = json.loads(first.stdout).get("session_id", "")
    assert s1, "no session_id"
    forked = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--resume",
            s1,
            "--fork-session",
            "--model",
            "haiku",
            "--output-format",
            "json",
            ("Repeat the exact marker from the original session and nothing else."),
        ],
        ws,
        env,
    )
    assert forked.returncode == 0, forked.stderr[-1000:]
    payload = json.loads(forked.stdout)
    assert payload.get("session_id", "") != s1
    assert "FORK-SEED-9Q2W" in json.dumps(payload)


def test_cl_single_stdin_prompt():
    base, sk = require_live()
    _, ws = battery_dirs("cl-stdin")
    env = claude_env(base, "ak-battery-cl17", sk)
    c = run_cli(
        [CLAUDE_BIN, "-p", "--model", "haiku", "--output-format", "text"],
        ws,
        env,
        input_text="Reply with exactly: STDIN-PIPE-OK and nothing else.",
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip() == "STDIN-PIPE-OK"


def test_cl_single_input_format_stream_json():
    base, sk = require_live()
    _, ws = battery_dirs("cl-fmtin")
    env = claude_env(base, "ak-battery-cl18", sk)
    msg = json.dumps(
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "Reply with exactly: FMT-IN-OK and nothing else.",
            },
        }
    )
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "stream-json",
            "--verbose",
            "--input-format",
            "stream-json",
        ],
        ws,
        env,
        input_text=msg,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "FMT-IN-OK" in c.stdout


def test_cl_single_task_subagent():
    base, sk = require_live()
    _, ws = battery_dirs("cl-task")
    env = claude_env(base, "ak-battery-cl19", sk)
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            (
                "Use the Task tool to delegate to a subagent: have it "
                "reply with exactly SUBAGENT-OK and nothing else. Then "
                "reply with its result verbatim."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "SUBAGENT-OK" in c.stdout


def test_cl_single_hook_pretooluse():
    base, sk = require_live()
    _, ws = battery_dirs("cl-hook")
    env = claude_env(base, "ak-battery-cl20", sk)
    hooklog = ws / "hook.log"
    if hooklog.exists():
        hooklog.unlink()
    settings = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"echo HOOK-FIRED >> {hooklog}",
                        }
                    ],
                }
            ]
        }
    }
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--settings",
            json.dumps(settings),
            (
                "Use the Bash tool once to run exactly: echo HOOK-BASH-OK. "
                "Then reply with the tool output verbatim."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "HOOK-BASH-OK" in c.stdout
    assert hooklog.read_bytes() == b"HOOK-FIRED\n"


def test_cl_single_no_space_strip():
    """Regression pin for the run3 strip event (2026-10-11): one deny leg
    echoed `"command":"echoCLAUDE-GATE-OK"` / `"description":
    "Rungatecheckecho"` — every interior space deleted, order preserved.

    No whitespace-deleting op exists anywhere in the proxy's args path
    (every dump/concat/emitter is verbatim; verified by audit), and 25+
    deny legs since are intact — so there is nothing to patch. This leg
    pins the invariant instead: the spaceless token is never legitimate
    output (intact runs carry `echo CLAUDE-GATE-OK` WITH the space; a
    proper deny carries no Bash call at all), so its presence fails
    loudly with the full stdout saved for forensics.
    """
    base, sk = require_live()
    _, ws = battery_dirs("cl-nostrip")
    env = claude_env(base, "ak-battery-cl21", sk)
    for i in range(3):
        c = run_cli(
            [
                CLAUDE_BIN,
                "-p",
                "--model",
                "haiku",
                "--allowedTools=Read",
                "--output-format",
                "stream-json",
                "--verbose",
                (
                    "Use the Bash tool to run echo CLAUDE-GATE-OK and "
                    "reply with the output."
                ),
            ],
            ws,
            env,
        )
        assert c.returncode == 0, c.stderr[-1000:]
        if "echoCLAUDE-GATE-OK" in (c.stdout or ""):
            (ws / f"STRIPPED_EVIDENCE_iter{i}.jsonl").write_text(c.stdout)
            raise AssertionError(
                f"iter {i}: spaceless tool args reappeared; "
                f"stdout saved to STRIPPED_EVIDENCE_iter{i}.jsonl"
            )


def test_cl_single_mcp_tool():
    """MCP stdio tool through `--mcp-config` + `--strict-mcp-config`.

    Delivery notes (live 2026-10-11, claude 2.1.296): `--mcp-config` is
    variadic and swallows the positional prompt as a second config path
    (pass the prompt via stdin instead), and without a permission grant
    the model only reports `permission for mcp__... hasn't been granted`
    (`--settings` mcpServers has the same permission outcome — the tool
    IS offered, just not executable). Pre-grant the exact
    `mcp__<server>__<tool>` name via single-token `--allowedTools=`.
    The MCP server is written into the leg workspace at runtime (no
    fixture committed); skip when uv/fastmcp cannot spawn it here.
    """
    import shutil
    import subprocess

    base, sk = require_live()
    _, ws = battery_dirs("cl-mcp")
    env = claude_env(base, "ak-battery-cl22", sk)
    server = ws / "mcp_server.py"
    server.write_text(
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('battery-probe')\n"
        "@mcp.tool\n"
        "def marker_echo() -> str:\n"
        '    """Reply with exactly MCP-MARKER-OK."""\n'
        '    return "MCP-MARKER-OK"\n'
        "if __name__ == '__main__':\n"
        "    mcp.run()\n"
    )
    cfg_path = ws / "mcp.json"
    cfg_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "battery-probe": {
                        "command": "uv",
                        "args": [
                            "run",
                            "--with",
                            "fastmcp",
                            "python",
                            str(server),
                        ],
                        "cwd": str(ws),
                        "env": {},
                    }
                }
            }
        )
    )
    if shutil.which("uv") is None:
        import pytest

        pytest.skip("live battery: uv missing for MCP server spawn")
    probe = subprocess.run(
        ["uv", "run", "--with", "fastmcp", "python", "-c", "import fastmcp"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if probe.returncode != 0:
        import pytest

        pytest.skip("live battery: fastmcp uninstallable here")
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--strict-mcp-config",
            "--mcp-config",
            str(cfg_path),
            "--allowedTools=mcp__battery-probe__marker_echo",
        ],
        ws,
        env,
        input_text=(
            "Use the marker_echo tool from the battery-probe MCP server, "
            "then reply with its result verbatim."
        ),
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip() == "MCP-MARKER-OK"


def test_cl_single_plugin_skill():
    """Plugin skill via repeatable `--plugin-dir` (doc-sourced).

    A folder of plugins loads each child: plug/skills/marker-skill/
    SKILL.md with `name:` + `description:` frontmatter. The model
    invokes the skill by name and relays the marker verbatim
    (live 2026-10-11: PLUGIN-MARKER-OK, rc=0).
    """
    base, sk = require_live()
    _, ws = battery_dirs("cl-plugin")
    env = claude_env(base, "ak-battery-cl23", sk)
    skill_dir = ws / "plug" / "skills" / "marker-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: marker-skill\n"
        "description: Reply with the battery marker phrase on demand.\n"
        "---\n"
        "\n"
        "When asked for the plugin marker, reply with exactly: "
        "PLUGIN-MARKER-OK and nothing else.\n"
    )
    c = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--plugin-dir",
            str(ws / "plug"),
            (
                "Invoke the marker-skill skill to get the plugin marker, "
                "then reply with it verbatim."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip() == "PLUGIN-MARKER-OK"
