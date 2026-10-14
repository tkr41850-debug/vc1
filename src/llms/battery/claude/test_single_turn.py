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
    """Bash-echo via text + stream-json halves (no-tool-use retry).

    Live 2026-10-14 (flake probe, 2 failures in 10 runs): haiku
    occasionally answers from text without calling any tool — the same
    model turn flake documented on the allowed_tools leg. Both halves
    retry once on marker/tool_use absence; a second miss fails loudly.
    """
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-bash")
    env = claude_env(base, "ak-battery-cl02", sk)
    prompt = (
        "Use the Bash tool once to run exactly: echo CLAUDE-BASH-OK. "
        "Then reply with the tool output verbatim."
    )
    c = None
    try:
        for _ in range(2):
            c = run_cli(
                [
                    CLAUDE_BIN,
                    "-p",
                    "--model",
                    "haiku",
                    "--allowedTools=Bash",
                    prompt,
                ],
                ws,
                env,
            )
            assert c.returncode == 0, c.stderr[-1000:]
            if "CLAUDE-BASH-OK" in c.stdout:
                break
    except subprocess.TimeoutExpired:
        pytest.xfail("claude CLI hung on Bash-echo text half (harness dynamics)")
    assert c is not None
    assert "CLAUDE-BASH-OK" in c.stdout
    sj = None
    try:
        for _ in range(2):
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
                    prompt,
                ],
                ws,
                env,
            )
            assert sj.returncode == 0, sj.stderr[-1000:]
            if '"name":"Bash"' in sj.stdout.replace(" ", ""):
                break
    except subprocess.TimeoutExpired:
        pytest.xfail("claude CLI hung on Bash-echo stream-json half (harness dynamics)")
    assert sj is not None
    # The CLI stream-json init event always lists the FULL tool set
    # (server-side gating, not client-side filtering — live 2026-10-12:
    # --allowedTools=Bash still offered all 23). Pin the enforcement
    # that matters: a Bash tool_use actually ran with the exact input.
    events = [json.loads(line) for line in sj.stdout.splitlines() if line.strip()]
    uses = [
        e
        for e in events
        if e.get("type") == "assistant"
        and any(
            (b.get("type") == "tool_use" and b.get("name") == "Bash")
            for b in (e.get("message", {}).get("content") or [])
            if isinstance(b, dict)
        )
    ]
    assert uses, "no Bash tool_use in stream-json envelope"
    assert "echo CLAUDE-BASH-OK" in sj.stdout
    assert "CLAUDE-BASH-OK" in sj.stdout


def test_cl_single_write():
    """Write-exact-bytes with a stale-file guard (flake probe 2026-10-14).

    Run 5 failed with FileNotFoundError on target.read_bytes(): under
    -n 3 the leg's fixed workspace (/tmp/llms-battery-cl-write) is
    shared with nothing else, so the missing file means the Write
    never landed that turn — either the model skipped the tool call
    (the haiku no-tool-use flake) or a concurrent probe run wiped the
    workspace mid-leg (battery_dirs clears files at leg start). Retry
    once on absence; a second miss fails loudly.
    """
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-write")
    env = claude_env(base, "ak-battery-cl03", sk)
    target = ws / "cl-write.txt"
    c = None
    try:
        for _ in range(2):
            if target.exists():
                target.unlink()
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
            if target.exists() and "DONE" in c.stdout:
                break
    except subprocess.TimeoutExpired:
        pytest.xfail("claude CLI hung on Write leg (harness dynamics)")
    assert c is not None
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
                "Do both in this turn, without any preamble or "
                "narration — emit the Write and Bash calls immediately: "
                "(1) Use Write to create "
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
                "Do both in this turn, without any preamble or "
                "narration — emit the Write and Bash calls immediately: "
                "(1) Use Write to create "
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
    # Live 2026-10-13: haiku occasionally answers from text without
    # calling any tool (hook_started + result, no tool_use) — a model
    # turn flake, not a proxy steer (the proxy only steers declared
    # native tools when required keys are missing). One retry rides
    # the same assertion; a second skip still fails loudly.
    allow = None
    for _ in range(2):
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
        if '"name":"Bash"' in allow.stdout.replace(" ", ""):
            break
    assert allow is not None
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


def test_cl_single_task_close():
    """Task subagent full lifecycle: delegate + TaskStop teardown.

    Ports the codex close leg (claude's capture has no close_agent —
    teardown is TaskStop on the background Task id): the model
    delegates to a subagent for TASK-CLOSE-OK, stops it via TaskStop,
    and replies with the marker verbatim. Marker-in-stdout proves the
    delegatee ran; rc=0 proves the stop did not fail the turn.
    """
    base, sk = require_live()
    _, ws = battery_dirs("cl-task-close")
    env = claude_env(base, "ak-battery-cl24", sk)
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
                "reply with exactly TASK-CLOSE-OK and nothing else. "
                "Stop the task with TaskStop, then reply with its "
                "result verbatim."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "TASK-CLOSE-OK" in c.stdout


def test_cl_single_dontask():
    """--permission-mode dontAsk leg: Bash-echo runs with no prompt hang."""
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-dontask")
    env = claude_env(base, "ak-battery-cl25", sk)
    prompt = (
        "Use the Bash tool once to run exactly: echo CLAUDE-DONTASK-OK. "
        "Then reply with the tool output verbatim."
    )
    # Live model flake: haiku occasionally answers from text without
    # calling any tool (no tool_use in the turn). One retry rides the
    # same assertion; a second miss still fails loudly.
    c = None
    try:
        for _ in range(2):
            c = run_cli(
                [
                    CLAUDE_BIN,
                    "-p",
                    "--model",
                    "haiku",
                    "--permission-mode",
                    "dontAsk",
                    "--allowedTools=Bash",
                    prompt,
                ],
                ws,
                env,
            )
            assert c.returncode == 0, c.stderr[-1000:]
            if "CLAUDE-DONTASK-OK" in c.stdout:
                break
    except subprocess.TimeoutExpired:
        pytest.xfail("claude dontAsk leg timed out (harness dynamics)")
    assert c is not None
    assert c.returncode == 0, c.stderr[-1000:]
    assert "CLAUDE-DONTASK-OK" in c.stdout


def test_cl_single_task_create():
    """Task-management round-trip probe (TaskCreate + TaskList / TodoWrite).

    Asks haiku-via-zen to create a trivial task then list it back and
    relay the round-trip marker. The fixture-pinned CLI toolset
    (tests/fixtures/claude-schema.jsonl, 23 tools) declares no
    TaskCreate/TaskList/TodoWrite — claude-side delegation there is the
    Agent tool (see test_cl_single_task_subagent) — so when none of the
    three is offered in the stream-json init event the leg xfails with
    a clear 'tool not offered' message instead of failing.
    """
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-task-create")
    env = claude_env(base, "ak-battery-cl26", sk)
    prompt = (
        "Use the TaskCreate tool to create a trivial task titled "
        "battery-probe-task, then use the TaskList tool to list tasks. "
        "If TaskCreate/TaskList are unavailable but TodoWrite is offered, "
        "use TodoWrite to add a trivial todo instead. Then reply with "
        "exactly TASK-ROUNDTRIP-OK plus the created task's id."
    )
    task_tools = ("TaskCreate", "TaskList", "TodoWrite")
    c = None
    for _ in range(2):
        try:
            c = run_cli(
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
        except subprocess.TimeoutExpired:
            pytest.xfail("claude CLI hung on task round-trip (harness dynamics)")
        assert c is not None
        assert c.returncode == 0, c.stderr[-1000:]
        flat = c.stdout.replace(" ", "")
        if "TASK-ROUNDTRIP-OK" in c.stdout and any(
            f'"name":"{name}"' in flat for name in task_tools
        ):
            break
    assert c is not None
    lines = [json.loads(line) for line in c.stdout.splitlines() if line.strip()]
    init = next(
        (e for e in lines if e.get("type") == "system" and e.get("subtype") == "init"),
        None,
    )
    assert init is not None, "no system init event in task stream"
    offered = [t.get("name") for t in (init.get("tools") or []) if isinstance(t, dict)]
    if not any(name in offered for name in task_tools):
        pytest.xfail(
            "TaskCreate/TaskList/TodoWrite not offered by haiku-via-zen "
            f"(init lists {len(offered)} tools) - tool not offered, "
            "leg not applicable"
        )
    assert "TASK-ROUNDTRIP-OK" in c.stdout
    flat = c.stdout.replace(" ", "")
    assert any(
        f'"name":"{name}"' in flat for name in task_tools
    ), "round-trip marker relayed without any task tool_use"


def test_cl_single_cron():
    """Cron round-trip probe (CronCreate then CronList).

    Asks haiku-via-zen to schedule an inert one-shot reminder, list the
    session jobs, and relay the created entry. Arg keys are coached
    explicitly (cron + prompt required, recurring false for one-shot —
    per tests/fixtures/claude-schema.jsonl; CronList takes no args),
    the same key-coaching lesson as the codex send_input leg. If the
    tools are absent from the stream-json init event the leg xfails
    with a clear 'tool not offered' message instead of failing.
    """
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-cron")
    env = claude_env(base, "ak-battery-cl27", sk)
    prompt = (
        "Use the CronCreate tool to schedule a one-shot reminder with "
        "cron '41 4 5 5 *', prompt 'battery-cron-probe', recurring false, "
        "then use the CronList tool to list scheduled jobs. Then reply "
        "with exactly CRON-ROUNDTRIP-OK plus the created entry's id."
    )
    c = None
    for _ in range(2):
        try:
            c = run_cli(
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
        except subprocess.TimeoutExpired:
            pytest.xfail("claude CLI hung on cron round-trip (harness dynamics)")
        assert c is not None
        assert c.returncode == 0, c.stderr[-1000:]
        flat = c.stdout.replace(" ", "")
        if (
            "CRON-ROUNDTRIP-OK" in c.stdout
            and '"name":"CronCreate"' in flat
            and '"name":"CronList"' in flat
        ):
            break
    assert c is not None
    lines = [json.loads(line) for line in c.stdout.splitlines() if line.strip()]
    init = next(
        (e for e in lines if e.get("type") == "system" and e.get("subtype") == "init"),
        None,
    )
    assert init is not None, "no system init event in cron stream"
    offered = [t.get("name") for t in (init.get("tools") or []) if isinstance(t, dict)]
    if "CronCreate" not in offered or "CronList" not in offered:
        pytest.xfail(
            "CronCreate/CronList not offered by haiku-via-zen "
            f"(init lists {len(offered)} tools) - tool not offered, "
            "leg not applicable"
        )
    assert "CRON-ROUNDTRIP-OK" in c.stdout
    flat = c.stdout.replace(" ", "")
    assert '"name":"CronCreate"' in flat, "marker relayed without CronCreate tool_use"
    assert '"name":"CronList"' in flat, "marker relayed without CronList tool_use"


def test_cl_single_notebook_edit():
    import json
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-nbedit")
    env = claude_env(base, "ak-battery-cl28", sk)
    nb = ws / "cl-nb.ipynb"
    nb.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "id": "nb-cell-1",
                        "cell_type": "code",
                        "source": ['print("NB-V1")'],
                        "outputs": [],
                        "execution_count": None,
                        "metadata": {},
                    }
                ],
            }
        )
    )
    before = nb.read_bytes()
    try:
        c = run_cli(
            [
                CLAUDE_BIN,
                "-p",
                "--model",
                "haiku",
                "--allowedTools=NotebookEdit",
                f"--add-dir={ws}",
                (
                    "Use the NotebookEdit tool with notebook_path "
                    f"{nb} and cell_id nb-cell-1 to replace the cell "
                    'source with print("NB-EDIT-OK"). Reply DONE.'
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("claude model hung on NotebookEdit dispatch (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    after = nb.read_bytes()
    assert after != before, "notebook bytes unchanged"
    assert b"NB-V1" not in after
    payload = json.loads(after.decode())
    src = "".join(payload["cells"][0]["source"])
    assert "NB-EDIT-OK" in src
    assert "DONE" in c.stdout


def test_cl_single_no_ask_stall():
    """AskUserQuestion/EnterPlanMode must not stall non-interactive -p turns.

    Mirrors the codex question_direct/request_user_input pins: the prompt
    is answerable WITHOUT those tools, so a text reply with rc=0 proves
    the turn completed instead of hanging on Plan-mode input.
    """
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-noask")
    env = claude_env(base, "ak-battery-cl29", sk)
    try:
        c = run_cli(
            [
                CLAUDE_BIN,
                "-p",
                "--model",
                "haiku",
                "Reply with exactly NOASK-PATH-OK and nothing else. "
                "Do not call any tools.",
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("claude model hung on no-tool turn (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("NOASK-PATH-OK")


def test_cl_single_adddir_scope():
    """Out-of-cwd Edit via --add-dir on a NON-cwd dir (GAP 9).

    Battery legs always run with cwd=ws; test_cl_single_edit covers
    Edit inside cwd (--add-dir=ws). This leg proves proxy path handling
    outside cwd: the seed lives in a sibling dir granted only via
    --add-dir and addressed by absolute path. A same-named decoy in cwd
    guards against the edit landing in the wrong root.
    """
    import subprocess

    import pytest

    base, sk = require_live()
    _, ws = battery_dirs("cl-adddir-scope")
    env = claude_env(base, "ak-battery-cl30", sk)
    other = ws.parent / "elsewhere"
    other.mkdir(parents=True, exist_ok=True)
    target = other / "cl-scope-edit.txt"
    decoy = ws / "cl-scope-edit.txt"
    prompt = (
        "Use the Edit tool to replace CLAUDE-SCOPE-V1 with "
        f"CLAUDE-SCOPE-V2 in {target}. Reply DONE."
    )
    c = None
    for _ in range(2):
        target.write_bytes(b"CLAUDE-SCOPE-V1")
        decoy.write_bytes(b"CLAUDE-SCOPE-DECOY")
        try:
            c = run_cli(
                [
                    CLAUDE_BIN,
                    "-p",
                    "--model",
                    "haiku",
                    "--allowedTools=Edit",
                    f"--add-dir={other}",
                    prompt,
                ],
                ws,
                env,
            )
        except subprocess.TimeoutExpired:
            pytest.xfail("claude CLI hung on out-of-cwd Edit (harness dynamics)")
        assert c is not None
        if c.returncode == 0 and "DONE" in (c.stdout or ""):
            break
    assert c is not None
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"CLAUDE-SCOPE-V2"
    assert decoy.read_bytes() == b"CLAUDE-SCOPE-DECOY"
    assert "DONE" in c.stdout
