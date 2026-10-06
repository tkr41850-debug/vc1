"""Unit-test the hermetic 1:1 translation table entries the compat layer adds.

Add cases here BEFORE touching the implementation (systematic-debugging
Phase 4: failing test first, single fix, verify). Each entry documents
the live capture or harness declaration it mirrors, and untranslatable
arms must fail open (return None), never steer or synthesize a guess.
"""

# Verbatim downstream UAs (proxy log, 2026-10-06 probes): prefix matching
# only — versions/platforms shift per release, so exact equality would
# break on every harness upgrade.
CODEX_UA = "codex_exec/0.154.0 (Debian 13.0.0; x86_64) screen (codex_exec; 0.154.0)"
CLAUDE_UA = "claude-cli/2.1.290 (external, sdk-cli)"


def test_detect_family_prefers_ua_prefix_falls_back_to_shape():
    from llms.proxy.compat import detect_family
    from llms.proxy.ir import ToolDef

    # UA prefix wins when present (verbatim strings above).
    assert detect_family(CODEX_UA, "responses", ()) == "codex-plain"
    assert detect_family(CODEX_UA + " extra-suffix", "responses", ()) == "codex-plain"
    assert detect_family(CLAUDE_UA, "messages", ()) == "claude"
    # No UA: responses + functions namespace -> luna; top-level
    # function tools -> codex-plain; messages leg -> claude.
    luna_ns = ToolDef(
        "functions",
        "",
        {},
        kind="namespace",
        options={"tools": [{"name": "exec", "description": "Run JS"}]},
    )
    assert detect_family("", "responses", (luna_ns,)) == "luna"
    plain = ToolDef("exec_command", "run", {"required": ["cmd"]})
    assert detect_family("", "responses", (plain,)) == "codex-plain"
    assert detect_family(None, "messages", ()) == "claude"
    # Nothing recognizable -> unknown (today's behavior, *-only entries).
    assert detect_family("", "responses", ()) == "unknown"


def test_dispatch_shell_to_cmd_runner_all_families():
    # shell{"command"} -> exec_command{"cmd"} on every family owning a
    # cmd-shaped runner (live spark leg, zero-steer turn 2026-10-06).
    from llms.proxy.compat import translate_to_client
    from llms.proxy.ir import ToolDef

    runner = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    assert translate_to_client(
        "shell", '{"command": "cat /tmp/hi"}', "codex-plain", owned, defs
    ) == ("exec_command", '{"cmd": "cat /tmp/hi"}')
    assert translate_to_client(
        "shell", '{"command": "cat /tmp/hi"}', "unknown", owned, defs
    ) == ("exec_command", '{"cmd": "cat /tmp/hi"}')
    # Missing payload key or no cmd-runner: None (fail open, never guess).
    assert (
        translate_to_client(
            "shell", '{"workdir": "/tmp"}', "codex-plain", owned, defs
        )
        is None
    )
    assert (
        translate_to_client("shell", '{"command": "x"}', "codex-plain", {}, {})
        is None
    )
    # Same-name ownership wins: the client declared `shell` itself, so the
    # owned argument-correction path applies, not a rewrite.
    own_shell = ToolDef(
        "Shell",
        "mine",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    assert (
        translate_to_client(
            "shell",
            '{"command": "x"}',
            "codex-plain",
            {"shell": "Shell"},
            {"shell": own_shell},
        )
        is None
    )
