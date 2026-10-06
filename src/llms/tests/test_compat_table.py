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
