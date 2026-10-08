"""Hermetic: claude-leg captured schemas drive notice + translate paths.

Uses the REAL 23-tool declaration captured from claude-cli 2.1.291
(stub server, zero quota — claude-schema.jsonl) to pin:
1. build_tool_notice renders the claude tools as model-facing text
   (the outbound contract: client tools ride the systemprompt, never
   the wire on the anon tier);
2. detect_family maps the claude-cli UA + messages leg -> claude;
3. TO_CLIENT ships the four claude rows (shell/read/write/edit onto
   Bash/Read/Write/Edit — Task 4/5, proven by the capture's required
   keys); the row contract itself is pinned in test_compat_table.
"""

import json

CAPTURE = "/home/uqmm/.claude/jobs/8c5ef74f/tmp/claude-schema.jsonl"


def _claude_tools():
    from llms.proxy.ir import ToolDef

    with open(CAPTURE) as f:
        body = json.loads(f.readline())["body"]
    out = []
    for t in body["tools"]:
        out.append(
            ToolDef(
                t["name"],
                t.get("description", ""),
                t.get("input_schema", {}),
                kind=t.get("type", "function"),
            )
        )
    return tuple(out)


def test_claude_capture_drives_notice_and_family():
    from llms.proxy.client_tools import build_tool_notice, owned_tool_names
    from llms.proxy.compat import CLAUDE, TO_CLIENT, detect_family
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    tools = _claude_tools()
    assert len(tools) == 23
    # Family: claude-cli UA prefix -> claude (also messages-leg fallback).
    assert (
        detect_family("claude-cli/2.1.291 (external, sdk-cli)", "messages", tools)
        == CLAUDE
    )
    assert detect_family(None, "messages", tools) == CLAUDE
    # Ownership: Read/Write/Edit/Bash declared under own names.
    owned = owned_tool_names(tools)
    for name in ("Read", "Write", "Edit", "Bash"):
        assert name.lower() in owned, name
    # Notice renders them as model-facing text (systemprompt path).
    notice = build_tool_notice(tools, GENUINE_TOOL_NAMES)
    assert "Write" in notice and "Read" in notice
    # Claude rows ship (Task 4/5): shell/read/write/edit translate
    # onto Bash/Read/Write/Edit (row contract pinned in
    # test_compat_table; here the capture proves end-to-end wiring).
    assert {row[0] for row in TO_CLIENT if row[1] == CLAUDE} == {
        "shell",
        "read",
        "write",
        "edit",
    }
