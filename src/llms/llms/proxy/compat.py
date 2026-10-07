"""Opencode<->harness tool compatibility tables (pure: no pipeline/forward imports).

Same discipline as client_tools: pure helpers + tables, so the classifier
(pipeline._classify_calls, shared by the streaming fold in forward.py) and
the history-echo site (translate.to_zen_responses) can both consume them.
"""

from __future__ import annotations

CODEX_PLAIN = "codex-plain"
LUNA = "luna"
CLAUDE = "claude"
DSH = "dsh"
UNKNOWN = "unknown"


def detect_family(user_agent: str | None, ingress: str, tools: tuple = ()) -> str:
    """Harness family from downstream UA prefix, else leg+tool shape.

    UA matches on PREFIX only (``codex`` / ``claude-cli`` / ...): harness
    versions and platforms shift per release, so exact equality would
    break on every upgrade (live: ``codex_exec/0.154.0 (Debian ...)``,
    ``claude-cli/2.1.290 (external, sdk-cli)``). No UA (relays strip
    headers; scripts omit them): fall back to ingress leg plus tool
    shape — a deferred ``functions`` namespace means the luna leg, plain
    top-level function tools the codex-plain leg, messages the claude
    leg. Nothing recognizable returns ``unknown`` (callers apply only
    ``*`` all-family entries — today's behavior, never a wrong-family
    rewrite).
    """
    ua = (user_agent or "").lower()
    if ua.startswith("codex"):
        return CODEX_PLAIN
    if ua.startswith("claude"):
        return CLAUDE
    if "dsh" in ua or "deepseek" in ua:
        return DSH
    if ingress == "messages":
        return CLAUDE
    # Chat with no UA signal and nothing declared: unknown (fail open).
    # Only a positive dsh/deepseek UA claims the dsh family — the leg
    # alone is not evidence (generic OpenAI clients land here too).
    namespaced = False
    declared = False
    for t in tools or ():
        name = getattr(t, "name", "")
        if name:
            declared = True
        try:
            from llms.proxy.client_tools import _namespace_exec_desc

            if _namespace_exec_desc(t):
                namespaced = True
        except Exception:
            pass
    if namespaced:
        return LUNA
    if declared:
        return CODEX_PLAIN
    return UNKNOWN


def _shell_to_cmd(payload: dict, required: list) -> str | None:
    """shell {"command"} onto a cmd-shaped client runner, or None.

    Capability-exact: only a ``cmd`` key among the client required keys
    (same contract as client_tools._translate_genuine_args). Extra
    genuine keys (workdir/timeout/...) are dropped — the client schema
    is the contract. Pure helper.
    """
    import json as _json

    if "command" not in payload:
        return None
    if "cmd" in set(required or []):
        return _json.dumps({"cmd": payload["command"]})
    return None


def _execute_to_exec_channel(payload: dict, required: list) -> str | None:
    """execute {"code"} onto the nested exec channel, or None.

    The genuine Code Mode JS runtime and the harness exec orchestrator
    run the SAME JavaScript (live luna 2026-10-06: the model emits
    `execute {"code": "await tools.apply_patch(...)"}` instead of the
    harness's `custom_tool_call exec` channel). The inner source
    replays verbatim as the channel input — no re-derivation, no
    guess. The `required` contract is unused (the channel takes raw
    input, not JSON keys); the client-missing check in the dispatcher
    below still gates. Non-string or blank `code`: None (no guess —
    the generic steer applies).
    """
    code = payload.get("code")
    if not isinstance(code, str) or not code.strip():
        return None
    return code.strip()


def rewrap_bare_patch_exec_input(payload_text: str) -> str | None:
    """Bare apply_patch text in an exec input -> channel-ready JS, or None.

    Mechanism 13 (live luna 2026-10-07, round 5): under a natural
    prompt the model drops the `await tools.apply_patch(...)`
    wrapper and emits the patch text as the whole `exec` input
    (`'*** Begin Patch\\n*** Add File: ...'` — 15 turns running,
    `Script failed` + `SyntaxError` each time; the harness JS
    parser cannot run a bare patch). Detect the marker at the
    start (after stripping an optional single layer of matching
    quotes — the model wraps the whole input in `'...'` verbatim)
    and re-wrap it as the channel invocation. Returns None when
    the input is not a bare patch (already-wrapped JS, shell
    commands, anything else ride through untouched).
    Pure helper.
    """
    if not isinstance(payload_text, str):
        return None
    text = payload_text.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1].strip()
    if not text.startswith("*** Begin Patch"):
        return None
    if "tools.apply_patch" in text:
        return None
    return f"await tools.apply_patch({text!r})"


def _client_channel(
    owned: dict, defs: dict, family: str, client_tools: tuple = ()
) -> tuple[str, str] | None:
    """(owned key, declared name) of the file-write channel, or None.

    Nested legs (luna) execute ONLY through the `exec` JS
    orchestrator: the channel target is the owned `exec` entry.
    The gate is the deferred namespace's `exec` description in
    `client_tools` (the classifier passes the raw ToolDef tuple;
    owned/defs alone cannot tell an orchestrator `exec` from a raw
    nested spec carrying the same name).
    """
    if family == LUNA:
        from llms.proxy import client_tools as _ct

        declared = owned.get("exec")
        if declared is None:
            return None
        for tool in client_tools:
            if _ct._namespace_exec_desc(tool):
                return "exec", declared
        return None
    return None


def _write_to_apply_patch(payload: dict, required: list) -> str | None:
    """genuine `write {path, content}` onto the nested apply_patch channel.

    Family-gated (luna only — the row carries the family): synthesizes
    the Add-File marker grammar proven live 2026-10-07 (round 4):
    `*** Begin Patch\\n*** Add File: <path>\\n+<content-line>\\n***
    End Patch`. Each content line gets a `+` prefix (multi-line
    content joins with newlines); the marker/filename lines carry no
    trailing `***`. The classifier's `declared == "exec"` path feeds
    the returned patch text through `__exec_rewrite__` into the exec
    channel (see the `execute` row). Non-string path/content: None
    (no guess — the generic steer applies).
    """
    path = payload.get("path")
    content = payload.get("content")
    if not isinstance(path, str) or not path.strip():
        return None
    if not isinstance(content, str):
        return None
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    added = "\n".join(f"+{line}" for line in lines)
    return f"*** Begin Patch\n*** Add File: {path}\n{added}\n*** End Patch"


def _edit_to_apply_patch(payload: dict, required: list) -> str | None:
    """genuine `edit {path, oldString, newString}` onto apply_patch Update.

    Family-gated (luna only): synthesizes the Update-File grammar
    proven live 2026-10-07 (round 4):
    `*** Begin Patch\\n*** Update File: <path>\\n@@\\n-<old>\\n+<new>\\n***
    End Patch`. The harness applies the `old`→`new` line swap
    (unified-diff hunk from the rollout record). `replaceAll` is
    refused (None): the Update grammar addresses one hunk, and a
    global replace synthesized from a single old/new pair would
    mis-fire on repeated lines. Missing/non-string fields: None.
    """
    path = payload.get("path")
    old = payload.get("oldString")
    new = payload.get("newString")
    if not isinstance(path, str) or not path.strip():
        return None
    if not isinstance(old, str) or not isinstance(new, str):
        return None
    if not old or old == new:
        return None
    if payload.get("replaceAll"):
        return None
    old_lines = old.split("\n")
    new_lines = new.split("\n")
    if old_lines and old_lines[-1] == "":
        old_lines = old_lines[:-1]
    if new_lines and new_lines[-1] == "":
        new_lines = new_lines[:-1]
    removed = "\n".join(f"-{line}" for line in old_lines)
    added = "\n".join(f"+{line}" for line in new_lines)
    return (
        f"*** Begin Patch\n*** Update File: {path}\n"
        f"@@\n{removed}\n{added}\n*** End Patch"
    )


def _write_to_shell_redirect(payload: dict, required: list) -> str | None:
    """genuine `write {path, content}` onto a cmd-runner shell redirect.

    Family-gated (codex-plain only — the row carries the family):
    synthesizes `printf %s > <path> <<'EOF' ... EOF` proven live
    2026-10-07 (round 4, spark leg: `printf 'hello-shell\\\\n' >
    shell-write.txt` executed, file byte-exact). Heredoc with a
    quoted delimiter (no interpolation, no expansion); the content
    rides the body lines verbatim. Delimiter collision (content
    contains a line equal to the delimiter) or non-string
    path/content: None — never synthesize a command that would
    truncate or corrupt the file (the generic steer applies).
    """
    path = payload.get("path")
    content = payload.get("content")
    if not isinstance(path, str) or not path.strip():
        return None
    if not isinstance(content, str):
        return None
    if "cmd" not in set(required or []):
        return None
    delimiter = "EOF"
    for line in content.split("\n"):
        if line.strip() == delimiter:
            return None
    body = content if content.endswith("\n") else content + "\n"
    return f"printf %s > {path} <<'{delimiter}'\n{body}{delimiter}"


# Dispatch table: (genuine name, family ["*" = any], client lowered name,
# argument translator). Family-specific rows require positive detection
# (Task 1); unknown families get ``*`` rows only — never a wrong-family
# rewrite.
TO_CLIENT: tuple = (
    ("shell", "*", "exec_command", _shell_to_cmd),
    ("execute", "*", "exec", _execute_to_exec_channel),
    ("write", LUNA, "exec", _write_to_apply_patch),
    ("edit", LUNA, "exec", _edit_to_apply_patch),
    ("write", CODEX_PLAIN, "exec_command", _write_to_shell_redirect),
)


def translate_to_client(
    genuine_name: str,
    arguments: str,
    family: str,
    owned: dict,
    client_defs: dict,
    client_tools: tuple = (),
) -> tuple[str, str] | None:
    """Genuine name + usable args -> (client tool name, client args), or None.

    Returns None when untranslatable (same-name ownership wins, bad
    payload, no table row, client missing): callers fail open (pass
    through + error log), never synthesize a guessed call. Pure helper.
    """
    import json as _json

    from llms.proxy.ir import ToolDef

    lowered = genuine_name.lower() if isinstance(genuine_name, str) else ""
    # Same-name ownership wins: when the client declared the genuine name
    # itself, the owned path validates it — the translation must not
    # shadow the client's own declaration (its redirect corrects
    # arguments; a rewrite would bypass it).
    if lowered in owned:
        return None
    try:
        payload = _json.loads(arguments) if isinstance(arguments, str) else None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for genuine, fam, client_lower, convert in TO_CLIENT:
        if genuine != lowered or (fam != "*" and fam != family):
            continue
        if client_lower == "exec":
            # File-write channel rows (write/edit -> apply_patch): the
            # nested legs execute ONLY through the `exec` JS
            # orchestrator, so resolve the channel there — never a bare
            # nested name (a bare function_call fails lookup). The
            # gate is a documented exec channel in client_tools
            # (owned alone is not enough: fixture nested lists may
            # carry an `exec` spec outside the orchestrator). The
            # `execute` row is NOT channel-gated: genuine Code Mode
            # `execute {"code"}` already carries the channel input
            # verbatim (live luna 2026-10-06), so a declared `exec`
            # entry suffices.
            if genuine == "execute":
                declared = owned.get("exec")
                tool = client_defs.get("exec")
                if declared is None or not isinstance(tool, ToolDef):
                    continue
                required = (tool.parameters or {}).get("required") or []
                converted = convert(payload, required)
                if converted is not None:
                    return declared, converted
                continue
            channel = _client_channel(owned, client_defs, family, client_tools)
            if channel is None:
                continue
            _, declared = channel
            tool = client_defs.get("exec")
            required = (tool.parameters or {}).get("required") or []
            converted = convert(payload, required)
            if converted is not None:
                return declared, converted
            continue
        declared = owned.get(client_lower)
        tool = client_defs.get(client_lower)
        if declared is None or not isinstance(tool, ToolDef):
            continue
        required = (tool.parameters or {}).get("required") or []
        converted = convert(payload, required)
        if converted is not None:
            return declared, converted
    return None


def log_untranslatable(trace_id: str, name: str, family: str) -> None:
    """Fail-open breadcrumb for a name no table row covers.

    The call steers with the generic correction (not a table
    translation); this error log is the signal to add a proven 1:1
    row later (schema audit reviews these lines).
    """
    import logging as _logging

    _logging.getLogger("zen_proxy").error(
        "[%s] no compat entry: %s (family=%s) — generic steer",
        trace_id,
        name,
        family,
    )
