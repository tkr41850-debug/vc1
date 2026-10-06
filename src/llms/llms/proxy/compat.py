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


# Dispatch table: (genuine name, family ["*" = any], client lowered name,
# argument translator). Family-specific rows require positive detection
# (Task 1); unknown families get ``*`` rows only — never a wrong-family
# rewrite.
TO_CLIENT: tuple = (
    ("shell", "*", "exec_command", _shell_to_cmd),
)


def translate_to_client(
    genuine_name: str,
    arguments: str,
    family: str,
    owned: dict,
    client_defs: dict,
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

    The call passes through unchanged; this error log is the signal to
    add a proven 1:1 row later (schema audit reviews these lines).
    """
    import logging as _logging

    _logging.getLogger("zen_proxy").error(
        "[%s] no compat entry: %s (family=%s) — passing through",
        trace_id,
        name,
        family,
    )
