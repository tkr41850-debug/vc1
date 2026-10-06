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
