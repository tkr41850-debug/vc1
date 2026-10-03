"""Client-tool compatibility helpers (pure: no pipeline/forward imports)."""

from __future__ import annotations

import json

from llms.proxy.ir import ToolDef


def owned_tool_names(tools: tuple[ToolDef, ...]) -> dict[str, str]:
    owned: dict[str, str] = {}
    for t in tools:
        if t.name and t.name.lower() not in owned:
            owned[t.name.lower()] = t.name
    return owned


def convert_call_name(name: str, owned: dict[str, str]) -> str:
    return owned.get(name.lower(), name)


def missing_required_keys(arguments: str, tool: ToolDef) -> list[str] | None:
    try:
        parsed = json.loads(arguments)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    required = (tool.parameters or {}).get("required") or []
    return [k for k in required if k not in parsed]


def build_tool_notice(tools: tuple[ToolDef, ...], genuine_names) -> str:
    genuine_lower = {str(n).lower() for n in genuine_names}
    lines = [
        "Client tools (prefer these; where a name collides with a default tool, "
        "use the client tool and its parameter shape):"
    ]
    for t in tools:
        if not t.name:
            continue
        marker = (
            " (override — prefer over same-named default tool)"
            if t.name.lower() in genuine_lower
            else ""
        )
        lines.append(
            f"- '{t.name}'{marker}: {t.description}\n"
            f"  Parameters: {json.dumps(t.parameters or {})}"
        )
    return "\n".join(lines) if len(lines) > 1 else ""
