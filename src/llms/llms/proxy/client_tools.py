"""Client-tool compatibility helpers (pure: no pipeline/forward imports)."""

from __future__ import annotations

import json

from llms.proxy.ir import ToolDef


def owned_tool_names(tools: tuple[ToolDef, ...]) -> dict[str, str]:
    owned: dict[str, str] = {}
    for t in tools:
        # Last declaration wins (matches the defs map in _classify_calls
        # callers): casing convert and required-keys validation agree on
        # which duplicate the client meant.
        if t.name:
            owned[t.name.lower()] = t.name
        # Deferred-namespace nested tools dispatch by name alone (no
        # namespace on the wire), so they are owned too: description-
        # parsed entries first (they carry the call signatures), then
        # raw nested specs not documented in the exec description.
        exec_desc = _namespace_exec_desc(t)
        if exec_desc:
            for nname, _ in _nested_exec_tools(exec_desc):
                owned[nname.lower()] = nname
            nested = t.options.get("tools")
            if isinstance(nested, list):
                for n in nested:
                    if not isinstance(n, dict) or not n.get("name"):
                        continue
                    nname = str(n.get("name"))
                    if nname.lower() not in owned:
                        owned[nname.lower()] = nname
    return owned


def split_call_name(name: str) -> tuple[str | None, str]:
    """Split a model call name into (namespace, bare name).

    Codex flattens namespaced tools with a `.` separator
    (`flat_tool_name` in the harness: `{namespace}.{name}` for
    non-default namespaces, bare name otherwise — live:
    `default.exec_command`). The harness applies its own defaulting
    (`with_default_namespace`), so strip the prefix for ownership,
    routing, and casing decisions and re-attach it on lookup failure.
    A name without `.` has no namespace. Returns ("", name) for
    non-string input so callers steer it as undeclared. Splits on the
    FIRST dot only (namespaces never nest).
    """
    if not isinstance(name, str) or not name:
        return "", name if isinstance(name, str) else ""
    head, sep, tail = name.partition(".")
    if sep and head and tail:
        return head, tail
    return "", name


def convert_call_name(name: str, owned: dict[str, str]) -> str:
    # Code-mode `ns__name` calls convert on the bare name (the harness
    # fills its default namespace when absent, and `__` only nests a
    # NON-default one).
    _, bare = split_call_name(name)
    return owned.get(bare.lower(), name)


def _is_freeform_decl(decl: str) -> bool:
    """True when an exec nested-tool declaration takes a bare string input.

    Freeform nested tools declare a single `(input: string)` parameter
    (live luna shape: `apply_patch(input: string)`); function-shaped
    nested tools declare an `args` object (`exec_command(args: {...})`).
    The harness parses a Custom payload's input as raw text, so only
    the freeform shape dispatches as `custom_tool_call`. Matched on the
    declaration line only — never on prose (a description mentioning
    "input" must not flip the route).
    """
    import re

    return re.search(r"\(\s*input\s*:\s*string\s*\)", decl) is not None


def nested_tool_defs(tools: tuple[ToolDef, ...]) -> dict[str, ToolDef]:
    """Lowered nested-tool name -> synthetic ToolDef for validation.

    Deferred-namespace nested tools have no top-level ToolDef, so the
    required-keys check needs a stand-in. Description-parsed entries
    synthesize required keys from the declaration: a required `args`
    object (exec_command-style: `exec_command(args: {...})` with at
    least one non-optional field) requires `args`, so a bare call
    steers with an argument correction instead of passing through
    with unusable arguments. Freeform `(input: string)` entries carry
    no required keys — the raw input IS the payload. Raw nested specs
    with a parameters object keep it.
    """
    import re

    defs: dict[str, ToolDef] = {}
    for t in tools:
        exec_desc = _namespace_exec_desc(t)
        if not exec_desc:
            continue
        for nname, entry in _nested_exec_tools(exec_desc):
            # Required `args`: the declaration calls the tool with an
            # args object having at least one required (non-`?`) field.
            # get_goal-style `args: {}` (all optional/empty) requires
            # nothing — a bare call is usable.
            required: list[str] = []
            m = re.search(
                rf"{re.escape(nname)}\s*\(\s*args\s*:\s*\{{(.*?)\}}\s*\)",
                entry,
                re.DOTALL,
            )
            if m:
                fields = m.group(1)
                if re.search(r"^\s*[A-Za-z0-9_]+(?!\?)\s*:", fields, re.MULTILINE):
                    required = ["args"]
            defs[nname.lower()] = ToolDef(
                nname,
                "",
                {"type": "object", "required": required} if required else {},
            )
        nested = t.options.get("tools")
        if isinstance(nested, list):
            for n in nested:
                if not isinstance(n, dict) or not n.get("name"):
                    continue
                nname = str(n.get("name"))
                if nname.lower() in defs:
                    continue
                params = n.get("parameters")
                defs[nname.lower()] = ToolDef(
                    nname,
                    str(n.get("description") or ""),
                    dict(params) if isinstance(params, dict) else {},
                    kind=str(n.get("type", "function")),
                )
    return defs


def dispatchable_names(tools: tuple[ToolDef, ...]) -> dict[str, tuple[str, str]]:
    """Lowered model-callable name -> (dispatch item type, declared name).

    The harness dispatches by item *type* as well as name (codex source:
    router `build_tool_call` emits a Function vs Custom payload by the
    response item type, and registry `matches_kind` fatals a mismatch).
    A top-level `function` tool dispatches as a `function_call`. The
    deferred namespace's nested tools come from the `exec` description
    (parsed via _nested_exec_tools — the namespace options carry only
    the raw nested specs): an `(input: string)` freeform declaration
    dispatches as `custom_tool_call` with raw-string input, an `args`
    object declaration as `function_call` with JSON arguments. Nested
    items carry no namespace (the harness fills its default when
    absent). Freeform top-level custom (e.g. apply_patch on the spark
    leg) dispatches as `custom_tool_call` with raw-string input. Last
    declaration wins, matching owned_tool_names.
    """
    out: dict[str, tuple[str, str]] = {}
    for t in tools:
        exec_desc = _namespace_exec_desc(t)
        if exec_desc:
            # Deferred exec namespace: the nested tools dispatch by
            # name alone (no namespace on the wire). The container
            # itself is not model-callable — do not route it.
            for nname, entry in _nested_exec_tools(exec_desc):
                item_type = (
                    "custom_tool_call" if _is_freeform_decl(entry) else "function_call"
                )
                out[nname.lower()] = (item_type, nname)
            # Raw nested specs not documented in the exec description
            # (e.g. luna's plain `wait` function) route by declared
            # type. Description-parsed entries win on collision.
            nested = t.options.get("tools")
            if isinstance(nested, list):
                for n in nested:
                    if not isinstance(n, dict) or not n.get("name"):
                        continue
                    nname = str(n.get("name"))
                    if nname.lower() in out:
                        continue
                    item_type = (
                        "function_call"
                        if n.get("type", "function") == "function"
                        else "custom_tool_call"
                    )
                    out[nname.lower()] = (item_type, nname)
        elif t.kind == "custom":
            # Freeform top-level custom (e.g. apply_patch on the spark
            # leg): Custom payload only — must not fall through to the
            # function default below (registry `matches_kind` fatals a
            # Custom handler on a Function payload).
            if t.name:
                out[t.name.lower()] = ("custom_tool_call", t.name)
        elif t.name:
            out[t.name.lower()] = ("function_call", t.name)
    return out


def missing_required_keys(arguments: str, tool: ToolDef) -> list[str] | None:
    if not isinstance(arguments, str) or not arguments.strip():
        # Blank arguments carry no keys at all: every required key is
        # missing (not "unparseable" — the model emitted a no-arg call,
        # typically for a tool whose schema demands arguments). Return
        # the full missing list so the redirect names the keys instead
        # of the vaguer not-valid-JSON text.
        required = (tool.parameters or {}).get("required") or []
        return list(required)
    try:
        parsed = json.loads(arguments)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    required = (tool.parameters or {}).get("required") or []
    return [k for k in required if k not in parsed]


def build_tool_redirect(
    call_name: str, arguments: str, owned: dict[str, str], defs: dict
) -> str | None:
    """Actionable redirect for a steered call, or None when undeclared.

    Owned-but-invalid calls (the client declared this name but the
    arguments miss required keys or are not valid JSON) get a correction:
    the tool exists, the *arguments* were wrong, with the missing keys
    and the declared parameter shape named. Undeclared names return None
    (callers fall back to the not-available text): claiming a missing
    tool "exists" would contradict the tool list the model was given.
    Pure helper (shared by the synthesize and streaming steer paths).
    Splits code-mode `ns__name` first (same ownership key as the
    classifier), so the correction names the declared tool.
    """
    from llms.proxy.ir import ToolDef

    _, bare = split_call_name(call_name) if isinstance(call_name, str) else ("", None)
    lowered = bare.lower() if isinstance(bare, str) else None
    if lowered is None or lowered not in owned:
        return None
    tool = defs.get(lowered)
    if not isinstance(tool, ToolDef):
        return None
    missing = missing_required_keys(
        arguments if isinstance(arguments, str) else "", tool
    )
    declared = owned[lowered]
    params = json.dumps(tool.parameters or {})
    if missing:
        return (
            f"Tool '{declared}' was called with the wrong arguments "
            f"(missing required: {', '.join(missing)}). "
            f"Its parameter shape is: {params}. "
            f"Retry the call with corrected arguments."
        )
    return (
        f"Tool '{declared}' was called with arguments that are not valid "
        f"JSON. Its parameter shape is: {params}. "
        f"Retry the call with corrected arguments."
    )


def rewrite_steered_call(
    call: dict, arguments: str, route: dict[str, tuple[str, str]]
) -> dict | None:
    """Rewrite one steered call to its harness-dispatchable form, or None.

    The harness dispatches by response item *type* as well as name:
    nested `custom` tools (exec, and apply_patch on the luna leg) only
    execute as `custom_tool_call` items (registry `matches_kind` fatals
    a Custom handler on a Function payload, and vice versa). A steered
    call whose lowered name the client declared under a namespace is
    therefore re-typed to the declared item type with the raw input
    preserved — the retry then dispatches instead of failing lookup.
    Top-level function tools (route "function_call") need no rewrite:
    returns None so the caller replays the call verbatim. Splits
    code-mode `ns__name` first (same ownership key as the classifier).
    Pure helper (shared by the synthesize and streaming steer paths).
    """
    name = call.get("name")
    _, bare = split_call_name(name) if isinstance(name, str) else ("", None)
    lowered = bare.lower() if isinstance(bare, str) else None
    if lowered is None or lowered not in route:
        return None
    item_type, declared = route[lowered]
    if item_type != "custom_tool_call":
        return None
    out = dict(call)
    out["type"] = "custom_tool_call"
    out["name"] = declared
    # Custom input is a raw string (JS source for exec, patch text for
    # apply_patch) — never JSON arguments. Preserve verbatim.
    if "input" not in out:
        out["input"] = arguments if isinstance(arguments, str) else ""
    out.pop("arguments", None)
    return out


def _nested_exec_tools(description: str) -> list[tuple[str, str]]:
    """(name, declaration) pairs parsed from an exec custom-tool description.

    Codex's deferred `functions` namespace carries a single `custom`
    tool, `exec`: a JS orchestrator whose description documents the
    nested tools it can call (`exec_command`, `apply_patch`, ...) as
    `### \\`<name>\\`` sections each with an `exec tool declaration:`
    TypeScript block. The notice cannot re-attach the 10KB description
    verbatim, so extract the per-tool call signatures the model needs
    to invoke them. Returns [] when the shape is unrecognized (fail
    soft — the caller falls back to the raw description).
    """
    import re

    sections = re.split(r"^### `([^`]+)`\s*$", description, flags=re.MULTILINE)
    # sections[0] is preamble; then (name, body) pairs.
    out: list[tuple[str, str]] = []
    for i in range(1, len(sections) - 1, 2):
        name = sections[i].strip()
        body = sections[i + 1]
        m = re.search(r"exec tool declaration:\s*```ts\s*(.*?)```", body, re.DOTALL)
        decl = m.group(1).strip() if m else ""
        # First prose line as the short description.
        prose = ""
        for line in body.strip().splitlines():
            line = line.strip()
            if line and not line.startswith("exec tool declaration"):
                prose = line
                break
        if name:
            out.append((name, f"{prose}\n{decl}" if decl else prose))
    return out


def build_tool_notice(tools: tuple[ToolDef, ...], genuine_names) -> str:
    genuine_lower = {str(n).lower() for n in genuine_names}
    named = [t for t in tools if t.name]
    if not named:
        return ""
    lines = [
        (
            "Your harness can only execute the tools listed below — call "
            "them by these exact names with these exact parameter shapes. "
            "Any other tool name will fail in your harness: in particular, "
            "the default tools read, shell, write, edit, glob, grep, "
            "skill, subagent, webfetch, websearch, execute, and question "
            "are NOT available here unless listed below — never call them."
        ),
    ]
    # HOW-TO-INVOKE FIRST: the model acts on the first actionable line.
    # The deferred namespace's nested tools run INSIDE the exec JS
    # orchestrator (`await tools.<name>(...)` as the exec `input`
    # string) — a bare `function_call` item named `exec_command`
    # reaches the harness as an undeclared top-level name and fails
    # lookup. State this before the per-tool entries so the model
    # never has to infer the channel from the signatures alone.
    for t in named:
        exec_desc = _namespace_exec_desc(t)
        if exec_desc and _nested_exec_tools(exec_desc):
            lines.append(
                "To run a nested tool, emit ONE `custom_tool_call` item "
                "named `exec` whose `input` is JavaScript calling it on "
                "the `tools` object — e.g. to run a shell command: "
                '`{"type": "custom_tool_call", "name": "exec", '
                '"input": "await tools.exec_command({cmd: '
                '"cat /tmp/codex-cap-ws/probe.txt"})"}`. Never emit '
                "a bare `function_call` named `exec_command`, "
                "`apply_patch`, or any other nested name."
            )
            break
    for t in named:
        marker = (
            " (override — prefer over same-named default tool)"
            if t.name.lower() in genuine_lower
            else ""
        )
        nested = _namespace_exec_entries(t)
        if nested is not None:
            # Codex deferred `functions` namespace: the model invokes
            # the nested tools (exec_command, apply_patch, ...) through
            # the `exec` JS orchestrator — list those with their call
            # signatures instead of the bare container (whose empty
            # description and {} parameters read as "nothing available
            # here").
            lines.append(
                f"- '{t.name}'{marker}: invokes the nested tools below "
                "through `exec` (see HOW above) — it is not itself a "
                "callable tool:"
            )
            lines.extend(nested)
            continue
        lines.append(
            f"- '{t.name}'{marker}: {t.description}\n"
            f"  Parameters: {json.dumps(t.parameters or {})}"
        )
    return "\n".join(lines)


def _namespace_exec_desc(t: ToolDef) -> str:
    """The exec custom-tool description for a namespace, or "".

    Shared lookup for the notice renderer and the dispatch route: both
    key off the deferred `functions`-style namespace wrapping an `exec`
    custom tool whose description documents the nested tools.
    """
    if t.kind != "namespace":
        return ""
    nested = t.options.get("tools")
    if not isinstance(nested, list):
        return ""
    for n in nested:
        if isinstance(n, dict) and n.get("name") == "exec":
            return str(n.get("description") or "")
    return ""


def _namespace_exec_entries(t: ToolDef) -> list[str] | None:
    """Notice lines for a namespace wrapping an exec custom tool.

    Returns None when `t` is not that shape (caller renders normally).
    """
    exec_desc = _namespace_exec_desc(t)
    if not exec_desc:
        return None
    entries = _nested_exec_tools(exec_desc)
    if not entries:
        return None
    return [f"  - '{name}': {decl}" for name, decl in entries]


def has_nested_exec_channel(tools: tuple[ToolDef, ...]) -> bool:
    """True when a deferred namespace wraps a documented exec orchestrator.

    Gates redirect guidance that names the `custom_tool_call`/`exec`
    channel: only legs whose harness actually exposes it (luna-style
    `functions` namespace with parsed nested tools) should teach it —
    on plain function legs (spark) that text contradicts the usable
    tool list in the same message. Same shape condition as the notice
    HOW-TO line.
    """
    for t in tools:
        exec_desc = _namespace_exec_desc(t)
        if exec_desc and _nested_exec_tools(exec_desc):
            return True
    return False
