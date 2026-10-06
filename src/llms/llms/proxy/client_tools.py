"""Client-tool compatibility helpers (pure: no pipeline/forward imports)."""

from __future__ import annotations

import json

from llms.proxy.ir import ToolDef


def owned_tool_names(tools: tuple[ToolDef, ...]) -> dict[str, str]:
    owned: dict[str, str] = {}
    for t in tools:
        # Last declaration wins (matches the defs map in _classify_calls
        # callers): casing convert and required-keys validation agree on
        # which duplicate the client meant. Code-mode `ns.name`
        # declarations ALSO own the bare name: the harness dispatches
        # by ToolName with its own defaulting (`with_default_namespace`
        # fills `default` when absent), so a bare `exec_command` call
        # resolves against a `default.exec_command` declaration. Keying
        # only the dotted form leaves the bare call unowned and every
        # such turn steers forever as undeclared (live spark leg: the
        # harness declares bare `exec_command` while the model emits
        # `default.exec_command` — the reverse direction of the same
        # asymmetry — so both keyings must resolve). The dotted form
        # stays too (exact-name match first).
        if t.name:
            owned[t.name.lower()] = t.name
            _, bare = split_call_name(t.name)
            if bare and bare.lower() not in owned:
                owned[bare.lower()] = t.name
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
    synthesize required keys from the declaration's INNER arg fields:
    `exec_command(args: {cmd: string; ...})` requires `cmd` — the wire
    key the harness actually validates (`await
    tools.exec_command({cmd: ...})`; the `args:` wrapper is a TS
    declaration artifact, never a wire key). Requiring the wrapper
    name instead rejects every valid call (`missing: ['args']` on
    `{"cmd": ...}`) and steers forever with a wrong-argument
    correction (live luna leg). get_goal-style `args: {}` (all
    optional/empty) requires nothing — a bare call is usable.
    Freeform `(input: string)` entries carry no required keys — the
    raw input IS the payload. Raw nested specs with a parameters
    object keep it.
    """
    import re

    defs: dict[str, ToolDef] = {}
    for t in tools:
        exec_desc = _namespace_exec_desc(t)
        if not exec_desc:
            continue
        for nname, entry in _nested_exec_tools(exec_desc):
            # Required inner fields: non-optional (`?`-less) keys of
            # the args object. Comment lines (`// ...`) never match
            # the field pattern (no trailing colon-atom).
            required: list[str] = []
            m = re.search(
                rf"{re.escape(nname)}\s*\(\s*args\s*:\s*\{{(.*?)\}}\s*\)",
                entry,
                re.DOTALL,
            )
            if m:
                required = [
                    name
                    for name, optional in re.findall(
                        r"^\s*([A-Za-z0-9_]+)(\?)?\s*:",
                        m.group(1),
                        re.MULTILINE,
                    )
                    if not optional
                ]
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
            # Both the declared form AND the bare name route: same
            # harness-defaulting asymmetry as owned_tool_names (a dotted
            # `default.exec_command` declaration also serves bare
            # `exec_command` calls). The declared form routes first;
            # the bare key only fills when absent.
            out[t.name.lower()] = ("function_call", t.name)
            _, _bare = split_call_name(t.name)
            if _bare and _bare.lower() not in out:
                out[_bare.lower()] = ("function_call", t.name)
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


def _minimal_arg_example(tool) -> str:
    """One-line argument example naming only the required keys.

    The full parameter schema (~1.7KB for exec_command) never converted
    the model (live spark leg: two verbatim `default.exec_command {}`
    repeats after the full-shape correction, then budget exhaustion).
    The notice already carries the full shape — the correction needs
    the missing key plus a minimal retry sketch, not the schema again.
    Placeholder values only; the model fills real ones.
    """
    required = (tool.parameters or {}).get("required") or []
    props = (tool.parameters or {}).get("properties") or {}
    bits = []
    for key in required:
        kind = (props.get(key) or {}).get("type", "string")
        bits.append(f'"{key}": "<{kind}>"')
    return "{" + ", ".join(bits) + "}"


def build_tool_redirect(
    call_name: str,
    arguments: str,
    owned: dict[str, str],
    defs: dict,
    genuine_names: tuple = (),
) -> str | None:
    """Actionable redirect for a steered call, or None when undeclared.

    Owned-but-invalid calls (the client declared this name but the
    arguments miss required keys or are not valid JSON) get a correction:
    the tool exists, the *arguments* were wrong, naming the exact
    emitted name (the model binds the correction to its own call —
    live: the bare-name correction never converted a
    `default.exec_command` emitter), the missing keys, and a minimal
    retry sketch. The full parameter schema is deliberately NOT
    repeated: it runs ~1.7KB, never converted the model, and the
    notice already carries it. Undeclared names return None (callers
    fall back to the not-available text): claiming a missing tool
    "exists" would contradict the tool list the model was given.
    Pure helper (shared by the synthesize and streaming steer paths).
    Splits code-mode `ns__name` first (same ownership key as the
    classifier), so the correction names the declared tool.
    Shell-runner calls with the `shell` alias sanctioned (genuine_names
    carries upstream `shell` and the client owns a cmd-shaped runner)
    redirect to the ALIAS, not the client name: the client name has no
    wire schema (outbound is genuine-12-only), so a "retry as
    `exec_command` with {cmd}" names keys the model cannot fill from
    (live spark A/B, 2026-10-06: same notice, instructions naming
    `shell`+`command` filled cleanly with zero steers while
    `exec_command`+`cmd` emitted `{}` x3 and failed closed).
    """
    from llms.proxy.ir import ToolDef

    _, bare = split_call_name(call_name) if isinstance(call_name, str) else ("", None)
    lowered = bare.lower() if isinstance(bare, str) else None
    if lowered is None or lowered not in owned:
        return None
    tool = defs.get(lowered)
    if not isinstance(tool, ToolDef):
        return None
    # Name the exact emitted form, not the bare declared name: the
    # correction must read as a reply to the call the model made.
    emitted = call_name if isinstance(call_name, str) else bare
    declared = owned[lowered]
    genuine_lower = {str(n).lower() for n in genuine_names}
    for client_lower, alias, alias_key in _ALIASES:
        if lowered != client_lower or alias not in genuine_lower:
            continue
        if tool.kind != "function" or not _is_cmd_shaped(tool):
            continue
        return (
            f"Tool '{emitted}' was called without its required "
            f"argument(s): {alias_key}. "
            f"Retry as '{alias}' with arguments like "
            f'{{"{alias_key}": "cat /tmp/hi"}}. (Your harness runs '
            "it as its own shell tool.)"
        )
    missing = missing_required_keys(
        arguments if isinstance(arguments, str) else "", tool
    )
    example = _minimal_arg_example(tool)
    if missing:
        return (
            f"Tool '{emitted}' was called without its required "
            f"argument(s): {', '.join(missing)}. "
            f"Retry as '{declared}' with arguments like {example}."
        )
    return (
        f"Tool '{emitted}' was called with arguments that are not valid "
        f"JSON. Retry as '{declared}' with arguments like {example}."
    )


def steer_to_equivalent(
    call_name: str,
    arguments: str,
    owned: dict[str, str],
    defs: dict,
    genuine_names: tuple = (),
) -> str | None:
    """Directed correction for an undeclared genuine-overlay name, or None.

    The model was offered the genuine tool upstream but the client
    exposes the capability under its own name (live: spark `read` —
    the harness has no file reader, only `exec_command`). The generic
    not-available redirect never converted the model (it re-emitted
    `read`, then `shell`, then gave up client-side). When the payload
    maps onto exactly one client tool's required keys, name that tool
    and its argument shape directly: "don't call read — call
    exec_command with {cmd}". Returns None when the name is owned
    (the argument-correction path owns it), unmapped, or ambiguous
    (two client tools match — a guess would misroute; the generic
    list lets the model choose). Pure helper (shared by the
    synthesize and streaming steer paths).
    """
    from llms.proxy.ir import ToolDef

    _, bare = split_call_name(call_name) if isinstance(call_name, str) else ("", None)
    lowered = bare.lower() if isinstance(bare, str) else None
    if lowered is None or lowered not in ("read", "shell", "write", "edit"):
        return None
    if lowered in owned:
        return None
    try:
        payload = json.loads(arguments) if isinstance(arguments, str) else None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if lowered == "read":
        # No client tool reads files by path (a shared `path` key is
        # not a file reader — see _translate_genuine_args), but a
        # shell-runner serves the capability via `cat`: direct the
        # model there with the command built. Ambiguous (zero or two+
        # cmd-runners) or pathless: no guess, generic list applies.
        return _read_via_shell_redirect(bare, payload, owned, defs, genuine_names)
    if lowered in ("write", "edit"):
        # No directed equivalent: the capability (file creation/editing)
        # has no same-shape client tool — on nested legs only
        # apply_patch serves it, and only via the exec orchestrator
        # channel (a bare function_call fails lookup), which a
        # "retry as 'apply_patch'" sentence would mis-teach. Return
        # None so the generic list + exec-channel guidance applies
        # (it names apply_patch AND the custom_tool_call form).
        return None
    # Upstream `shell` with the alias sanctioned: steer onto the alias
    # itself (it rides the wire with its schema), not the client runner
    # name (notice text alone — the model cannot fill from it; live
    # spark A/B 2026-10-06). Falls through to the client-name match
    # below only when unsanctioned.
    genuine_lower = {str(n).lower() for n in genuine_names}
    if lowered == "shell":
        for client_lower, alias, alias_key in _ALIASES:
            if alias not in genuine_lower:
                continue
            for owned_lower, declared in owned.items():
                tool = defs.get(owned_lower)
                if (
                    not isinstance(tool, ToolDef)
                    or tool.kind != "function"
                    or not _is_cmd_shaped(tool)
                    or owned_lower != client_lower
                ):
                    continue
                if alias_key not in payload:
                    continue
                return (
                    f"Tool '{bare}' is not available in this session — use "
                    f"'{alias}' instead with these arguments: "
                    + json.dumps({alias_key: payload[alias_key]})
                    + f". Retry the call as '{alias}'."
                )
    matches: list[tuple[str, str]] = []
    for client_lower, declared in owned.items():
        tool = defs.get(client_lower)
        if not isinstance(tool, ToolDef):
            continue
        required = (tool.parameters or {}).get("required") or []
        translated = _translate_genuine_args(lowered, payload, required)
        if translated is not None:
            matches.append((declared, translated))
    if len(matches) != 1:
        return None
    declared, translated = matches[0]
    return (
        f"Tool '{bare}' is not available in this session — use "
        f"'{declared}' instead with these arguments: {translated}. "
        f"Retry the call as '{declared}'."
    )


def _read_via_shell_redirect(
    bare: str,
    payload: dict,
    owned: dict[str, str],
    defs: dict,
    genuine_names: tuple = (),
) -> str | None:
    """'read {path}' -> 'run shell {command: cat path}', or None.

    The client has no file reader, but a `cmd`-shaped shell runner
    serves the capability via `cat` — reached through the upstream
    `shell` alias (single quotes avoid the rewritten-command confusion
    of nested double quotes): the `{cmd: ...}` text is display-only
    on plain legs (no wire schema), while `shell`+`command` rides the
    wire and fills (live spark A/B 2026-10-06). Falls back to the
    client runner name only when the alias is unsanctioned. Exactly one
    such runner must exist — zero or two+ means no guess (the generic
    tool list applies). The path is the model's own payload string,
    passed through as the cat operand.
    """
    import shlex as _shlex

    from llms.proxy.ir import ToolDef

    if "path" not in payload or not isinstance(payload["path"], str):
        return None
    genuine_lower = {str(n).lower() for n in genuine_names}
    runners: list[str] = []
    for client_lower, declared in owned.items():
        tool = defs.get(client_lower)
        if not isinstance(tool, ToolDef):
            continue
        required = set((tool.parameters or {}).get("required") or [])
        if required == {"cmd"} or "cmd" in required:
            runners.append(declared)
    if len(runners) != 1:
        return None
    quoted = _shlex.quote(payload["path"])
    for client_lower, alias, alias_key in _ALIASES:
        runner = runners[0]
        tool = defs.get(runner.lower())
        if (
            alias in genuine_lower
            and isinstance(tool, ToolDef)
            and tool.kind == "function"
            and _is_cmd_shaped(tool)
            and runner.lower() == client_lower
        ):
            return (
                f"Tool '{bare}' is not available in this session — to read "
                f"a file, use '{alias}' instead with these arguments: "
                + json.dumps({alias_key: f"cat {quoted}"})
                + f". Retry the call as '{alias}'."
            )
    return (
        f"Tool '{bare}' is not available in this session — to read a "
        f"file, use '{runners[0]}' instead with these arguments: "
        + json.dumps({"cmd": f"cat {quoted}"})
        + f". Retry the call as '{runners[0]}'."
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


def display_tool_names(tools: tuple[ToolDef, ...]) -> list[str]:
    """Model-callable names for redirect lists (excludes containers).

    The redirect's "use one of these tools instead" list must name only
    what the model can actually emit: nested tools dispatch by bare
    name, the deferred `functions`-style namespace container itself is
    not model-callable (a bare call to it fails lookup). Listing the
    container (live: "Use one of these tools instead: functions, wait")
    taught the model a name it cannot use. Duplicates collapse;
    insertion order is preserved (callers sort).
    """
    names: list[str] = []
    seen: set[str] = set()
    for t in tools:
        exec_desc = _namespace_exec_desc(t)
        if exec_desc:
            for nname, _ in _nested_exec_tools(exec_desc):
                if nname.lower() not in seen:
                    seen.add(nname.lower())
                    names.append(nname)
            nested = t.options.get("tools")
            if isinstance(nested, list):
                for n in nested:
                    if not isinstance(n, dict) or not n.get("name"):
                        continue
                    nname = str(n.get("name"))
                    if nname.lower() not in seen:
                        seen.add(nname.lower())
                        names.append(nname)
            continue
        if t.name and t.name.lower() not in seen:
            seen.add(t.name.lower())
            names.append(t.name)
    return names


def translate_genuine_call(
    call_name: str,
    arguments: str,
    owned: dict[str, str],
    defs: dict,
    family: str = "unknown",
) -> tuple[str, str] | None:
    """Genuine name + usable args -> (client tool name, client arguments).

    The model emits genuine-overlay names it was offered upstream
    (shell/write/edit/...) with valid arguments — steering those
    burns a turn teaching names the client declared under its OWN
    names (spark: `exec_command` with `cmd`) for the same capability.
    When the client declared an equivalent tool and the arguments
    carry the genuine tool's payload, rewrite instead of steering:
    `shell` {"command": ...} -> client `exec_command`
    {"cmd": ...}. Returns None when there is no equivalent (caller
    falls back to the steer path). `read` never translates (see
    _translate_genuine_args): a shared `path` key is not a file
    reader. Splits code-mode `ns__name` first (same ownership key
    as the classifier). `family` selects family-specific compat rows
    (unknown families get `*` all-family rows only — never a
    wrong-family rewrite); the explicit table wins, the legacy
    client-scan loop below only covers pairs the table lacks. Pure
    helper.
    """
    from llms.proxy.ir import ToolDef

    lowered = call_name.lower() if isinstance(call_name, str) else ""
    if lowered not in ("read", "shell", "write", "edit"):
        return None
    # Same-name ownership wins: when the client declared the genuine
    # name itself (casing-insensitive), the owned path validates it —
    # the translation must not shadow the client's own declaration
    # (its redirect corrects arguments; a rewrite would bypass it).
    if lowered in owned:
        return None
    try:
        from llms.proxy.compat import translate_to_client as _compat_translate
    except Exception:
        _compat_translate = None
    if _compat_translate is not None:
        try:
            compat = _compat_translate(lowered, arguments, family, owned, defs)
        except Exception:
            compat = None
        if compat is not None:
            return compat
    try:
        import json as _json

        payload = _json.loads(arguments) if isinstance(arguments, str) else None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for client_lower, declared in owned.items():
        tool = defs.get(client_lower)
        if not isinstance(tool, ToolDef):
            continue
        required = (tool.parameters or {}).get("required") or []
        translated = _translate_genuine_args(lowered, payload, required)
        if translated is None:
            continue
        return declared, translated
    return None


def _translate_genuine_args(genuine: str, payload: dict, required: list) -> str | None:
    """Map one genuine call's payload onto client required keys, or None.

    Capability-exact rewrites only: `shell` {"command"} -> {"cmd"}
    (spark exec_command). `read`/`write` NEVER translate: a client tool
    that merely shares the `path` key (live: spark + luna nested
    `view_image`, an image-path viewer) is not a file reader/writer,
    and a rename onto it replays bytes the harness fails client-side.
    Extra genuine keys (workdir/timeout/...) are dropped — the client
    schema is the contract. Unknown client shapes return None (no
    guess).
    """
    import json as _json

    required_set = set(required or [])
    if genuine == "shell":
        if "command" not in payload:
            return None
        if required_set == {"cmd"} or "cmd" in required_set:
            return _json.dumps({"cmd": payload["command"]})
        return None
    if genuine in ("read", "write", "edit"):
        # No translation: sharing a `path` key does not make a client
        # tool a file reader/writer (live `read` renamed onto
        # view_image, and live luna `write` renamed onto nested
        # view_image — both failed client-side). Same-shape "passthrough"
        # is no safer (`edit` {path, content} onto any tool requiring
        # those keys has the identical capability mismatch). The steer
        # path teaches exec_command/cat (read) or apply_patch via the
        # exec-channel guidance (write/edit) instead.
        return None
    return None


def exec_channel_source(
    nested_name: str, arguments: str, tools: tuple[ToolDef, ...]
) -> str | None:
    """JS `tools.<name>(...)` source invoking one nested tool, or None.

    The deferred `functions`-style namespace only executes its nested
    tools (exec_command, apply_patch, ...) through the `exec` JS
    orchestrator as a `custom_tool_call` named `exec` — a bare
    `function_call` named exec_command fails lookup. Rewrites a valid
    nested call into that channel: args-object declarations become
    `await tools.<name>(<json>)`; freeform `(input: string)`
    declarations (apply_patch patch text) pass the raw input through
    unchanged. The signatures live in the namespace's exec description
    (parsed via _nested_exec_tools), not in the synthetic ToolDefs.
    Returns None when the name is not a documented nested tool.
    """
    lowered = nested_name.lower() if isinstance(nested_name, str) else ""
    decl: str | None = None
    for t in tools:
        exec_desc = _namespace_exec_desc(t)
        if not exec_desc:
            continue
        decl = next(
            (d for n, d in _nested_exec_tools(exec_desc) if n.lower() == lowered),
            None,
        )
        if decl is not None:
            break
    if decl is None:
        return None
    if _is_freeform_decl(decl):
        # Patch text / raw string input: the orchestrator takes it
        # verbatim (no JSON wrapper — the harness parses Custom input
        # as raw text).
        return arguments if isinstance(arguments, str) else ""
    try:
        payload = json.loads(arguments) if isinstance(arguments, str) else None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    # The harness parses the exec input as JS source and the nested
    # methods take an object: the JSON payload embeds directly as the
    # object literal (JSON string escapes are valid JS). Matches the
    # form the steer guidance already teaches the model.
    return f"await tools.{nested_name}({json.dumps(payload)})"


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


# Client tools the notice sanctions an upstream alias for: (client
# lowered, upstream alias, alias required key). The alias rides the
# wire with its schema (genuine-12-only outbound) while the client name
# travels as notice text alone — emissions under the client name come
# back `{}` (live spark A/B 2026-10-06), so notice, redirect, and steer
# all point shell commands at the alias, and the classifier rewrites it
# back (translate_genuine_call).
_ALIASES: tuple[tuple[str, str, str], ...] = (("exec_command", "shell", "command"),)


def build_tool_notice(
    tools: tuple[ToolDef, ...], genuine_names, *, shell_alias: bool = False
) -> str:
    genuine_lower = {str(n).lower() for n in genuine_names}
    named = [t for t in tools if t.name]
    if not named:
        return ""
    # Upstream-`shell` directive (NOT an either/or choice): the client
    # shell-runner has no wire schema — outbound is genuine-12-only, so
    # the model offered `exec_command`+`cmd` as notice text alone emits
    # `{}` and never fills (live spark A/B, 2026-10-06: the SAME notice
    # produced a clean translated `{"cmd": "cat /tmp/hi"}` turn with zero
    # steers when instructions named `shell`+`command`, and `{}` x3 then
    # fail-closed when they named `exec_command`+`cmd`). Genuine `shell`
    # DOES ride the wire with its required `command` schema, and
    # translate_genuine_call rewrites it onto the client runner — so the
    # notice directs shell commands at `shell` ONLY and demotes the
    # client name to a pointer (naming both as equal options lets the
    # model pick the schema-less one). Nested legs keep the tight
    # sketches (shell_alias=False): a bare "call `shell`" sentence would
    # mis-teach where tools run through the exec orchestrator channel.
    alias_lines: list[str] = []
    aliased: set[str] = set()
    if shell_alias:
        for client_lower, alias, _ in _ALIASES:
            if alias not in genuine_lower:
                continue
            if client_lower in genuine_lower:
                continue
            tool = next((t for t in named if t.name.lower() == client_lower), None)
            if tool is None or tool.kind != "function":
                continue
            if not _is_cmd_shaped(tool):
                continue
            aliased.add(client_lower)
            alias_lines.append(
                f"- '{alias}': run every shell command as {GENUINE_SHELL_SCHEMA_LINE}."
            )
    declared_lower = {t.name.lower() for t in named}
    shell_offered = bool(alias_lines) or "shell" in declared_lower
    # No contradiction: when the shell directive applies (or the client
    # declared `shell` itself), the header must not ban `shell` in the
    # same breath the notice offers it.
    banned_shell = "" if shell_offered else ", shell"
    lines = [
        (
            "Your harness can only execute the tools listed below — call "
            "them by these exact names with these exact parameter shapes. "
            "Use each name EXACTLY as written below: never add a namespace "
            "prefix (e.g. never `default.exec_command` — always bare "
            "`exec_command`). "
            "Any other tool name will fail in your harness: in particular, "
            f"the default tools read{banned_shell}, write, edit, glob, grep, "
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
            entries = _nested_exec_tools(exec_desc)
            freeform = next((n for n, d in entries if _is_freeform_decl(d)), None)
            if freeform is not None:
                how = (
                    "To run a nested tool, emit ONE `custom_tool_call` "
                    "item named `exec` whose `input` is JavaScript calling "
                    "it on the `tools` object — e.g. to create a file: "
                    '`{"type": "custom_tool_call", "name": "exec", '
                    f'"input": "await tools.{freeform}('
                    "'*** Begin Patch ***\\n*** Add File: <path>\\n"
                    "<content>\\n*** End Patch ***')\"}`; e.g. to run a "
                    "shell command: "
                    '`{"type": "custom_tool_call", "name": "exec", '
                    '"input": "await tools.exec_command({cmd: '
                    '"cat /tmp/codex-cap-ws/probe.txt"})"}`. Never emit '
                    "a bare `function_call` named `exec_command`, "
                    "`apply_patch`, or any other nested name."
                )
            else:
                how = (
                    "To run a nested tool, emit ONE `custom_tool_call` "
                    "item named `exec` whose `input` is JavaScript calling "
                    "it on the `tools` object — e.g. to run a shell "
                    "command: "
                    '`{"type": "custom_tool_call", "name": "exec", '
                    '"input": "await tools.exec_command({cmd: '
                    '"cat /tmp/codex-cap-ws/probe.txt"})"}`. Never emit '
                    "a bare `function_call` named `exec_command`, "
                    "`apply_patch`, or any other nested name."
                )
            lines.append(how)
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
        # Plain function leg: the tool already travels upstream as a
        # genuine-12 entry with its full schema — the notice only needs
        # the name, the call shape, and one example. Dumping the full
        # client schema here (~1.7KB for exec_command) teaches the
        # model the WRONG lesson: it emits the namespaced form
        # (`default.exec_command`) with `{}` and ignores the required
        # key (live spark leg, 3x `{}` then budget exhaustion).
        # Shell-runner with an upstream `shell` directive: demote the
        # client name to a pointer (it has no wire schema — outbound is
        # genuine-12-only — so a full client sketch only competes with
        # the `shell` directive and the model picks the schema-less
        # name; live spark A/B 2026-10-06).
        if t.name.lower() in aliased:
            lines.append(
                f"- '{t.name}'{marker}: harness runner — do NOT call it "
                "directly; run shell commands via `shell` above."
            )
        else:
            lines.append(f"- '{t.name}'{marker}: {_call_sketch(t)}")
    # The `shell` directive rides AFTER the client tool entries (never
    # before the HOW-TO or the tool list): the actionable client names
    # come first and the alias reads as the working route, not a
    # footnote — but the client name is never framed as an equal
    # choice (see above).
    lines.extend(alias_lines)
    return "\n".join(lines)


def _is_cmd_shaped(tool: ToolDef) -> bool:
    """True when a client tool is the `shell` -> `cmd` rewrite target.

    Same contract as _translate_genuine_args("shell", ...): a cmd key
    among the required keys (exact {"cmd"} or a superset — extra
    required keys ride the sketch verbatim and the strict check stays
    the source of truth). Pure helper for the notice alias gate.
    """
    required = (tool.parameters or {}).get("required") or []
    return "cmd" in set(required)


# The upstream-offered `shell` schema line for the notice alias: the
# exact key the model fills when the notice names it (live spark
# 2026-10-06 diagnostic: `shell` + `command` in the notice ->
# `shell {"command": "cat /tmp/hi"}` turn 1 -> translated to
# `exec_command {"cmd": ...}`, zero steers). Byte-fidelity with
# zen_tools is NOT required here — this line never rides the wire
# (genuine-12-only outbound); it is notice text, and the classifier
# keys on names, never on this schema.
GENUINE_SHELL_SCHEMA_LINE = '`shell` with {"command": "<string>"} (run a shell command)'


def _call_sketch(t) -> str:
    """One-line call sketch: description + required-keys example."""
    required = (t.parameters or {}).get("required") or []
    props = (t.parameters or {}).get("properties") or {}
    bits = []
    for key in required:
        kind = (props.get(key) or {}).get("type", "string")
        if key == "cmd":
            bits.append('"cmd": "cat /tmp/hi"')
        else:
            bits.append(f'"{key}": "<{kind}>"')
    example = "{" + ", ".join(bits) + "}"
    desc = (t.description or "").split("\n")[0][:160]
    return f"{desc} Call as {example}."


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
    Freeform (patch-text) tools render their full declaration; args
    tools render a one-line call shape (name + required keys) — the
    full TS declaration with every optional key buries the lead (live
    luna: the exec_command entry's 1.7KB declaration crowded out the
    freeform apply_patch entry the write probe needed).
    """
    exec_desc = _namespace_exec_desc(t)
    if not exec_desc:
        return None
    entries = _nested_exec_tools(exec_desc)
    if not entries:
        return None
    return [f"  - '{name}': {_short_decl(name, decl)}" for name, decl in entries]


def _short_decl(name: str, decl: str) -> str:
    """One entry's notice text: full text for freeform, shape for args."""
    import re

    lines = decl.splitlines()
    prose = lines[0] if lines else ""
    body = "\n".join(lines[1:])
    m = re.search(r"args:\s*\{(.*?)\}", body, re.DOTALL)
    if not m:
        # Freeform ((input: string)) or unparseable: the declaration
        # IS the call shape — keep it whole.
        return decl
    required = [
        n
        for n, opt in re.findall(
            r"^\s*([A-Za-z0-9_]+)(\?)?\s*:", m.group(1), re.MULTILINE
        )
        if not opt
    ]
    shape = ", ".join(required) if required else "no required args"
    out = (
        f"{prose} Call as args-object {{{shape}}} on tools.{name} via exec."
        if prose
        else f"Call as args-object {{{shape}}} on tools.{name} via exec."
    )
    return out


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
