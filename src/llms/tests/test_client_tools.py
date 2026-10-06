from llms.proxy.ir import ToolDef

SHELL = ToolDef(
    "Shell",
    "Run a command",
    {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
)


def test_owned_map_prefers_declared_casing():
    from llms.proxy.client_tools import owned_tool_names

    assert owned_tool_names((SHELL,)) == {"shell": "Shell"}


def test_owned_map_dotted_declaration_also_owns_bare():
    # Harness dispatch asymmetry (live spark leg): the client declares
    # bare `exec_command` while the model emits `default.exec_command`
    # (and vice versa — a dotted declaration must serve bare calls).
    # The harness fills its default namespace when absent, so both
    # keyings resolve to the declaration; otherwise the turn steers
    # forever as undeclared.
    from llms.proxy.client_tools import dispatchable_names, owned_tool_names
    from llms.proxy.ir import ToolDef

    dotted = ToolDef(
        "default.exec_command",
        "Runs a command",
        {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    )
    owned = owned_tool_names((dotted,))
    assert owned["exec_command"] == "default.exec_command"
    assert owned["default.exec_command"] == "default.exec_command"
    route = dispatchable_names((dotted,))
    assert route["exec_command"] == ("function_call", "default.exec_command")
    # Bare declarations are unaffected (no dotted keys appear).
    assert owned_tool_names((SHELL,)) == {"shell": "Shell"}


def test_owned_map_includes_nested_names():
    # Deferred-namespace nested tools dispatch by bare name, so the
    # classifier must own them (else exec_command steers forever).
    from llms.proxy.client_tools import nested_tool_defs, owned_tool_names

    owned = owned_tool_names((_luna_namespace(),))
    assert owned["exec_command"] == "exec_command"
    assert owned["apply_patch"] == "apply_patch"
    assert owned["wait"] == "wait"
    assert owned["functions"] == "functions"
    # Nested defs validate: args-object entries require `args` (a
    # bare call steers with a correction); freeform entries require
    # nothing (raw input IS the payload). Raw specs keep their
    # declared parameters.
    defs = nested_tool_defs((_luna_namespace(),))
    # Inner wire keys, not the TS `args:` wrapper (which is never a
    # wire key — requiring it rejects every valid call live).
    assert defs["exec_command"].parameters.get("required") == ["cmd"]
    assert defs["apply_patch"].parameters == {}
    assert defs["wait"].parameters == {"type": "object", "properties": {}}


def test_convert_call_name_rewrites_to_declared_casing():
    from llms.proxy.client_tools import convert_call_name

    assert convert_call_name("shell", {"shell": "Shell"}) == "Shell"
    assert convert_call_name("Shell", {"shell": "Shell"}) == "Shell"
    assert convert_call_name("frobnicate", {"shell": "Shell"}) == "frobnicate"


def test_missing_required_keys_valid_missing_and_unparseable():
    from llms.proxy.client_tools import missing_required_keys

    assert missing_required_keys('{"cmd": "echo hi"}', SHELL) == []
    assert missing_required_keys('{"command": "echo hi"}', SHELL) == ["cmd"]
    # Blank is a no-arg call, not unparseable prose: every required key
    # is missing, so the redirect names the keys (live: codex emits
    # empty-args exec_command calls, and "not valid JSON" never told
    # the model which argument to supply).
    assert missing_required_keys("", SHELL) == ["cmd"]
    assert missing_required_keys("not-json{{{", SHELL) is None


def test_notice_empty_without_tools_marks_overrides():
    from llms.proxy.client_tools import build_tool_notice
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    assert build_tool_notice((), GENUINE_TOOL_NAMES) == ""
    notice = build_tool_notice((SHELL,), GENUINE_TOOL_NAMES)
    assert "'Shell'" in notice and "override" in notice and '"cmd"' in notice
    # The harness can only execute the tools below: the notice names the
    # unroutable defaults (read/shell/write/...) so the model stops
    # calling prepended genuine names its harness cannot dispatch.
    assert "can only execute the tools listed below" in notice
    # The header names the bare form as the anti-prefix example (live
    # spark leg: the model emitted `default.exec_command` with `{}` and
    # ignored the required key until told the name is exactly bare).
    assert "never `default.exec_command` — always bare" in notice


def test_redirect_owned_missing_keys_names_tool_and_shape():
    from llms.proxy.client_tools import build_tool_redirect, owned_tool_names

    owned = owned_tool_names((SHELL,))
    defs = {"shell": SHELL}
    text = build_tool_redirect("shell", '{"command": "echo hi"}', owned, defs)
    assert text is not None
    assert "'Shell'" in text and "not available" not in text
    assert "cmd" in text and '"cmd"' in text  # missing key + sketch
    # The full schema is deliberately NOT repeated (it never converted
    # the model and the notice already carries it): the correction is
    # a short retry sketch, not a schema dump.
    assert "properties" not in text


def test_redirect_names_emitted_form_for_prefixed_calls():
    # Live spark leg: the model emits `default.exec_command` while the
    # declared name is the bare `exec_command`. The correction must
    # bind to the call the model made (exact emitted form) while the
    # retry names the declared tool — a bare-name-only correction
    # never converted the prefixed emitter.
    from llms.proxy.client_tools import build_tool_redirect, owned_tool_names

    owned = owned_tool_names((SHELL,))
    defs = {"shell": SHELL}
    text = build_tool_redirect("default.Shell", "{}", owned, defs)
    assert text is not None
    assert "'default.Shell'" in text  # exact emitted form
    assert "Retry as 'Shell'" in text  # declared retry target
    assert "cmd" in text and "properties" not in text


def test_redirect_owned_bad_json_says_args_not_tool():
    from llms.proxy.client_tools import build_tool_redirect, owned_tool_names

    owned = owned_tool_names((SHELL,))
    text = build_tool_redirect("shell", "", owned, {"shell": SHELL})
    assert text is not None
    # Blank means a no-arg call: the redirect names the missing keys
    # (with the shape) rather than the vaguer not-valid-JSON text.
    assert "cmd" in text and '"cmd"' in text and "not available" not in text
    text = build_tool_redirect("shell", "not-json{{{", owned, {"shell": SHELL})
    assert text is not None
    assert "not valid" in text and "not available" not in text


def test_redirect_undeclared_returns_none():
    from llms.proxy.client_tools import build_tool_redirect, owned_tool_names

    owned = owned_tool_names((SHELL,))
    assert (
        build_tool_redirect("frobnicate", '{"x": 1}', owned, {"shell": SHELL}) is None
    )


def test_notice_namespace_exec_lists_nested_tools():
    # Live luna shape: the deferred `functions` namespace wraps a
    # single `exec` custom tool whose description documents the nested
    # tools (exec_command, apply_patch, ...). The notice must list
    # those nested names with their call signatures — the bare
    # container ('functions', empty description, {} params) reads as
    # "nothing available" and the model refuses file work.
    from llms.proxy.client_tools import build_tool_notice
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    ns = ToolDef(
        "functions",
        "",
        {},
        kind="namespace",
        options={
            "tools": [
                {
                    "type": "custom",
                    "name": "exec",
                    "description": (
                        "Run JavaScript code.\n"
                        "### `exec_command`\n"
                        "Runs a command.\n"
                        "\nexec tool declaration:\n"
                        "```ts\n"
                        "declare const tools: { exec_command(args: {\n"
                        "  cmd: string;\n"
                        "}): Promise<unknown>; };\n"
                        "```\n"
                        "### `apply_patch`\n"
                        "Edit files, freeform.\n"
                        "\nexec tool declaration:\n"
                        "```ts\n"
                        "declare const tools: { apply_patch(input: string): Promise<unknown>; };\n"
                        "```\n"
                    ),
                }
            ]
        },
    )
    notice = build_tool_notice((ns,), GENUINE_TOOL_NAMES)
    assert "'exec_command'" in notice and "cmd" in notice
    assert "'apply_patch'" in notice


def test_notice_nested_patch_example_names_custom_form():
    # Live luna 2026-10-06: the notice's only `custom_tool_call` example
    # used a shell command, and the apply_patch entry showed a TS
    # declaration — the model concluded "there's no `exec` custom tool
    # available in this session" and made no call at all (single
    # ingress, zero steers). The example must name the patch shape in
    # the Custom form so a file-write turn has a concrete pattern to
    # bind the payload to.
    from llms.proxy.client_tools import build_tool_notice
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    notice = build_tool_notice((_luna_namespace(),), GENUINE_TOOL_NAMES)
    assert "custom_tool_call" in notice and "apply_patch" in notice
    assert "await tools.apply_patch(" in notice


def test_notice_nested_entries_tighten_args_keep_freeform():
    # Args-object nested tools render a one-line call shape (name +
    # required keys), not the full TS declaration: the full text
    # buries the lead (live luna: the 1.7KB exec_command declaration
    # crowded out the apply_patch entry the write probe needed).
    # Freeform (patch-text) tools keep their full declaration — it IS
    # the call shape.
    from llms.proxy.client_tools import _short_decl

    assert _short_decl(
        "exec_command",
        "Runs a command.\ndeclare const tools: { exec_command(args: {\n  cmd: string;\n  login?: boolean;\n}): Promise<unknown>; };",
    ) == "Runs a command. Call as args-object {cmd} on tools.exec_command via exec."
    freeform = (
        "Edit files.\ndeclare const tools: { apply_patch(input: string): Promise<unknown>; };"
    )
    assert _short_decl("apply_patch", freeform) == freeform


def test_notice_namespace_without_exec_renders_normally():
    # A namespace NOT wrapping exec (e.g. multi_agent_v1) renders as a
    # plain entry — no nested extraction attempted.
    from llms.proxy.client_tools import build_tool_notice
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    ns = ToolDef("multi_agent_v1", "Spawn agents.", {}, kind="namespace")
    notice = build_tool_notice((ns,), GENUINE_TOOL_NAMES)
    assert "'multi_agent_v1'" in notice
    assert "dispatch container" not in notice


def _luna_namespace():
    # Live luna shape: deferred `functions` namespace wrapping exec
    # (documents exec_command + apply_patch) plus plain nested tools.
    return ToolDef(
        "functions",
        "",
        {},
        kind="namespace",
        options={
            "tools": [
                {
                    "type": "custom",
                    "name": "exec",
                    "description": (
                        "Run JavaScript code.\n"
                        "### `exec_command`\n"
                        "Runs a command.\n"
                        "\nexec tool declaration:\n"
                        "```ts\n"
                        "declare const tools: { exec_command(args: {\n"
                        "  cmd: string;\n"
                        "}): Promise<unknown>; };\n"
                        "```\n"
                        "### `apply_patch`\n"
                        "Edit files, freeform.\n"
                        "\nexec tool declaration:\n"
                        "```ts\n"
                        "declare const tools: { apply_patch(input: string): Promise<unknown>; };\n"
                        "```\n"
                    ),
                },
                {
                    "type": "function",
                    "name": "wait",
                    "description": "Wait.",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]
        },
    )


def test_route_luna_nested_names_with_item_types():
    # The harness dispatches by item type as well as name: only the
    # freeform (input: string) nested decl routes Custom; args-object
    # nested tools route Function. The container itself is not
    # model-callable and must not route.
    from llms.proxy.client_tools import dispatchable_names

    route = dispatchable_names((_luna_namespace(),))
    assert route["exec_command"] == ("function_call", "exec_command")
    assert route["apply_patch"] == ("custom_tool_call", "apply_patch")
    assert route["wait"] == ("function_call", "wait")
    # The container itself is not model-callable and must not route;
    # the raw nested `exec` spec routes Custom (direct JS invocation).
    assert "functions" not in route
    assert route["exec"] == ("custom_tool_call", "exec")


def test_route_top_level_custom_routes_custom():
    # Spark leg: freeform top-level custom (apply_patch) dispatches as
    # Custom only — never the function default (matches_kind fatals).
    from llms.proxy.client_tools import dispatchable_names

    t = ToolDef("apply_patch", "Edit.", {}, kind="custom")
    assert dispatchable_names((t,)) == {
        "apply_patch": ("custom_tool_call", "apply_patch")
    }
    assert dispatchable_names((SHELL,)) == {"shell": ("function_call", "Shell")}


def test_rewrite_custom_route_retypes_with_raw_input():
    # A steered Custom-route call replays upstream as custom_tool_call
    # with the raw input preserved (never JSON arguments).
    from llms.proxy.client_tools import dispatchable_names, rewrite_steered_call

    route = dispatchable_names((_luna_namespace(),))
    out = rewrite_steered_call(
        {"type": "function_call", "call_id": "c1", "name": "apply_patch"},
        "*** Begin Patch ***",
        route,
    )
    assert out is not None
    assert out["type"] == "custom_tool_call"
    assert out["input"] == "*** Begin Patch ***"
    assert "arguments" not in out


def test_rewrite_function_route_returns_none():
    # Function-route calls need no rewrite: the caller replays verbatim.
    from llms.proxy.client_tools import dispatchable_names, rewrite_steered_call

    route = dispatchable_names((_luna_namespace(),))
    call = {"type": "function_call", "call_id": "c1", "name": "exec_command"}
    assert rewrite_steered_call(call, '{"cmd": "x"}', route) is None
    assert rewrite_steered_call(call, '{"cmd": "x"}', {}) is None


def test_split_call_name_strips_code_mode_namespace():
    # Codex flattens non-default namespaces with `.` (`flat_tool_name`):
    # `default.exec_command` is name `exec_command` in namespace
    # `default`, not a literal tool name.
    from llms.proxy.client_tools import convert_call_name, split_call_name

    assert split_call_name("default.exec_command") == ("default", "exec_command")
    assert split_call_name("exec_command") == ("", "exec_command")
    assert split_call_name("") == ("", "")
    # Casing convert keys on the bare name (live: namespaced calls
    # steered forever as undeclared).
    assert (
        convert_call_name("default.exec_command", {"exec_command": "exec_command"})
        == "exec_command"
    )


def test_rewrite_namespaced_call_splits_first():
    from llms.proxy.client_tools import rewrite_steered_call

    out = rewrite_steered_call(
        {"type": "function_call", "name": "exec_command"},
        "*** Begin Patch ***",
        {"apply_patch": ("custom_tool_call", "apply_patch")},
    )
    assert out is None  # function-route: no rewrite


def test_display_tool_names_excludes_namespace_container():
    # The redirect's tool list must name only what the model can emit:
    # nested tools by bare name, never the deferred container itself
    # (a bare `functions` call fails lookup — listing it taught the
    # model an unusable name; live luna cycled write/shell/read).
    from llms.proxy.client_tools import display_tool_names

    names = display_tool_names((_luna_namespace(),))
    assert "functions" not in names
    assert "exec_command" in names and "apply_patch" in names
    assert "wait" in names


def test_translate_genuine_shell_onto_client_exec_command():
    # Genuine-overlay `shell` with usable args rewrites onto the
    # client's equivalent tool (spark exec_command/cmd) instead of
    # steering a name the client declared under its own name.
    from llms.proxy.client_tools import translate_genuine_call
    from llms.proxy.ir import ToolDef

    shell = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": shell}
    assert translate_genuine_call("shell", '{"command": "echo hi"}', owned, defs) == (
        "exec_command",
        '{"cmd": "echo hi"}',
    )
    # Missing payload or no equivalent tool: no translation (steer).
    assert translate_genuine_call("shell", '{"workdir": "/tmp"}', owned, defs) is None
    assert translate_genuine_call("glob", '{"pattern": "x"}', owned, defs) is None
    # Same-name ownership wins: the client declared `shell` itself,
    # so the owned path (argument correction) applies, not a rewrite.
    own_shell = ToolDef(
        "Shell",
        "mine",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    assert (
        translate_genuine_call(
            "shell",
            '{"command": "echo hi"}',
            {"shell": "Shell"},
            {"shell": own_shell},
        )
        is None
    )


def test_exec_channel_source_builds_js_invocation():
    # Args-object nested tools become a tools.<name>(<json>) call on
    # the exec `tools` object; freeform tools pass raw input through.
    from llms.proxy.client_tools import exec_channel_source

    ns = (_luna_namespace(),)
    # The harness invokes nested tools as tools.<name>(<json>) with
    # the INNER keys (cmd, not the TS `args:` wrapper): the payload
    # embeds directly as the object literal.
    assert (
        exec_channel_source("exec_command", '{"cmd": "cat f"}', ns)
        == 'await tools.exec_command({"cmd": "cat f"})'
    )
    assert (
        exec_channel_source("apply_patch", "*** Begin Patch ***", ns)
        == "*** Begin Patch ***"
    )
    # Unknown nested names and bad payloads: no rewrite (steer).
    assert exec_channel_source("frobnicate", "{}", ns) is None
    assert exec_channel_source("exec_command", "not-json", ns) is None


def test_translate_genuine_read_never_renames():
    # `read` never translates: a client tool sharing the `path` key
    # (live spark `view_image`, an image-path viewer) is not a file
    # reader — renaming onto it replays a text read the harness
    # fails client-side (`unsupported call: view_image`). The steer
    # path teaches exec_command/cat instead (steer_to_equivalent).
    from llms.proxy.client_tools import translate_genuine_call
    from llms.proxy.ir import ToolDef

    viewer = ToolDef(
        "view_image",
        "view",
        {"type": "object", "properties": {"path": {}}, "required": ["path"]},
    )
    owned = {"view_image": "view_image"}
    defs = {"view_image": viewer}
    assert (
        translate_genuine_call("read", '{"path": "/tmp/f.txt"}', owned, defs)
        is None
    )


def test_translate_genuine_write_never_renames_onto_viewer():
    # `write` never translates: same shared-`path`-key trap as `read`
    # (live luna: a file write renamed onto nested `view_image`, an
    # image viewer documented in the exec description — the harness
    # fails it client-side). The steer path teaches apply_patch via
    # the exec-channel guidance instead. `edit` shares the rule.
    from llms.proxy.client_tools import (
        _translate_genuine_args,
        steer_to_equivalent,
        translate_genuine_call,
    )
    from llms.proxy.ir import ToolDef

    viewer = ToolDef(
        "view_image",
        "view",
        {"type": "object", "properties": {"path": {}}, "required": ["path"]},
    )
    owned = {"view_image": "view_image"}
    defs = {"view_image": viewer}
    args = '{"content": "x", "path": "/tmp/f.txt"}'
    assert translate_genuine_call("write", args, owned, defs) is None
    assert translate_genuine_call("edit", args, owned, defs) is None
    assert _translate_genuine_args("write", {"content": "x", "path": "f"}, ["path"]) is None
    # No directed equivalent either: a bare "retry as apply_patch"
    # sentence would mis-teach (nested tools only run via the exec
    # orchestrator channel) — the generic list + channel guidance
    # applies instead.
    assert steer_to_equivalent("write", args, owned, defs) is None
    assert steer_to_equivalent("edit", args, owned, defs) is None


def test_steer_to_equivalent_directs_shell_onto_cmd_runner():
    # Undeclared `shell` with a command payload maps onto the single
    # client cmd-runner: the redirect names the tool AND the exact
    # arguments (the generic list never converted the model live).
    from llms.proxy.client_tools import steer_to_equivalent
    from llms.proxy.ir import ToolDef

    runner = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    assert steer_to_equivalent(
        "shell", '{"command": "cat f"}', owned, defs
    ) == (
        "Tool 'shell' is not available in this session — use "
        "'exec_command' instead with these arguments: "
        '{"cmd": "cat f"}. Retry the call as \'exec_command\'.'
    )
    # Owned names stay on the argument-correction path; unknown names
    # and ambiguous runners (zero or two cmd tools) get no directed
    # redirect (the generic tool list applies).
    assert steer_to_equivalent("exec_command", '{"cmd": "x"}', owned, defs) is None
    assert steer_to_equivalent("glob", '{"pattern": "x"}', owned, defs) is None
    assert (
        steer_to_equivalent(
            "shell",
            '{"command": "x"}',
            {"a": "a", "b": "b"},
            {
                "a": runner,
                "b": ToolDef(
                    "b", "r", {"type": "object", "properties": {"cmd": {}},
                               "required": ["cmd"]}
                ),
            },
        )
        is None
    )


def test_steer_to_equivalent_directs_read_via_cat():
    # Undeclared `read` has no file-reader equivalent, but a cmd
    # runner serves the capability via `cat`: direct the model there
    # (live spark: `read` re-emitted forever off the generic list).
    from llms.proxy.client_tools import steer_to_equivalent
    from llms.proxy.ir import ToolDef

    runner = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    assert steer_to_equivalent(
        "read", '{"path": "/tmp/f.txt"}', owned, defs
    ) == (
        "Tool 'read' is not available in this session — to read a "
        "file, use 'exec_command' instead with these arguments: "
        '{"cmd": "cat /tmp/f.txt"}. Retry the call as \'exec_command\'.'
    )
    # Paths with spaces quote; no runner or no path means no guess.
    assert steer_to_equivalent(
        "read", '{"path": "/tmp/my f.txt"}', owned, defs
    ) == (
        "Tool 'read' is not available in this session — to read a "
        "file, use 'exec_command' instead with these arguments: "
        '{"cmd": "cat \'/tmp/my f.txt\'"}. Retry the call as \'exec_command\'.'
    )
    assert steer_to_equivalent("read", '{"path": "f"}', {}, {}) is None
    assert steer_to_equivalent("read", '{"offset": 3}', owned, defs) is None


def _spark_runner():
    from llms.proxy.ir import ToolDef

    return ToolDef(
        "exec_command",
        "Runs a command",
        {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    )


def test_notice_shell_directive_plain_leg():
    # Live spark A/B 2026-10-06: the client shell-runner has no wire
    # schema (outbound is genuine-12-only), so instructions naming
    # `exec_command`+`cmd` emit `{}` x3 while `shell`+`command` fills
    # cleanly. The notice directs shell commands at upstream `shell`
    # ONLY and demotes the client runner to a pointer — never an
    # either/or choice (the model picks the schema-less name).
    from llms.proxy.client_tools import build_tool_notice
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    notice = build_tool_notice(
        (_spark_runner(),), GENUINE_TOOL_NAMES, shell_alias=True
    )
    assert "- 'shell': run every shell command as" in notice
    assert '"command"' in notice
    assert "do NOT call it directly" in notice
    assert "EITHER" not in notice
    # No contradiction: the header must not ban `shell` in the same
    # breath the directive offers it.
    assert "read, shell, write" not in notice


def test_notice_shell_directive_off_by_default_and_gated():
    # Default (and nested legs): tight sketches, `shell` still banned,
    # no directive line, no pointer demotion.
    from llms.proxy.client_tools import build_tool_notice
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    for tools, kw in (
        ((_spark_runner(),), {}),
        ((_spark_runner(),), {"shell_alias": False}),
        ((_luna_namespace(),), {"shell_alias": True}),
    ):
        notice = build_tool_notice(tools, GENUINE_TOOL_NAMES, **kw)
        assert "run every shell command" not in notice
        assert "do NOT call it directly" not in notice
    # Gated on the client actually declaring a cmd-shaped function
    # tool: a non-cmd tool (or a client that declares `shell` itself)
    # gets no directive.
    from llms.proxy.ir import ToolDef

    other = ToolDef(
        "frobnicate",
        "Does things",
        {"type": "object", "properties": {"x": {}}, "required": ["x"]},
    )
    notice = build_tool_notice((other,), GENUINE_TOOL_NAMES, shell_alias=True)
    assert "run every shell command" not in notice
    assert "read, shell, write" in notice


def test_redirect_shell_runner_points_at_alias():
    # Owned-but-invalid `exec_command` calls redirect to upstream
    # `shell` (which rides the wire with its schema), not the client
    # name (notice text alone — the model cannot fill from it).
    from llms.proxy.client_tools import build_tool_redirect, owned_tool_names
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    runner = _spark_runner()
    owned = owned_tool_names((runner,))
    defs = {"exec_command": runner}
    text = build_tool_redirect(
        "exec_command", "{}", owned, defs, GENUINE_TOOL_NAMES
    )
    assert text is not None
    assert "Retry as 'shell'" in text
    assert '"command"' in text
    assert "Retry as 'exec_command'" not in text
    # Without the sanction (genuine names absent): legacy client-name
    # correction, unchanged.
    text = build_tool_redirect("exec_command", "{}", owned, defs)
    assert text is not None
    assert "Retry as 'exec_command'" in text


def test_steer_shell_undeclared_points_at_alias():
    # The model emits upstream `shell` (genuine-overlay name the client
    # never declared): with the sanction it steers onto itself with the
    # alias shape, not onto the client runner name.
    from llms.proxy.client_tools import steer_to_equivalent
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    runner = _spark_runner()
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    text = steer_to_equivalent(
        "shell", '{"command": "cat f"}', owned, defs, GENUINE_TOOL_NAMES
    )
    assert text is not None
    assert "Retry the call as 'shell'" in text
    assert '{"command": "cat f"}' in text
    assert "as 'exec_command'" not in text
    # No sanction: legacy directed correction onto the client runner.
    text = steer_to_equivalent("shell", '{"command": "cat f"}', owned, defs)
    assert text is not None
    assert "Retry the call as 'exec_command'" in text


def test_steer_read_via_cat_points_at_alias():
    # `read {path}` served via `cat`: the alias route names
    # `shell`+`command`, the fallback the client runner.
    from llms.proxy.client_tools import steer_to_equivalent
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    runner = _spark_runner()
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    text = steer_to_equivalent(
        "read", '{"path": "/tmp/f.txt"}', owned, defs, GENUINE_TOOL_NAMES
    )
    assert text is not None
    assert "use 'shell' instead" in text
    assert '{"command": "cat /tmp/f.txt"}' in text
    text = steer_to_equivalent("read", '{"path": "/tmp/f.txt"}', owned, defs)
    assert text is not None
    assert "use 'exec_command' instead" in text


def test_steer_execute_genuine_points_at_nested_channel():
    # Live luna apply_patch probe (2026-10-06): the model emits the
    # genuine `execute` JS-runtime tool (Code Mode catalog) to run
    # `tools.apply_patch(...)`. The generic redirect lists client
    # tools and the exec-channel form — but the model kept re-emitting
    # `execute` twice, so the stranded Code Mode call needs a pointed
    # correction naming the custom_tool_call input and the patch-text
    # payload. Unowned genuine name: the ONLY steer arm that can name
    # it. Non-nested legs keep the generic text (no exec channel to
    # teach).
    from llms.proxy.client_tools import owned_tool_names, steer_to_equivalent
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES
    from tests.test_client_tools import _luna_namespace

    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    from llms.proxy.client_tools import nested_tool_defs

    defs.update(nested_tool_defs(ns))
    text = steer_to_equivalent(
        "execute",
        '{"code": "await tools.apply_patch(1)"}',
        owned,
        defs,
        GENUINE_TOOL_NAMES,
        ns,
    )
    assert text is not None
    assert "custom_tool_call" in text
    assert "await tools.apply_patch(" in text
    assert "Retry the call as 'exec'" in text
    # Plain function leg: no exec channel — generic path (None here;
    # the _steer_output_for fallback renders the tool list).
    from tests.test_client_tools import _spark_runner

    runner = _spark_runner()
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    assert (
        steer_to_equivalent(
            "execute", '{"code": "x"}', owned, defs, GENUINE_TOOL_NAMES
        )
        is None
    )
