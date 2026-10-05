from llms.proxy.ir import ToolDef

SHELL = ToolDef(
    "Shell",
    "Run a command",
    {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
)


def test_owned_map_prefers_declared_casing():
    from llms.proxy.client_tools import owned_tool_names

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
    assert defs["exec_command"].parameters.get("required") == ["args"]
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
    assert "exec_command" not in notice  # generic guidance, not client names


def test_redirect_owned_missing_keys_names_tool_and_shape():
    from llms.proxy.client_tools import build_tool_redirect, owned_tool_names

    owned = owned_tool_names((SHELL,))
    defs = {"shell": SHELL}
    text = build_tool_redirect("shell", '{"command": "echo hi"}', owned, defs)
    assert text is not None
    assert "'Shell'" in text and "not available" not in text
    assert "cmd" in text and '"cmd"' in text  # missing key + shape


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
    assert convert_call_name("default.exec_command", {"exec_command": "exec_command"}) == "exec_command"


def test_rewrite_namespaced_call_splits_first():
    from llms.proxy.client_tools import rewrite_steered_call

    out = rewrite_steered_call(
        {"type": "function_call", "name": "exec_command"},
        "*** Begin Patch ***",
        {"apply_patch": ("custom_tool_call", "apply_patch")},
    )
    assert out is None  # function-route: no rewrite
