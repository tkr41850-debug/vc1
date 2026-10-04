from llms.proxy.ir import ToolDef

SHELL = ToolDef(
    "Shell",
    "Run a command",
    {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
)


def test_owned_map_prefers_declared_casing():
    from llms.proxy.client_tools import owned_tool_names

    assert owned_tool_names((SHELL,)) == {"shell": "Shell"}


def test_convert_call_name_rewrites_to_declared_casing():
    from llms.proxy.client_tools import convert_call_name

    assert convert_call_name("shell", {"shell": "Shell"}) == "Shell"
    assert convert_call_name("Shell", {"shell": "Shell"}) == "Shell"
    assert convert_call_name("frobnicate", {"shell": "Shell"}) == "frobnicate"


def test_missing_required_keys_valid_missing_and_unparseable():
    from llms.proxy.client_tools import missing_required_keys

    assert missing_required_keys('{"cmd": "echo hi"}', SHELL) == []
    assert missing_required_keys('{"command": "echo hi"}', SHELL) == ["cmd"]
    assert missing_required_keys("", SHELL) is None
    assert missing_required_keys("not-json{{{", SHELL) is None


def test_notice_empty_without_tools_marks_overrides():
    from llms.proxy.client_tools import build_tool_notice
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    assert build_tool_notice((), GENUINE_TOOL_NAMES) == ""
    notice = build_tool_notice((SHELL,), GENUINE_TOOL_NAMES)
    assert "'Shell'" in notice and "override" in notice and '"cmd"' in notice


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
    assert "not valid" in text and "not available" not in text


def test_redirect_undeclared_returns_none():
    from llms.proxy.client_tools import build_tool_redirect, owned_tool_names

    owned = owned_tool_names((SHELL,))
    assert (
        build_tool_redirect("frobnicate", '{"x": 1}', owned, {"shell": SHELL}) is None
    )
