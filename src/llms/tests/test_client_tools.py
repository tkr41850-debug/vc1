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
