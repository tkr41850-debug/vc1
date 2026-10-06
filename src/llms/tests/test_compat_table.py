"""Unit-test the hermetic 1:1 translation table entries the compat layer adds.

Add cases here BEFORE touching the implementation (systematic-debugging
Phase 4: failing test first, single fix, verify). Each entry documents
the live capture or harness declaration it mirrors, and untranslatable
arms must fail open (return None), never steer or synthesize a guess.
"""

# Verbatim downstream UAs (proxy log, 2026-10-06 probes): prefix matching
# only — versions/platforms shift per release, so exact equality would
# break on every harness upgrade.
CODEX_UA = "codex_exec/0.154.0 (Debian 13.0.0; x86_64) screen (codex_exec; 0.154.0)"
CLAUDE_UA = "claude-cli/2.1.290 (external, sdk-cli)"


def test_detect_family_prefers_ua_prefix_falls_back_to_shape():
    from llms.proxy.compat import detect_family
    from llms.proxy.ir import ToolDef

    # UA prefix wins when present (verbatim strings above).
    assert detect_family(CODEX_UA, "responses", ()) == "codex-plain"
    assert detect_family(CODEX_UA + " extra-suffix", "responses", ()) == "codex-plain"
    assert detect_family(CLAUDE_UA, "messages", ()) == "claude"
    # No UA: responses + functions namespace -> luna; top-level
    # function tools -> codex-plain; messages leg -> claude.
    luna_ns = ToolDef(
        "functions",
        "",
        {},
        kind="namespace",
        options={"tools": [{"name": "exec", "description": "Run JS"}]},
    )
    assert detect_family("", "responses", (luna_ns,)) == "luna"
    plain = ToolDef("exec_command", "run", {"required": ["cmd"]})
    assert detect_family("", "responses", (plain,)) == "codex-plain"
    assert detect_family(None, "messages", ()) == "claude"
    # Nothing recognizable -> unknown (today's behavior, *-only entries).
    assert detect_family("", "responses", ()) == "unknown"


def test_family_for_breaks_ua_tie_with_requested_model():
    # The MODEL_ALIASES mapping rides luna (`gpt-*`) on spark upstream
    # behind the SAME codex UA: UA alone says codex-plain while the
    # tools carry the deferred `functions` namespace (live luna probe
    # 2026-10-06: family=codex-plain logged, generic redirect, no file).
    # The requested (pre-alias) model name breaks the tie — but only
    # with namespace evidence (a bare gpt- UA/model never claims luna).
    from llms.proxy.ir import RequestIR
    from llms.proxy.pipeline import _family_for
    from tests.test_client_tools import _luna_namespace, _spark_runner

    luna_req = RequestIR(
        model="muse-spark-1.3-contributor-free", tools=(_luna_namespace(),)
    )
    spark_req = RequestIR(
        model="muse-spark-1.3-contributor-free", tools=(_spark_runner(),)
    )

    class _H:
        def __init__(self, ua):
            self.headers = {"user-agent": ua}

    assert (
        _family_for(_H(CODEX_UA), "responses", luna_req, "gpt-5.6-luna") == "luna"
    )
    assert (
        _family_for(_H(CODEX_UA), "responses", luna_req, "muse-spark-1.3-contributor-free")
        == "codex-plain"
    )
    assert (
        _family_for(_H(CODEX_UA), "responses", spark_req, "gpt-5.6-luna")
        == "codex-plain"
    )


def test_classifier_steers_empty_custom_exec_call():
    # Live luna shape (/tmp/exec_stream.txt, 2026-10-06): the model emits
    # `custom_tool_call exec` with EMPTY input, and a stray
    # `function_call_arguments.done {"arguments":"{}","name":"default.exec"}`
    # frame. The parser's done handler overwrites the announced name
    # unconditionally, so the fold judges `default.exec` — an owned
    # name (`exec` in the default namespace) with no payload the
    # harness cannot execute. It must STEER (with the exec-channel
    # correction), never pass through. Folds the live bytes so the
    # test pins the fold+classifier contract, not a hand-built dict.
    from llms.proxy.client_tools import (
        dispatchable_names,
        nested_tool_defs,
        owned_tool_names,
    )
    from llms.proxy.forward import fold_stream_calls
    from llms.proxy.pipeline import _classify_calls
    from tests.test_client_tools import _luna_namespace

    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    route = dispatchable_names(ns)
    lines = [
        (
            'data: {"type":"response.output_item.added","output_index":0,'
            '"item":{"id":"call_e1","type":"custom_tool_call","name":"exec",'
            '"input":"","call_id":"call_e1","status":"in_progress"}}'
        ),
        ": ping",
        (
            'data: {"type":"response.function_call_arguments.done",'
            '"output_index":0,"item_id":"call_e1","name":"default.exec",'
            '"arguments":"{}"}'
        ),
        ": ping",
        ('data: {"type":"response.completed","response":{"status":"completed"}}'),
        ": ping",
    ]
    (call,) = fold_stream_calls(lines, "responses")
    assert call["name"] == "default.exec"
    assert call.get("input") == ""
    passed, steer = _classify_calls([call], owned, defs, route, ns, ())
    assert passed == []
    assert [c["name"] for c in steer] == ["default.exec"]


def test_fold_keeps_custom_input_for_valid_exec_rewrite():
    # A rewritten turn replays as `custom_tool_call exec` with the JS
    # invocation in `input` (no JSON arguments, no input deltas). The
    # fold must carry that input so the tail-check classifier judges
    # the payload the client will execute — not an empty shape that
    # steers the proxy's own rewrite (live: streaming steer budget
    # exhausted on the rewritten clean turn).
    import json as _json

    from llms.proxy.forward import fold_stream_calls

    js = 'await tools.exec_command({"cmd": "echo hi"})'
    lines = [
        "data: "
        + _json.dumps(
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "call_n1",
                    "type": "custom_tool_call",
                    "name": "exec",
                    "input": js,
                    "call_id": "call_n1",
                    "status": "in_progress",
                },
            }
        ),
        ": ping",
        ('data: {"type":"response.completed","response":{"status":"completed"}}'),
        ": ping",
    ]
    (call,) = fold_stream_calls(lines, "responses")
    assert call["name"] == "exec"
    assert call.get("type") == "custom_tool_call"
    assert call.get("input") == js


def test_write_to_apply_patch_needs_live_grammar_proof():
    # Task 3 gate: NO write->apply_patch table entry ships until the
    # Add-File marker grammar is verified live against the harness
    # (2026-10-06: 4 FAIL-nofile attempts; the marker text the proxy
    # would synthesize is still a guess). This test pins the CURRENT
    # contract — translate_to_client("write", ...) is None (fail open,
    # never synthesize) — so a future entry must update this test
    # with the live proof, not slip in silently.
    from llms.proxy.compat import translate_to_client
    from llms.proxy.ir import ToolDef

    nested_patch = ToolDef(
        "apply_patch",
        "Edit files, freeform.",
        {},
        kind="custom",
    )
    owned = {"apply_patch": "apply_patch"}
    defs = {"apply_patch": nested_patch}
    assert (
        translate_to_client(
            "write",
            '{"path": "grammar-add.txt", "content": "hello-grammar"}',
            "luna",
            owned,
            defs,
        )
        is None
    )


def test_dispatch_shell_to_cmd_runner_all_families():
    # shell{"command"} -> exec_command{"cmd"} on every family owning a
    # cmd-shaped runner (live spark leg, zero-steer turn 2026-10-06).
    from llms.proxy.compat import translate_to_client
    from llms.proxy.ir import ToolDef

    runner = ToolDef(
        "exec_command",
        "run",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": runner}
    assert translate_to_client(
        "shell", '{"command": "cat /tmp/hi"}', "codex-plain", owned, defs
    ) == ("exec_command", '{"cmd": "cat /tmp/hi"}')
    assert translate_to_client(
        "shell", '{"command": "cat /tmp/hi"}', "unknown", owned, defs
    ) == ("exec_command", '{"cmd": "cat /tmp/hi"}')
    # Missing payload key or no cmd-runner: None (fail open, never guess).
    assert (
        translate_to_client(
            "shell", '{"workdir": "/tmp"}', "codex-plain", owned, defs
        )
        is None
    )
    assert (
        translate_to_client("shell", '{"command": "x"}', "codex-plain", {}, {})
        is None
    )
    # Same-name ownership wins: the client declared `shell` itself, so the
    # owned argument-correction path applies, not a rewrite.
    own_shell = ToolDef(
        "Shell",
        "mine",
        {"type": "object", "properties": {"cmd": {}}, "required": ["cmd"]},
    )
    assert (
        translate_to_client(
            "shell",
            '{"command": "x"}',
            "codex-plain",
            {"shell": "Shell"},
            {"shell": own_shell},
        )
        is None
    )


def test_streaming_steer_log_records_call_arguments(caplog):
    # Live luna apply_patch probes (2026-10-06, probes 3-4): the steer
    # INFO named the call (`execute`) but not its arguments, so the
    # pointed execute->exec rewrap's silence live was unanswerable —
    # empty `code` (correct generic) vs populated `code` (arm should
    # have fired). The log must carry truncated arguments so the next
    # live probe answers that question from the log alone.
    import json as _json
    import logging as _logging

    import httpx as _httpx

    from tests.conftest import TEST_HEADERS, TEST_SECRET, build_app_client
    from tests.conftest import make_settings as _make_settings

    code = "await tools.apply_patch('*** Begin Patch ***')"
    seen: list = []

    async def handler(request):
        seen.append(_json.loads(request.content.decode()))
        if len(seen) == 1:
            body = (
                'data: {"type":"response.output_item.added","output_index":1,'
                '"item":{"id":"call_ex1","type":"function_call","name":"execute",'
                '"arguments":"{}",'
                + '"call_id":"call_ex1","status":"in_progress"}}\n\n'
                + 'data: {"type":"response.function_call_arguments.delta",'
                '"output_index":1,"item_id":"call_ex1",'
                '"delta":'
                + _json.dumps(_json.dumps({"code": code}))
                + "}\n\n"
                + 'data: {"type":"response.completed",'
                '"response":{"id":"resp_dead1","status":"completed"}}\n\n'
            )
        else:
            body = (
                'data: {"type":"response.completed",'
                '"response":{"id":"resp_clean1","status":"completed"}}\n\n'
            )
        return _httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    client = _httpx.AsyncClient(
        transport=_httpx.MockTransport(handler),
        base_url="https://opencode.ai/zen/v1",
    )
    tools = [
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "tools": [{"type": "custom", "name": "exec", "description": "Run JS"}],
        }
    ]
    with (
        caplog.at_level(_logging.INFO, logger="zen_proxy"),
        build_app_client(
            _make_settings(data_dir="/tmp/pytest-steer-log"),
            client,
            seed_key=TEST_SECRET,
        ) as tc,
    ):
        r = tc.post(
            "/v1/responses",
            json={
                "model": "gpt-5.6-luna",
                "input": "do it",
                "stream": True,
                "tools": tools,
            },
            headers=TEST_HEADERS,
        )
    assert r.status_code == 200
    steer_lines = [
        rec.getMessage()
        for rec in caplog.records
        if "steering streaming tool call" in rec.getMessage()
    ]
    assert steer_lines, "expected a streaming steer log line"
    assert "execute" in steer_lines[0]
    assert "apply_patch" in steer_lines[0]
