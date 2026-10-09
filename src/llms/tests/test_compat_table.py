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

    assert _family_for(_H(CODEX_UA), "responses", luna_req, "gpt-5.6-luna") == "luna"
    assert (
        _family_for(
            _H(CODEX_UA), "responses", luna_req, "muse-spark-1.3-contributor-free"
        )
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


def test_fold_keeps_custom_input_for_valid_exec_rewrite():  # A rewritten turn replays as `custom_tool_call exec` with the JS
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


def test_bare_patch_exec_input_rewraps_as_apply_patch():
    # Mechanism 13 (live luna 2026-10-07, round 5): under a natural
    # prompt the model drops the `await tools.apply_patch(...)`
    # wrapper and emits the patch text as the whole `custom_tool_call
    # exec` input — 15 turns running, harness `Script failed` +
    # `SyntaxError` each time. The classifier must re-wrap the bare
    # marker text into the channel invocation (passthrough with the
    # __exec_rewrite__ marker), not steer it: steering taught the
    # wrapper for 15 turns and the model never re-added it.
    # Non-patch exec inputs (already-wrapped JS, shell commands)
    # ride through untouched.
    from llms.proxy.client_tools import (
        dispatchable_names,
        nested_tool_defs,
        owned_tool_names,
    )
    from llms.proxy.compat import rewrap_bare_patch_exec_input
    from llms.proxy.pipeline import _classify_calls
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES
    from tests.test_client_tools import _luna_namespace

    # Pure helper level: marker detection + quote-strip + re-wrap.
    # The re-wrap quotes the patch with repr: the harness decodes the
    # JS string escapes, so wire `\\n` becomes the real newlines the
    # patch parser needs (same encoding the round-4 UPDATE proof ran).
    bare = "'*** Begin Patch\\n*** Add File: w.txt\\n+hi\\n*** End Patch'"
    assert rewrap_bare_patch_exec_input(bare) == (
        "await tools.apply_patch('*** Begin Patch\\\\n"
        "*** Add File: w.txt\\\\n+hi\\\\n*** End Patch')"
    )
    assert (
        rewrap_bare_patch_exec_input(
            "*** Begin Patch\n*** Delete File: d.txt\n*** End Patch"
        )
        == "await tools.apply_patch("
        "'*** Begin Patch\\n*** Delete File: d.txt\\n*** End Patch')"
    )
    wrapped = "await tools.apply_patch('*** Begin Patch\\n*** End Patch')"
    assert rewrap_bare_patch_exec_input(wrapped) is None
    assert (
        rewrap_bare_patch_exec_input('await tools.exec_command({"cmd": "x"})') is None
    )
    assert rewrap_bare_patch_exec_input("") is None
    assert rewrap_bare_patch_exec_input(123) is None
    # Classifier level: bare-patch exec passthrough with the marker.
    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    route = dispatchable_names(ns)
    (call,) = [
        {
            "call_id": "c1",
            "name": "exec",
            "type": "custom_tool_call",
            "input": bare,
            "arguments": "",
        }
    ]
    passed, steer = _classify_calls(
        [call], owned, defs, route, ns, GENUINE_TOOL_NAMES, "luna", "t1"
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec"]
    assert passed[0]["__exec_rewrite__"] == {
        "name": "exec",
        "input": rewrap_bare_patch_exec_input(bare),
    }
    # Already-wrapped JS keeps its existing path (exec_channel_source
    # finds no nested name for `exec` itself; rewrite_steered_call
    # re-types — the marker assertion below pins no-regression, not
    # the exact legacy branch).
    (call,) = [
        {
            "call_id": "c2",
            "name": "exec",
            "type": "custom_tool_call",
            "input": wrapped,
            "arguments": "",
        }
    ]
    passed, steer = _classify_calls(
        [call], owned, defs, route, ns, GENUINE_TOOL_NAMES, "luna", "t1"
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec"]
    assert "__exec_rewrite__" not in passed[0] or isinstance(
        passed[0]["__exec_rewrite__"], dict
    )


def test_bare_patch_rewrap_rides_fold_to_classifier():
    # Fold-level pin for mechanism 13: the live wire shape is an
    # `output_item.added` custom_tool_call exec frame carrying the bare
    # patch text in `input` (no JSON arguments, no input deltas). The
    # fold must carry that input into the classifier entry so the
    # rewrap arm fires — the classifier-level test above uses a
    # hand-built dict and would pass even if the fold dropped the
    # input and the arm silently never fired live. Folds synthetic
    # wire bytes (same frame grammar as the empty-exec fold test),
    # then classifies on the luna leg.
    import json as _json

    from llms.proxy.client_tools import (
        dispatchable_names,
        nested_tool_defs,
        owned_tool_names,
    )
    from llms.proxy.compat import rewrap_bare_patch_exec_input
    from llms.proxy.forward import fold_stream_calls
    from llms.proxy.pipeline import _classify_calls
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES
    from tests.test_client_tools import _luna_namespace

    bare = "'*** Begin Patch\\n*** Add File: w.txt\\n+hi\\n*** End Patch'"
    lines = [
        "data: "
        + _json.dumps(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": "call_m13",
                    "type": "custom_tool_call",
                    "name": "exec",
                    "input": bare,
                    "call_id": "call_m13",
                    "status": "in_progress",
                },
            }
        ),
        ": ping",
        (
            'data: {"type":"response.function_call_arguments.done",'
            '"output_index":0,"item_id":"call_m13","name":"default.exec",'
            '"arguments":"{}"}'
        ),
        ": ping",
        ('data: {"type":"response.completed","response":{"status":"completed"}}'),
        ": ping",
    ]
    (call,) = fold_stream_calls(lines, "responses")
    assert call.get("type") == "custom_tool_call"
    assert call.get("input") == bare
    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    route = dispatchable_names(ns)
    passed, steer = _classify_calls(
        [call], owned, defs, route, ns, GENUINE_TOOL_NAMES, "luna", "t1"
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec"]
    assert passed[0]["__exec_rewrite__"] == {
        "name": "exec",
        "input": rewrap_bare_patch_exec_input(bare),
    }


def test_write_to_apply_patch_proven_live():
    # Task 3 gate RETIRED 2026-10-07 (round 4): the Add-File marker
    # grammar is verified live against the harness — luna leg,
    # `custom_tool_call exec` carrying the synthesized patch ran
    # `Script completed` with `FileChange update` and the workspace
    # file byte-exact (`new-line\n`, od-verified; Add-File likewise
    # `hello-grammar\n` round 3). The write->apply_patch row now
    # ships on the luna family only (exec channel required —
    # unknown families still fail open, never a wrong-family
    # rewrite). This test pins the SHIPPED contract — a future
    # regression to None reopens the gate.
    from llms.proxy.client_tools import nested_tool_defs, owned_tool_names
    from llms.proxy.compat import translate_to_client
    from tests.test_client_tools import _luna_namespace, _spark_runner

    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    assert translate_to_client(
        "write",
        '{"path": "grammar-add.txt", "content": "hello-grammar"}',
        "luna",
        owned,
        defs,
        ns,
    ) == (
        "exec",
        "*** Begin Patch\n*** Add File: grammar-add.txt\n+hello-grammar\n*** End Patch",
    )
    # Multi-line content: every line gets a `+` prefix.
    assert translate_to_client(
        "write",
        '{"path": "a.txt", "content": "l1\\nl2\\n"}',
        "luna",
        owned,
        defs,
        ns,
    ) == (
        "exec",
        "*** Begin Patch\n*** Add File: a.txt\n+l1\n+l2\n*** End Patch",
    )
    # Missing/non-string fields: no guess (generic steer applies).
    assert (
        translate_to_client("write", '{"path": "a.txt"}', "luna", owned, defs, ns)
        is None
    )
    assert (
        translate_to_client(
            "write",
            '{"path": "a.txt", "content": 3}',
            "luna",
            owned,
            defs,
            ns,
        )
        is None
    )
    # Family gate: unknown families fail open (never a wrong-family
    # rewrite); plain legs have no exec channel.
    assert (
        translate_to_client(
            "write",
            '{"path": "grammar-add.txt", "content": "hello-grammar"}',
            "unknown",
            owned,
            defs,
            ns,
        )
        is None
    )
    runner = (_spark_runner(),)
    from llms.proxy.client_tools import owned_tool_names as _owned

    s_owned = _owned(runner)
    s_defs = {t.name.lower(): t for t in runner if t.name}
    s_defs.update(nested_tool_defs(runner))
    assert (
        translate_to_client(
            "write",
            '{"path": "grammar-add.txt", "content": "hello-grammar"}',
            "luna",
            s_owned,
            s_defs,
            runner,
        )
        is None
    )
    # Same-name ownership wins when NO table row covers the pair: a
    # client `write` declaration on a family with no write row (dsh
    # declares one here — no write row ships for dsh) refuses the
    # translation so the owned path (argument correction) applies.
    # (On families WITH a write row — luna/claude — the row IS the
    # ownership decision for the pair and the translation ships.)
    from llms.proxy.ir import ToolDef

    own_write = ToolDef("write", "w", {})
    assert (
        translate_to_client(
            "write",
            '{"path": "x", "content": "y"}',
            "dsh",
            dict(owned, write="write"),
            dict(defs, write=own_write),
            ns + (own_write,),
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
        translate_to_client("shell", '{"workdir": "/tmp"}', "codex-plain", owned, defs)
        is None
    )
    assert (
        translate_to_client("shell", '{"command": "x"}', "codex-plain", {}, {}) is None
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
    from tests.test_client_tools import _luna_namespace

    tools = [
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "options": _luna_namespace().options,
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


def test_execute_rewrites_onto_nested_exec_channel():
    # Live luna apply_patch probes (2026-10-06, manual probe 8): the
    # model emits genuine `execute {"code": "await
    # tools.apply_patch(...)"}` — the SAME JavaScript the harness exec
    # orchestrator runs. The classifier must rewrite it onto the exec
    # channel (passthrough with the __exec_rewrite__ marker the replay
    # paths apply), not steer it: steering taught the channel three
    # turns running and the model never re-wrapped. Plain function
    # legs (spark: no exec channel) still steer generic.
    import json as _json

    from llms.proxy.client_tools import (
        dispatchable_names,
        nested_tool_defs,
        owned_tool_names,
    )
    from llms.proxy.compat import translate_to_client
    from llms.proxy.pipeline import _classify_calls
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES
    from tests.test_client_tools import _luna_namespace, _spark_runner

    js = "await tools.apply_patch('*** Begin Patch ***')"
    args = _json.dumps({"code": js})

    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    # Table level: inner JS replays verbatim as the channel input.
    assert translate_to_client("execute", args, "luna", owned, defs, ns) == (
        "exec",
        js,
    )
    # Classifier level: passthrough with the exec-rewrite marker.
    route = dispatchable_names(ns)
    (call,) = [{"call_id": "c1", "name": "execute", "arguments": args}]
    passed, steer = _classify_calls(
        [call], owned, defs, route, ns, GENUINE_TOOL_NAMES, "luna", "t1"
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec"]
    assert passed[0]["__exec_rewrite__"] == {"name": "exec", "input": js}
    # Blank code: no payload to re-wrap — generic steer, never a guess.
    blank = {"call_id": "c2", "name": "execute", "arguments": "{}"}
    passed, steer = _classify_calls(
        [blank], owned, defs, route, ns, GENUINE_TOOL_NAMES, "luna", "t1"
    )
    assert passed == []
    assert [c["name"] for c in steer] == ["execute"]
    # Plain leg: no exec channel — generic steer.
    runner = (_spark_runner(),)
    owned = owned_tool_names(runner)
    defs = {t.name.lower(): t for t in runner if t.name}
    defs.update(nested_tool_defs(runner))
    route = dispatchable_names(runner)
    passed, steer = _classify_calls(
        [dict(call)],
        owned,
        defs,
        route,
        runner,
        GENUINE_TOOL_NAMES,
        "codex-plain",
        "t1",
    )
    assert passed == []
    assert [c["name"] for c in steer] == ["execute"]


def test_edit_to_apply_patch_update_proven_live():
    # Round-4 proof (2026-10-07, luna leg): the Update-File grammar
    # ran `Script completed` with `FileChange update` (unified diff
    # `@@ -1 +1 @@\n-old-line\n+new-line\n`) and the workspace file
    # byte-exact `new-line\n`. The edit->apply_patch row ships on the
    # luna family only. `replaceAll` is refused (the Update grammar
    # addresses one hunk, never a global replace); empty/identical
    # old/new and missing fields steer generic.
    from llms.proxy.client_tools import nested_tool_defs, owned_tool_names
    from llms.proxy.compat import translate_to_client
    from tests.test_client_tools import _luna_namespace

    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    assert translate_to_client(
        "edit",
        '{"path": "f.txt", "oldString": "old-line", "newString": "new-line"}',
        "luna",
        owned,
        defs,
        ns,
    ) == (
        "exec",
        (
            "*** Begin Patch\n*** Update File: f.txt\n"
            "@@\n-old-line\n+new-line\n*** End Patch"
        ),
    )
    # replaceAll: the grammar cannot express it — no guess.
    assert (
        translate_to_client(
            "edit",
            '{"path": "f.txt", "oldString": "a", "newString": "b", "replaceAll": true}',
            "luna",
            owned,
            defs,
            ns,
        )
        is None
    )
    # Empty old / identical old-new: nothing to synthesize.
    assert (
        translate_to_client(
            "edit",
            '{"path": "f.txt", "oldString": "", "newString": "b"}',
            "luna",
            owned,
            defs,
            ns,
        )
        is None
    )
    assert (
        translate_to_client(
            "edit",
            '{"path": "f.txt", "oldString": "a", "newString": "a"}',
            "luna",
            owned,
            defs,
            ns,
        )
        is None
    )
    # Family gate: unknown families fail open.
    assert (
        translate_to_client(
            "edit",
            '{"path": "f.txt", "oldString": "a", "newString": "b"}',
            "unknown",
            owned,
            defs,
            ns,
        )
        is None
    )


def test_write_edit_classifier_rides_exec_rewrite_marker():
    # Classifier level (mirrors test_execute_rewrites_onto_nested_exec
    # _channel): genuine write/edit on luna passthrough renamed to
    # `exec` with the __exec_rewrite__ marker (the replay paths apply
    # it into the exec channel — a bare function_call named
    # apply_patch fails lookup), and genuine write on codex-plain
    # passthrough renamed to `exec_command` with __translated_args__
    # (plain function runner — replay swaps frame arguments).
    import json as _json

    from llms.proxy.client_tools import (
        dispatchable_names,
        nested_tool_defs,
        owned_tool_names,
    )
    from llms.proxy.pipeline import _classify_calls
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES
    from tests.test_client_tools import _luna_namespace, _spark_runner

    ns = (_luna_namespace(),)
    owned = owned_tool_names(ns)
    defs = {t.name.lower(): t for t in ns if t.name}
    defs.update(nested_tool_defs(ns))
    route = dispatchable_names(ns)
    (call,) = [
        {
            "call_id": "c1",
            "name": "write",
            "arguments": _json.dumps({"path": "w.txt", "content": "hi"}),
        }
    ]
    passed, steer = _classify_calls(
        [call], owned, defs, route, ns, GENUINE_TOOL_NAMES, "luna", "t1"
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec"]
    assert passed[0]["__exec_rewrite__"] == {
        "name": "exec",
        "input": "*** Begin Patch\n*** Add File: w.txt\n+hi\n*** End Patch",
    }
    (call,) = [
        {
            "call_id": "c2",
            "name": "edit",
            "arguments": _json.dumps(
                {"path": "w.txt", "oldString": "hi", "newString": "yo"}
            ),
        }
    ]
    passed, steer = _classify_calls(
        [call], owned, defs, route, ns, GENUINE_TOOL_NAMES, "luna", "t1"
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec"]
    assert passed[0]["__exec_rewrite__"]["input"] == (
        "*** Begin Patch\n*** Update File: w.txt\n@@\n-hi\n+yo\n*** End Patch"
    )
    runner = (_spark_runner(),)
    s_owned = owned_tool_names(runner)
    s_defs = {t.name.lower(): t for t in runner if t.name}
    s_defs.update(nested_tool_defs(runner))
    s_route = dispatchable_names(runner)
    (call,) = [
        {
            "call_id": "c3",
            "name": "write",
            "arguments": _json.dumps({"path": "s.txt", "content": "hi\n"}),
        }
    ]
    passed, steer = _classify_calls(
        [call],
        s_owned,
        s_defs,
        s_route,
        runner,
        GENUINE_TOOL_NAMES,
        "codex-plain",
        "t1",
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["exec_command"]
    assert passed[0]["__translated_args__"] == (
        '{"cmd": "printf \'%s\' \'hi\\n\' > s.txt"}'
    )


def test_write_to_shell_redirect_executes_byte_exact(tmp_path):
    # The synthesis must EXECUTE, not just classify: an earlier
    # heredoc shape (`printf %s > path <<'EOF' ...`) passed the
    # classifier but wrote 0 bytes at runtime (`printf` with no
    # argument prints empty — it never reads stdin; live tmux stress
    # 2026-10-09: six consecutive exit-0 turns left the file empty
    # and the model looped re-emitting `write`). The single-quoted
    # argument shape runs here through the harness shell path.
    import subprocess

    from llms.proxy.compat import _write_to_shell_redirect

    for content in ("hi\n", "it's\nmulti\nline\n", ""):
        out = tmp_path / "probe.txt"
        tra = _write_to_shell_redirect(
            {"path": str(out), "content": content}, ["cmd"]
        )
        assert tra is not None
        subprocess.run(["/bin/bash", "-lc", __import__("json").loads(tra)["cmd"]], check=True)
        assert out.read_text() == content


def test_write_to_shell_redirect_proven_live():
    # shell-redirect synthesis is verified live — spark leg,
    # genuine `function_call exec_command {"cmd": "printf
    # 'hello-shell\\\\n' > shell-write.txt"}` executed with `Process
    # exited with code 0`, file byte-exact `hello-shell\n` (od
    # verified; first spark-leg file ever created through any path).
    # The converter returns the client runner's `cmd` argument as a
    # JSON object (same contract as every other translation): a bare
    # command string is not valid arguments and steered the proxy's
    # own replayed frames as owned-but-invalid at the tail check
    # (live 2026-10-09: loop passed, tail failed closed, ~1/4 write
    # turns). The content rides a single-quoted printf argument —
    # never stdin (an earlier heredoc shape exited 0 but wrote 0
    # bytes; see test_write_to_shell_redirect_executes_byte_exact).
    # The write->exec_command row ships on codex-plain only.
    from llms.proxy.client_tools import nested_tool_defs, owned_tool_names
    from llms.proxy.compat import translate_to_client
    from tests.test_client_tools import _spark_runner

    runner = (_spark_runner(),)
    owned = owned_tool_names(runner)
    defs = {t.name.lower(): t for t in runner if t.name}
    defs.update(nested_tool_defs(runner))
    assert translate_to_client(
        "write",
        '{"path": "shell-write.txt", "content": "hello-shell\\n"}',
        "codex-plain",
        owned,
        defs,
        runner,
    ) == (
        "exec_command",
        '{"cmd": "printf \'%s\' \'hello-shell\\n\' > shell-write.txt"}',
    )
    # No delimiter games: content with quotes/newlines stays inline.
    assert (
        translate_to_client(
            "write",
            '{"path": "x.txt", "content": "it\'s\\nEOF\\nok"}',
            "codex-plain",
            owned,
            defs,
            runner,
        )
        == (
            "exec_command",
            '{"cmd": "printf \'%s\' \'it\'\\\\\'\'s\\nEOF\\nok\' > x.txt"}',
        )
    )
    # Missing/non-string fields: no guess.
    assert (
        translate_to_client(
            "write", '{"path": "x.txt"}', "codex-plain", owned, defs, runner
        )
        is None
    )
    # Family gate: the luna-owned WRITE still takes the apply_patch
    # row there (the shell row is codex-plain-only, and luna owns an
    # exec_command runner too — the family tag, not runner shape,
    # selects). Unknown families fail open.
    from tests.test_client_tools import _luna_namespace

    ns = (_luna_namespace(),)
    l_owned = owned_tool_names(ns)
    l_defs = {t.name.lower(): t for t in ns if t.name}
    l_defs.update(nested_tool_defs(ns))
    assert translate_to_client(
        "write",
        '{"path": "x.txt", "content": "y"}',
        "luna",
        l_owned,
        l_defs,
        ns,
    ) == (
        "exec",
        "*** Begin Patch\n*** Add File: x.txt\n+y\n*** End Patch",
    )
    assert (
        translate_to_client(
            "write",
            '{"path": "x.txt", "content": "y"}',
            "unknown",
            owned,
            defs,
            runner,
        )
        is None
    )
    # No cmd-runner at all: no guess.
    assert (
        translate_to_client(
            "write",
            '{"path": "x.txt", "content": "y"}',
            "codex-plain",
            {},
            {},
            runner,
        )
        is None
    )


def _claude_tools():
    # REAL 23-tool fixture: claude-cli 2.1.293 declaration captured live
    # 2026-10-08 (compat-matrix-claude-proxy.log steer gates
    # names=[...23...] family=claude). Bash takes {command},
    # Write {file_path, content}, Read {file_path}, Edit
    # {file_path, old_string, new_string} — key renames vs the
    # genuine-12 (command/file_path/...) are the rows' exact contract.
    import json as _json

    from llms.proxy.ir import ToolDef

    with open("/home/uqmm/.claude/jobs/8c5ef74f/tmp/claude-schema.jsonl") as f:
        body = _json.loads(f.readline())["body"]
    return tuple(
        ToolDef(
            t["name"],
            t.get("description", ""),
            t.get("input_schema", {}),
            kind=t.get("type", "function"),
        )
        for t in body["tools"]
        if isinstance(t, dict)
    )


def test_claude_rows_translate_genuine_onto_native_tools():
    # Task 4/5 (plan 2026-10-06): the claude leg declares native tools
    # under the SAME names the genuine overlay uses (Bash vs shell is
    # the only true rename; Read/Write/Edit collide case-insensitively
    # with read/write/edit) — the same-name guard must not refuse the
    # rows: it fires on exact-case declarations only
    # (Bash/Read/Write/Edit guard their own case; the model's lowercase
    # genuine emissions shell/read/write/edit translate).
    import json as _json

    from llms.proxy.client_tools import (
        dispatchable_names,
        nested_tool_defs,
        owned_tool_names,
    )
    from llms.proxy.compat import CLAUDE, translate_to_client
    from llms.proxy.pipeline import _classify_calls
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    tools = _claude_tools()
    owned = owned_tool_names(tools)
    defs = {t.name.lower(): t for t in tools if t.name}
    defs.update(nested_tool_defs(tools))
    route = dispatchable_names(tools)
    # shell{command} -> Bash{command}: verbatim key (capture: Bash
    # requires ["command"]; genuine shell carries command+extras).
    assert translate_to_client(
        "shell", '{"command": "echo tool-ok"}', CLAUDE, owned, defs, tools
    ) == ("Bash", '{"command": "echo tool-ok"}')
    # read{path} -> Read{file_path}: key rename (capture: Read
    # requires ["file_path"]).
    assert translate_to_client(
        "read", '{"path": "f.txt"}', CLAUDE, owned, defs, tools
    ) == (
        "Read",
        '{"file_path": "f.txt"}',
    )
    # write{path,content} -> Write{file_path,content}: key rename
    # (capture: Write requires ["file_path", "content"]).
    assert translate_to_client(
        "write",
        '{"path": "f.txt", "content": "hi"}',
        CLAUDE,
        owned,
        defs,
        tools,
    ) == ("Write", '{"file_path": "f.txt", "content": "hi"}')
    # edit{path,oldString,newString} -> Edit{file_path,old_string,
    # new_string}: key renames (capture: Edit requires
    # ["file_path", "old_string", "new_string"]).
    assert translate_to_client(
        "edit",
        '{"path": "f", "oldString": "a", "newString": "b"}',
        CLAUDE,
        owned,
        defs,
        tools,
    ) == ("Edit", '{"file_path": "f", "old_string": "a", "new_string": "b"}')
    # Missing keys / wrong family: None (fail open, never guess).
    assert (
        translate_to_client("shell", '{"workdir": "/tmp"}', CLAUDE, owned, defs, tools)
        is None
    )
    assert (
        translate_to_client("read", '{"path": "f"}', "unknown", owned, defs, tools)
        is None
    )
    assert (
        translate_to_client("write", '{"path": "f"}', CLAUDE, owned, defs, tools)
        is None
    )
    # Classifier level: genuine shell on claude passthrough renamed to
    # Bash (the replay swaps frame arguments — plain function route).
    (call,) = [
        {
            "call_id": "c1",
            "name": "shell",
            "arguments": _json.dumps({"command": "echo tool-ok"}),
        }
    ]
    passed, steer = _classify_calls(
        [call], owned, defs, route, tools, GENUINE_TOOL_NAMES, CLAUDE, "t1"
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["Bash"]
    assert passed[0]["__translated_args__"] == '{"command": "echo tool-ok"}'


def test_synthesized_messages_tool_use_feeds_classifier():
    # Live claude 2026-10-08: the CLI emitted `shell` tool_use blocks
    # 11x and got `No such tool available: shell` from its OWN
    # dispatcher each time — the proxy replayed the genuine name
    # downstream. Root cause: on the messages leg the synthesize path
    # folds upstream SSE into a messages JSON body (content[] with
    # tool_use, NO output[]), and `_genuine_calls_in` only scanned
    # `output[]` — ([], []), silently dropped, raw name rode
    # downstream. A synthesized messages body carrying a genuine call
    # must classify like a wire responses body (translate onto the
    # client tool, same markers the replay paths apply).
    from fastapi.responses import JSONResponse

    from llms.proxy.compat import CLAUDE
    from llms.proxy.pipeline import _genuine_calls_in
    from llms.proxy.zen_tools import GENUINE_TOOL_NAMES

    tools = _claude_tools()
    msg_body = {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "shell",
                "input": {"command": "echo shape-ok"},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    passed, steer = _genuine_calls_in(
        JSONResponse(status_code=200, content=msg_body),
        {"Bash"},
        client_tools=tools,
        genuine_names=GENUINE_TOOL_NAMES,
        family=CLAUDE,
        trace_id="t1",
    )
    assert steer == []
    assert [c["name"] for c in passed] == ["Bash"]
    assert passed[0]["__translated_args__"] == '{"command": "echo shape-ok"}'


def test_schema_audit_dropped_keys_pinned():
    # Task 8 schema audit (vs zen_tools.py GENUINE_TOOLS, 2026-10-07):
    # every compat row drops or refuses specific genuine keys. This
    # test pins each drop so a future schema widening (new required
    # key upstream) fails LOUDLY here instead of silently losing
    # data on the translation path.
    from llms.proxy.compat import (
        _edit_to_apply_patch,
        _edit_to_claude_edit,
        _execute_to_exec_channel,
        _read_to_claude_read,
        _shell_to_claude_bash,
        _shell_to_cmd,
        _write_to_apply_patch,
        _write_to_claude_write,
        _write_to_shell_redirect,
    )
    from llms.proxy.zen_tools import GENUINE_TOOLS

    schemas = {t["name"]: t["parameters"] for t in GENUINE_TOOLS}
    # shell: only `command` survives; workdir/timeout/background drop
    # (client cmd-runner takes cmd alone — the runner's cwd applies).
    assert set(schemas["shell"]["properties"]) == {
        "command",
        "workdir",
        "timeout",
        "background",
    }
    assert (
        _shell_to_cmd(
            {
                "command": "cat f",
                "workdir": "/tmp",
                "timeout": 5000,
                "background": True,
            },
            ["cmd"],
        )
        == '{"cmd": "cat f"}'
    )
    # execute: `code` replays verbatim; no keys exist to drop.
    assert set(schemas["execute"]["properties"]) == {"code"}
    assert (
        _execute_to_exec_channel({"code": "await tools.x()"}, []) == "await tools.x()"
    )
    # write: path+content both consumed; unknown extras ignored.
    assert set(schemas["write"]["properties"]) == {"path", "content"}
    assert (
        _write_to_apply_patch({"path": "a", "content": "b", "extra": 1}, [])
        == "*** Begin Patch\n*** Add File: a\n+b\n*** End Patch"
    )
    # edit: path+oldString+newString consumed; replaceAll REFUSED
    # (not dropped — a global replace has no Update-hunk form).
    assert set(schemas["edit"]["properties"]) == {
        "path",
        "oldString",
        "newString",
        "replaceAll",
    }
    assert (
        _edit_to_apply_patch(
            {
                "path": "f",
                "oldString": "a",
                "newString": "b",
                "replaceAll": False,
            },
            [],
        )
        == "*** Begin Patch\n*** Update File: f\n@@\n-a\n+b\n*** End Patch"
    )
    assert (
        _edit_to_apply_patch(
            {
                "path": "f",
                "oldString": "a",
                "newString": "b",
                "replaceAll": True,
            },
            [],
        )
        is None
    )
    # Spark shell synthesis puts path+content into the `cmd`
    # argument (JSON object); the content rides a single-quoted
    # printf argument (never stdin — the heredoc shape wrote 0 bytes).
    assert (
        _write_to_shell_redirect({"path": "s.txt", "content": "hi\n"}, ["cmd"])
        == '{"cmd": "printf \'%s\' \'hi\\n\' > s.txt"}'
    )
    # Claude rows (capture: claude-cli 2.1.293, 23 tools): per-entry
    # drops/renames vs the genuine-12 above.
    # shell->Bash: `command` verbatim; workdir/timeout/background drop
    # (Bash carries its own timeout/description — never synthesized
    # from the overlay payload; the client defaults apply).
    assert (
        _shell_to_claude_bash(
            {"command": "echo hi", "workdir": "/tmp", "timeout": 1},
            ["command"],
        )
        == '{"command": "echo hi"}'
    )
    assert _shell_to_claude_bash({"workdir": "/tmp"}, ["command"]) is None
    # read->Read: `path` -> `file_path` rename; offset/limit drop (the
    # client defaults apply — a whole-file read is what the overlay
    # asked for).
    assert set(schemas["read"]["properties"]) == {"path", "offset", "limit"}
    assert (
        _read_to_claude_read({"path": "f", "offset": 3, "limit": 10}, ["file_path"])
        == '{"file_path": "f"}'
    )
    assert _read_to_claude_read({"path": "f"}, ["other"]) is None
    # write->Write: `path` -> `file_path` rename, `content` verbatim.
    assert (
        _write_to_claude_write(
            {"path": "f", "content": "hi", "extra": 1},
            ["file_path", "content"],
        )
        == '{"file_path": "f", "content": "hi"}'
    )
    assert _write_to_claude_write({"path": "f"}, ["file_path", "content"]) is None
    # edit->Edit: `path`/`oldString`/`newString` renames;
    # replaceAll REFUSED (same discipline as _edit_to_apply_patch —
    # the client `replace_all` defaults False with different
    # semantics, and a synthesized global replace mis-fires).
    assert (
        _edit_to_claude_edit(
            {"path": "f", "oldString": "a", "newString": "b"},
            ["file_path", "old_string", "new_string"],
        )
        == '{"file_path": "f", "old_string": "a", "new_string": "b"}'
    )
    assert (
        _edit_to_claude_edit(
            {"path": "f", "oldString": "a", "newString": "b", "replaceAll": True},
            ["file_path", "old_string", "new_string"],
        )
        is None
    )
