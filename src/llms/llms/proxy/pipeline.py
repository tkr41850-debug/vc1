from __future__ import annotations

import asyncio
import json
import math
import os
import time

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llms.proxy import dedup, sessions
from llms.proxy.admin_hub import get_hub
from llms.proxy.affinity import bucket_for
from llms.proxy.client_tools import (
    build_tool_notice,
    convert_call_name,
    missing_required_keys,
    owned_tool_names,
)
from llms.proxy.config import Settings
from llms.proxy.forward import forward, parse_body
from llms.proxy.ir import RequestIR
from llms.proxy.logging import log_ingress, log_upstream, new_trace_id, setup_logging
from llms.proxy.providers import RecentRequest
from llms.proxy.rate_limit import classify
from llms.proxy.router import ENDPOINT_PATH, pick, resolve_alias
from llms.proxy.stream_translate import (
    SseFramer,
    new_chat_id,
    new_msg_id,
    new_resp_id,
)
from llms.proxy.translate import (
    from_chat,
    from_messages,
    from_responses,
    to_zen_chat,
    to_zen_messages,
    to_zen_responses,
    with_model,
)
from llms.proxy.translate_response import (
    EMITTERS,
    convert_response,
    deltas_to_response_ir,
)
from llms.proxy.zen_fingerprint import note_free_tier_error
from llms.proxy.zen_headers import build_zen_headers, stable_session_id
from llms.proxy.zen_prompts import TITLE_PREFIX
from llms.proxy.zen_tools import GENUINE_TOOL_NAMES, GENUINE_TOOLS

logger = setup_logging()

FROM = {"responses": from_responses, "chat": from_chat, "messages": from_messages}
TO = {"responses": to_zen_responses, "chat": to_zen_chat, "messages": to_zen_messages}
DEFAULT_MODEL_ATTR = {
    "responses": "default_model",
    "chat": "default_chat_model",
    "messages": "default_messages_model",
}


def _convert_for(ingress: str, egress: str, model: str):
    if ingress == egress:
        return None
    return lambda payload: convert_response(egress, ingress, payload, model)


def _stream_for(ingress: str, egress: str, model: str):
    if ingress == egress:
        return None
    if (ingress, egress) not in {
        ("chat", "responses"),
        ("responses", "chat"),
        ("messages", "responses"),
        ("responses", "messages"),
        ("chat", "messages"),
        ("messages", "chat"),
    }:
        return None
    return (ingress, egress, model)


def _deltas_with_pending(parser, lines):
    """Yield parser deltas plus pending-call flush (fold path only).

    Lives here (not in the parser module) so the incremental emitter
    path can never call it by accident: only _synthesize_for drains
    pending calls, after the whole stream parsed.
    """
    framer = SseFramer()
    for line in lines:
        for payload in framer.feed(line):
            yield from parser.feed_payload(payload)
    for payload in framer.finish():
        yield from parser.feed_payload(payload)
    flush = getattr(parser, "flush_pending_calls", None)
    if callable(flush):
        yield from flush()
    done = parser.finish()
    if done is not None:
        yield done


def _synthesize_for(ingress: str, egress: str, model: str):
    """Fold upstream SSE into one downstream JSON body (responses leg only).

    Anonymous Zen requires stream:true even for single-shot callers; the
    deltas accumulate into a ResponseIR that the normal response emitters
    render in the ingress dialect.
    """
    if egress != "responses":
        return None

    def run(lines, trace_id):
        from llms.proxy.stream_translate import STREAM_PARSERS

        parser = STREAM_PARSERS[egress]()
        deltas = list(_deltas_with_pending(parser, lines))
        return EMITTERS[ingress](
            deltas_to_response_ir(
                iter(deltas),
                model,
                getattr(parser, "done_args", None) or {},
                getattr(parser, "names", None) or {},
                getattr(parser, "custom_items", None) or frozenset(),
            ),
            model,
        )

    return run


STEER_MAX_ITERS = int(os.getenv("ZEN_STEER_MAX_ITERS", "3"))


def _inflight_track(request, provider_id: str | None) -> None:
    """Increment the provider's in-flight count (§3, event loop only)."""
    if not provider_id:
        return
    try:
        registry = getattr(request.app.state, "providers", None)
        if registry is not None:
            registry.runtime(provider_id).in_flight += 1
    except Exception:
        pass


def _inflight_release(request, provider_id: str | None) -> None:
    """Decrement with a max(0, …) guard; schedules a throttled push."""
    if not provider_id:
        return
    try:
        registry = getattr(request.app.state, "providers", None)
        if registry is None:
            return
        rt = registry.runtime(provider_id)
        rt.in_flight = max(0, rt.in_flight - 1)
        hub = getattr(request.app.state, "admin_hub", None)
        publish = getattr(hub, "publish_throttled", None)
        if callable(publish):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            loop.create_task(publish("providers"))
    except Exception:
        pass


def _wrap_inflight(
    request,
    response,
    provider_id: str | None,
    *,
    model: str | None = None,
    warp_idx: int | None = None,
    started: float | None = None,
):
    """Wrap a StreamingResponse so body completion releases in-flight.

    Covers all four stream generators in forward.py at once; `finally`
    covers abrupt disconnect. JSON/synthesize legs decrement via
    _inflight_release after _record_usage instead.

    A fully-consumed stream also records the recent-request ring entry
    (model/warp_idx/started supplied by the call site): without this only
    JSON legs ever appear in Last requests (live), so a successful
    streaming probe promotes probation -> ready yet stays invisible.
    Disconnects record nothing — the request never completed.
    """
    if provider_id is None or not isinstance(response, StreamingResponse):
        return response
    iterator = response.body_iterator

    async def _tracking_iterator():
        completed = False
        try:
            async for chunk in iterator:
                yield chunk
            completed = True
        finally:
            _inflight_release(request, provider_id)
            # Stream probes settle at body exhaustion: a fully-consumed
            # probe promotes probation -> ready (the outcome-429 path ran
            # at headers time, so settle only needs the success arm here).
            # Disconnects (CancelledError / early close) stay armed — the
            # next request re-probes instead of riding an unverified exit.
            if completed:
                try:
                    registry = getattr(request.app.state, "providers", None)
                    if registry is not None:
                        _settle_probation(request, registry, provider_id, response)
                        if model is not None:
                            outcome = getattr(
                                getattr(request, "state", None),
                                "stream_outcome",
                                None,
                            )
                            error = (
                                ""
                                if outcome in (None, "completed")
                                else f"stream {outcome}"
                            )
                            elapsed_ms = (
                                (time.monotonic() - started) * 1000.0
                                if started is not None
                                else 0.0
                            )
                            await registry.runtime(provider_id).record(
                                RecentRequest(
                                    ts=time.time(),
                                    model=model,
                                    status=(
                                        response.status_code
                                        if hasattr(response, "status_code")
                                        else 0
                                    ),
                                    ms=elapsed_ms,
                                    warp_idx=warp_idx,
                                    error=str(error)[:200],
                                )
                            )
                except Exception:
                    pass

    response.body_iterator = _tracking_iterator()
    return response


def _valid_json(args: str) -> bool:
    """True when a steer-followup arguments string survives the upstream
    validator (non-blank valid JSON). Shared with the streaming fold."""
    if not isinstance(args, str) or not args.strip():
        return False
    try:
        json.loads(args)
        return True
    except Exception:
        return False


def _classify_calls(
    calls: list[dict],
    owned: dict[str, str],
    client_defs: dict[str, object],
    route: dict[str, tuple[str, str]] | None = None,
    client_tools: tuple = (),
    genuine_names: tuple = (),
    family: str = "unknown",
    trace_id: str = "",
) -> tuple[list[dict], list[dict]]:
    """Split folded model calls into (passthrough, steer) by client ownership.

    A call whose lowered name the client declared passes through with its
    name converted to the declared casing — provided its arguments carry
    the client's required keys. Ownership alone is not enough: the name
    must also be dispatchable (present in the route map). The
    deferred-namespace container itself (a bare `functions` call) is
    owned but has no dispatch route, so it steers like an undeclared
    name instead of replaying a turn the harness fails client-side.
    Calls whose dispatch route is `custom_tool_call` pass through
    re-typed: on legs with a documented `exec` orchestrator the nested
    call rewrites into that channel (a `custom_tool_call` named `exec`
    whose input is the JS invocation — the harness only executes
    nested tools through it); otherwise via rewrite_steered_call (the
    harness dispatches by item type, so a Custom-route call arriving
    as `function_call` would fail lookup client-side). Owned-but-invalid
    calls (missing required keys, unparseable/non-object args) steer
    like undeclared ones: the client cannot execute them, so the
    redirect + re-request recovers. Genuine-overlay names with usable
    arguments (`shell` with `command`, `read` with `path`, ...) rewrite
    onto the client's equivalent tool instead of steering a name the
    client declared under its own name for the same capability. A
    `shell` emission the client declared under its own shell-runner
    name dedups against that runner's own retransmit (same harness
    turn, same call_id): the model emits both the alias and the client
    name for one action, and the client executes the first while its
    own router rejects the duplicate — the duplicate steers, never
    replays. Pure helper (shared semantics with the streaming fold in
    forward.py). Takes folded call dicts (not a Response) so both the
    synthesize path (via _genuine_calls_in) and the streaming fold (via
    fold_stream_calls) share one classifier.
    """
    from llms.proxy.client_tools import (
        exec_channel_source,
        rewrite_steered_call,
        split_call_name,
        translate_genuine_call,
    )
    from llms.proxy.compat import rewrap_bare_patch_exec_input
    from llms.proxy.ir import ToolDef

    passthrough: list[dict] = []
    steer: list[dict] = []
    for call in calls:
        name = call.get("name")
        # Code-mode `ns__name` calls split to the bare name first: the
        # harness dispatches by ToolName (with_default_namespace), so
        # `default.exec_command` is `exec_command` in the default
        # namespace — ownership, routing, and casing all key on the
        # bare name (live: `default.exec_command` steered forever as
        # undeclared). A non-`default` namespace rides
        # `__route_namespace__` so the replay can re-attach it (the
        # harness reads the item's own namespace field); `default`
        # never re-attaches (the harness fills its own default, and a
        # foreign `default` value poisons lookup client-side).
        ns, bare = split_call_name(name) if isinstance(name, str) else ("", None)
        lowered = bare.lower() if isinstance(bare, str) else None
        routable = not route or (lowered is not None and lowered in route)
        if lowered is not None and lowered in owned and routable:
            # Custom-route calls (raw-string payload: patch text, JS
            # source) skip the JSON required-keys check — it would
            # always reject them. Nested names documented in the exec
            # description rewrite into the exec channel first (a bare
            # nested call fails lookup); otherwise re-type and pass
            # through with the input preserved verbatim.
            item_type = (
                route.get(lowered, ("function_call", lowered))[0]
                if route
                else "function_call"
            )
            if item_type == "custom_tool_call":
                fixed = {**call, "name": convert_call_name(name, owned)}
                args_text = call.get("arguments", "")
                args_text = args_text if isinstance(args_text, str) else ""
                # Empty Custom-route payload (live luna, 2026-10-06: the
                # model emits `custom_tool_call exec` with EMPTY input
                # for an apply_patch turn, and a stray done frame folds
                # it as `default.exec {}`): the client cannot execute an
                # input-less orchestrator call — steer with the
                # exec-channel correction so the retry carries a real
                # payload. Falls through to the steer path below.
                payload_text = call.get("input", args_text)
                if not (isinstance(payload_text, str) and payload_text.strip()):
                    steer.append(call)
                    continue
                # Nested names documented in the exec description only
                # execute through the `exec` orchestrator — even
                # Custom-route ones (apply_patch): a bare nested call
                # fails lookup, so the rewrite rides a marker the replay
                # paths apply (never as the call name — they match
                # frames by the emitted name). Other Custom-route calls
                # re-type with the input preserved verbatim.
                rewritten = None
                if lowered == "exec":
                    # Mechanism 13 (live luna 2026-10-07, round 5): the
                    # model drops the `await tools.apply_patch(...)`
                    # wrapper and emits the patch text as the whole
                    # exec input — the harness JS parser rejects it
                    # (`SyntaxError`). Re-wrap the bare marker text so
                    # the channel executes it (passthrough with the
                    # marker the replay paths apply).
                    rewrapped = rewrap_bare_patch_exec_input(payload_text)
                    if rewrapped is not None:
                        rewritten = {
                            **fixed,
                            "__exec_rewrite__": {
                                "name": "exec",
                                "input": rewrapped,
                            },
                        }
                if rewritten is None:
                    source = exec_channel_source(
                        fixed.get("name", ""), args_text, client_tools
                    )
                    if source is not None:
                        rewritten = {
                            **fixed,
                            "__exec_rewrite__": {"name": "exec", "input": source},
                        }
                    else:
                        rewritten = rewrite_steered_call(
                            fixed,
                            args_text,
                            route if route is not None else {},
                        )
                if rewritten is not None:
                    passthrough.append(rewritten)
                else:
                    fixed["type"] = "custom_tool_call"
                    passthrough.append(fixed)
                continue
            tool = client_defs.get(lowered)
            args = call.get("arguments", "")
            if (
                isinstance(tool, ToolDef)
                and missing_required_keys(args if isinstance(args, str) else "", tool)
                == []
            ):
                fixed = {**call, "name": convert_call_name(name, owned)}
                if ns and ns.lower() != "default":
                    fixed["__route_namespace__"] = ns
                if route is not None:
                    rewritten = rewrite_steered_call(
                        fixed, args if isinstance(args, str) else "", route
                    )
                    if rewritten is not None:
                        fixed = rewritten
                if "__exec_rewrite__" not in fixed:
                    # Nested names documented in the exec description
                    # only execute through the `exec` orchestrator —
                    # even Function-route ones (luna exec_command):
                    # a bare nested call fails lookup.
                    source = exec_channel_source(
                        fixed.get("name", ""),
                        args if isinstance(args, str) else "",
                        client_tools,
                    )
                    if source is not None:
                        fixed["__exec_rewrite__"] = {
                            "name": "exec",
                            "input": source,
                        }
                passthrough.append(fixed)
                continue
        # Genuine-overlay names with usable arguments (the model was
        # offered shell/write/edit upstream): rewrite onto the
        # client's equivalent tool instead of steering a name the
        # client declared under its own name for the same capability
        # (live: `shell` steered forever on the spark leg while the
        # client declares `exec_command` for the same thing). `read`
        # never rewrites (a shared `path` key is not a file reader) —
        # it steers, and the redirect teaches shell/command.
        args_text = call.get("arguments", "")
        args_text = args_text if isinstance(args_text, str) else ""
        translated = translate_genuine_call(
            bare if isinstance(bare, str) else "",
            args_text,
            owned,
            client_defs,
            family,
            client_tools,
        )
        if translated is not None:
            declared, new_args = translated
            # Same-turn duplicate retransmit: the model emits the alias
            # AND the client runner name for one action (live spark
            # harness 2026-10-06: translated `shell` +
            # `exec_command {"cmd": ...}` under one call_id). The
            # client executes the alias translation and rejects the
            # same-name retransmit as a duplicate — it steers, never
            # replays (a same-call_id double-execution would run the
            # command twice).
            dup = next(
                (
                    p
                    for p in passthrough
                    if call.get("call_id")
                    and call.get("call_id") in (p.get("call_id"), p.get("id"))
                ),
                None,
            )
            if dup is not None:
                steer.append(call)
                continue
            passed = {**call, "name": declared, "arguments": new_args}
            # Nested names documented in the exec description only
            # execute through the `exec` orchestrator — even a
            # genuine->client translation landing on one (luna: `shell`
            # onto nested `exec_command`): a bare function_call fails
            # lookup. The marker rides instead of __translated_args__
            # (the replay's argument-swap branches apply the translated
            # payload, which the JS input already carries — keeping both
            # renames the done frame and refolds a Frankenstein call).
            # The channel target itself (`exec`) is already the
            # orchestrator: its input IS the payload (a genuine
            # `execute {"code"}` table translation lands here with the
            # JS as arguments — no nested name to derive it from, and
            # feeding it back through exec_channel_source would ask
            # for a nested tool named `exec` that does not exist).
            if declared.lower() == "exec" and isinstance(new_args, str):
                passed["__exec_rewrite__"] = {"name": "exec", "input": new_args}
                passthrough.append(passed)
                continue
            source = exec_channel_source(passed.get("name", ""), new_args, client_tools)
            if source is not None:
                passed["__exec_rewrite__"] = {"name": "exec", "input": source}
            else:
                # The replay swaps frame arguments only when the call
                # was renamed (genuine name -> client name): verbatim
                # passthrough keeps the frames' own payload.
                passed["__translated_args__"] = new_args
            passthrough.append(passed)
            continue
        # Untranslatable genuine-overlay name: fail open (steer with
        # the existing correction) + error breadcrumb so the table
        # gains a proven 1:1 row later (schema audit reviews these).
        try:
            from llms.proxy.compat import log_untranslatable as _log_no_entry
        except Exception:
            _log_no_entry = None
        if _log_no_entry is not None:
            gname = bare if isinstance(bare, str) else ""
            gnames = {str(n).lower() for n in (genuine_names or ())}
            if gname.lower() in gnames:
                _log_no_entry(trace_id, gname, family)
        steer.append(call)
    return passthrough, steer


def _genuine_calls_in(
    response: Response,
    client_names: set[str],
    *,
    client_tools: tuple = (),
    genuine_names: tuple = (),
    family: str = "unknown",
    trace_id: str = "",
) -> tuple[list[dict], list[dict]]:
    """Split response function_calls into (passthrough, steer) candidates.

    Backward-compatible detector plus ownership classification in one
    pass: undeclared names steer (existing case-insensitive rule), while
    client-declared names split further by required-keys validity via
    _classify_calls (valid → passthrough with converted casing).
    Without client_tools every call steers (legacy detector behavior).
    """
    if not isinstance(response, JSONResponse) or response.status_code >= 400:
        return [], []
    try:
        payload = json.loads(response.body.decode())
    except Exception:
        return [], []
    if not isinstance(payload, dict):
        return [], []
    calls = []
    lowered = {n.lower() for n in client_names if isinstance(n, str)}
    for item in payload.get("output", []) or []:
        # Wire Custom items (non-streaming upstream shape: type
        # custom_tool_call, payload in `input`) feed the classifier
        # like the streaming fold's entries — the Custom-route branch
        # judges `input`, never JSON arguments. Skipping them here
        # silently drops the call: neither passthrough nor steer, so
        # the raw payload rides downstream (live luna 2026-10-07: 4x
        # bare-patch harness SyntaxError while the rewrap arm sat
        # idle one layer down).
        if not isinstance(item, dict):
            continue
        if item.get("type") == "custom_tool_call":
            if not client_tools:
                continue
            calls.append(item)
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        if isinstance(name, str) and name.lower() in lowered:
            if not client_tools:
                continue
            calls.append(item)
            continue
        calls.append(item)
    if not client_tools:
        return [], calls
    from llms.proxy.client_tools import dispatchable_names, nested_tool_defs

    owned = owned_tool_names(client_tools)
    defs = {t.name.lower(): t for t in client_tools if t.name}
    defs.update(nested_tool_defs(client_tools))
    return _classify_calls(
        calls,
        owned,
        defs,
        dispatchable_names(client_tools),
        client_tools,
        genuine_names,
        family,
        trace_id,
    )


def _steer_output_for(
    call, args, client_names, owned, defs, client_tools=(), genuine_names=()
):
    """Redirect text for one steered call.

    Owned-but-invalid calls get an argument correction (the tool exists;
    only the arguments were wrong). Undeclared genuine-overlay names
    that a same-capability client tool can serve (live: `read` on the
    spark leg, where only `exec_command` reads files) get a directed
    correction naming that tool and its argument shape — the generic
    not-available list never converted the model. All other undeclared
    names keep the not-available text with the client tool list. Pure
    dispatch — wording lives in client_tools.build_tool_redirect. The
    nested-tool (`exec` orchestrator) guidance appends only on legs
    whose harness actually exposes that channel — on plain function
    legs it would contradict the usable tool list in the same message.
    """
    from llms.proxy.client_tools import (
        build_tool_redirect,
        display_tool_names,
        has_nested_exec_channel,
        steer_to_equivalent,
    )

    equivalent = steer_to_equivalent(
        str(call.get("name", "")), args, owned, defs, genuine_names, client_tools
    )
    if equivalent is not None:
        return equivalent
    correction = build_tool_redirect(
        str(call.get("name", "")), args, owned, defs, genuine_names
    )
    if correction is not None:
        return correction
    display = display_tool_names(client_tools) if client_tools else sorted(client_names)
    text = f"Tool '{call.get('name', '')}' is not available in this session." + (
        f" Use one of these tools instead: {', '.join(sorted(display))}."
        if display
        else ""
    )
    if has_nested_exec_channel(client_tools):
        text += (
            " To run a nested tool (exec_command, apply_patch, ...), emit "
            + "ONE `custom_tool_call` item named `exec` whose `input` is "
            + "JavaScript calling it on the `tools` object "
            + '(e.g. `"input": "await tools.exec_command({cmd: "cat '
            + 'FILE"})"`).'
        )
    return text + " If none fits, answer directly without calling a tool."


async def _steer_genuine_calls(
    response: Response,
    *,
    client,
    url: str,
    headers: dict,
    outbound: dict,
    synthesize,
    trace_id: str,
    client_names: set[str],
    client_tools: tuple = (),
    genuine_names: tuple = (),
    family: str = "unknown",
) -> tuple[Response, dict]:
    """Answer undeclared tool calls with a redirect error and re-request.

    Only calls the client cannot execute steer: undeclared names, plus
    client-owned names whose arguments miss required keys or are not
    valid JSON (the client would fail the turn, so the redirect +
    re-request recovers). A call naming a client-declared tool with
    valid args passes straight back with its name converted to the
    declared casing — even one colliding with a genuine tool name in a
    different case — for the client to resolve. Only for
    non-streaming downstream (streaming passes calls through —
    mid-stream steering is a follow-up). Bounded; usage attributes the
    final turn only.
    """
    for _ in range(STEER_MAX_ITERS):
        passed, steer_calls = _genuine_calls_in(
            response,
            client_names,
            client_tools=client_tools,
            genuine_names=genuine_names,
            family=family,
            trace_id=trace_id,
        )
        owned = owned_tool_names(client_tools)
        defs = {t.name.lower(): t for t in client_tools if t.name}
        from llms.proxy.client_tools import nested_tool_defs as _nested_defs

        defs.update(_nested_defs(client_tools))
        if passed and not steer_calls:
            # Every call is client-owned and valid: convert casing on the
            # response body so the client dispatches its own declarations.
            # A Custom-route call also re-types to custom_tool_call with
            # its arguments moved to input (the harness dispatches by
            # item type — a Function-typed apply_patch fails lookup).
            try:
                payload = json.loads(response.body.decode())
            except Exception:
                break
            by_id = {}
            for item in payload.get("output", []) or []:
                # Wire Custom items carry the executable payload in
                # `input` (no call_id when the upstream announce lacks
                # one — match by id, same key the classifier's folded
                # entries carry). Skipping them here drops the
                # classifier's __exec_rewrite__ marker: the raw input
                # replays downstream (live luna 2026-10-07: 3x
                # bare-patch harness SyntaxError on the fixed proxy).
                if not isinstance(item, dict) or item.get("type") not in (
                    "function_call",
                    "custom_tool_call",
                ):
                    continue
                key = str(
                    item.get("call_id") or item.get("id") or item.get("item_id") or ""
                )
                if key:
                    by_id[key] = item
            for call in passed:
                key = str(
                    call.get("call_id") or call.get("id") or call.get("item_id") or ""
                )
                if key in by_id:
                    # Classifier rewrites (genuine name -> client name
                    # with translated arguments, nested name ->
                    # exec-channel custom call) apply alongside the
                    # casing convert: the downstream harness only
                    # executes the rewritten form.
                    by_id[key]["name"] = call["name"]
                    # Classifier markers never reach the client (popped):
                    # __translated_args__ (genuine->client argument
                    # rewrite) applies to the replayed arguments;
                    # __route_namespace__ re-attaches a non-default
                    # namespace for the dispatch replay.
                    translated = call.pop("__translated_args__", None)
                    if isinstance(translated, str):
                        by_id[key]["arguments"] = translated
                    elif "arguments" in call and isinstance(call["arguments"], str):
                        by_id[key]["arguments"] = call["arguments"]
                    # The classifier split code-mode `ns__name` to the
                    # bare name for ownership: re-attach a non-default
                    # namespace for the dispatch replay (the harness
                    # reads the item's own namespace field — a bare
                    # replay would land in the wrong namespace). The
                    # marker key never reaches the client (popped).
                    ns = call.pop("__route_namespace__", "")
                    if ns and ns.lower() != "default":
                        by_id[key]["namespace"] = ns
                    rewrite = call.pop("__exec_rewrite__", None)
                    if isinstance(rewrite, dict):
                        # Exec-channel rewrite: the nested call becomes
                        # ONE custom_tool_call named `exec` whose input
                        # is the JS invocation (the harness only
                        # executes nested tools through the `exec`
                        # orchestrator — a bare nested call fails
                        # lookup). Arguments move to input; the item id
                        # stays so the harness correlates the turn.
                        by_id[key]["name"] = str(rewrite.get("name", "exec"))
                        by_id[key]["type"] = "custom_tool_call"
                        by_id[key].pop("arguments", None)
                        by_id[key]["input"] = str(rewrite.get("input", ""))
                    elif call.get("type") == "custom_tool_call":
                        by_id[key]["type"] = "custom_tool_call"
                        if "arguments" in by_id[key]:
                            by_id[key]["input"] = by_id[key].pop("arguments")
            response = JSONResponse(status_code=200, content=payload)
            break
        calls = steer_calls
        if not calls:
            break
        names = sorted({str(c.get("name", "")) for c in calls})
        logger.info(
            "[%s] steering genuine tool call(s) %s back to client tools",
            trace_id,
            names,
        )
        followups: list = []
        for call in calls:
            call_id = str(call.get("call_id") or call.get("id") or "")
            args = call.get("arguments", "")
            args = args if isinstance(args, str) else ""
            followups.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": str(call.get("name", "")),
                    # Steer followups replay upstream: never leak a raw ""
                    # or non-JSON payload that the validator 400s.
                    "arguments": args if _valid_json(args) else "{}",
                }
            )
            followups.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _steer_output_for(
                        call,
                        args,
                        client_names,
                        owned,
                        defs,
                        client_tools,
                        genuine_names,
                    ),
                }
            )
        # Steered calls replay upstream as the client's own history so
        # the re-request sees the dead turn verbatim. Model output on
        # this path is Function-typed by construction (JSON body), so
        # the dead turn replays verbatim — no route re-typing needed
        # (contrast the streaming fold, which re-types Custom-marked
        # folded frames).
        outbound = dict(outbound, input=list(outbound.get("input", [])) + followups)
        response = await forward(
            client,
            url,
            headers,
            outbound,
            trace_id,
            synthesize_json=synthesize,
        )
    else:
        # Budget exhaustion without a clean turn: the last response still
        # carries calls the client cannot execute (live: model emitted a
        # fresh undeclared name every turn, so the loop died without a
        # repeat or a pass). Returning it would replay an unjudged dead
        # turn the client fails client-side (`unsupported call`) — fail
        # closed with the first steered call's redirect as the assistant
        # message instead, so the client surfaces the correction.
        tail_passed, tail_steer = _genuine_calls_in(
            response,
            client_names,
            client_tools=client_tools,
            genuine_names=genuine_names,
            family=family,
            trace_id=trace_id,
        )
        if not (tail_passed and not tail_steer) and tail_steer:
            tail_call = tail_steer[0]
            tail_args = tail_call.get("arguments", "")
            tail_args = tail_args if isinstance(tail_args, str) else ""
            logger.warning("[%s] steer budget exhausted; failing closed", trace_id)
            response = JSONResponse(
                status_code=200,
                content={
                    "id": f"resp_{trace_id}",
                    "object": "response",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": _steer_output_for(
                                        tail_call,
                                        tail_args,
                                        client_names,
                                        owned,
                                        defs,
                                        client_tools,
                                        genuine_names,
                                    ),
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                },
            )
    return response, outbound


def is_genuine_opencode(headers) -> bool:
    """True when the downstream client identifies as genuine opencode.

    Genuine opencode already carries the exact wire identity Zen's gate
    wants (canonical instructions + genuine tool set), so the anonymous
    shaping below (tool injection, chat sysprompt prefix) must not
    mangle it — passthrough instead. Only the opencode User-Agent
    counts: x-opencode-client/project/session headers are ones the
    proxy itself sets on the upstream leg, so honoring them here
    misfires whenever a client echoes proxied headers back (codex
    does exactly this — live: every codex turn arrived with
    x-opencode-client/project/session and lost its notice + tools).
    """
    ua = headers.get("user-agent", "")
    return ua.startswith("opencode/")


def _family_for(
    request: Request, ingress: str, req: RequestIR, requested: str = ""
) -> str:
    """Compat family for one downstream request (Task 1 detector).

    Downstream UA prefix first, else leg+tool-shape fallback; genuine
    opencode reports "unknown" (compat never renames its traffic —
    the table's `*` rows would otherwise rewrite its own calls).
    The requested (pre-alias) model name breaks the UA tie the alias
    mapping creates: `gpt-*` rides spark upstream but is the luna
    harness (deferred `functions` namespace), so a luna request behind
    a codex UA still detects luna when its tools carry the namespace.
    Failure degrades to "unknown" (today's behavior, `*` rows only).
    """
    try:
        from llms.proxy.compat import detect_family as _detect_family

        if is_genuine_opencode(request.headers):
            return "unknown"
        family = _detect_family(request.headers.get("user-agent"), ingress, req.tools)
        if family == "codex-plain" and (requested or "").lower().startswith("gpt-"):
            from llms.proxy.client_tools import _namespace_exec_desc

            if any(_namespace_exec_desc(t) for t in req.tools):
                return "luna"
        return family
    except Exception:
        return "unknown"


def _with_genuine_tools(outbound: dict) -> None:
    """Replace outbound tools with the genuine set, verbatim.

    Outbound carries ONLY the 12 genuine opencode definitions (byte-
    identical, in order): the free-tier gate fuzzy-matches the set, and
    client-declared extras must never ride the wire — any one of them
    can fail the upstream validator (seen live: gpt-5.6-luna codex
    session 400ing on `tools[12].description` length) and break
    inference before a single model turn runs. Client tools live in
    the system prompt instead (tool notice, built in translate from
    req.tools), which no validator checks. Response-side steering
    still classifies against req.tools, so unknown genuine-named
    calls steer back to the client's own declarations.
    Mutates outbound in place.
    """
    outbound["tools"] = [*GENUINE_TOOLS]


def _ensure_chat_system(outbound: dict) -> None:
    """Lead chat system content with the canonical prefix (anonymous only).

    The chat leg gates like responses (bisected live: canonical system
    streams 200, bare fails). Mutates outbound in place.
    """
    from llms.proxy.zen_prompts import SEAM

    messages = outbound.get("messages", [])
    original = ""
    rest = messages
    if (
        messages
        and isinstance(messages[0], dict)
        and messages[0].get("role") == "system"
    ):
        first, rest = messages[0], messages[1:]
        content = first.get("content", "")
        if isinstance(content, str):
            original = content
        elif isinstance(content, list):
            original = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    text = TITLE_PREFIX if not original else TITLE_PREFIX + SEAM + original
    outbound["messages"] = [{"role": "system", "content": text}, *rest]


def _note_free_tier_error(
    response: Response, settings: Settings, trace_id: str
) -> None:
    """Background a fingerprint re-check when Zen rejects the wire identity."""
    if not isinstance(response, JSONResponse):
        return
    try:
        payload = json.loads(response.body.decode())
    except Exception:
        return
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("error"), dict)
        and payload["error"].get("type") == "FreeTierError"
    ):
        note_free_tier_error(settings.data_dir, trace_id, settings.zen_base_url)


def _outcome_of(response: Response) -> tuple[str, float | None]:
    if isinstance(response, JSONResponse):
        try:
            payload = json.loads(response.body.decode())
        except Exception:
            payload = None
        return classify(response.status_code, payload, dict(response.headers))
    return "ok", None


def _reserved_session(request, ingress: str, model: str, body: dict) -> str | None:
    """Session id reserved by an earlier identical fresh attempt, if any."""
    table = getattr(request.app.state, "reservations", None)
    if table is None:
        return None
    return table.lookup(
        dedup.reserve_key(
            getattr(request.state, "secret_key", None),
            getattr(request.state, "affinity", None),
            ingress,
            model,
            body,
        )
    )


def _remember_reservation(
    request, ingress: str, model: str, body: dict, session_id: str
) -> None:
    table = getattr(request.app.state, "reservations", None)
    if table is None:
        return
    try:
        table.remember(
            dedup.reserve_key(
                getattr(request.state, "secret_key", None),
                getattr(request.state, "affinity", None),
                ingress,
                model,
                body,
            ),
            session_id,
        )
    except Exception as exc:
        logger.debug("session reservation failed: %r", exc)


def _dedup_background_finish(task, **kwargs) -> None:
    """Settle a deduped upstream task that outlived its foreground waiter.

    Stores JSON results for later claims; on timeout abandonment also runs
    the usage/conversation/warming bookkeeping the skipped tail would have
    done (fast completions leave those to the normal tail — exactly once
    either way). Best-effort: never raises.
    """
    table = kwargs["table"]
    key = kwargs["key"]
    entry = kwargs["entry"]
    try:
        if table.lookup(key) is not entry:
            try:
                task.result()
            except BaseException:
                pass
            return
        response = task.result()
    except BaseException:
        try:
            table.drop(key)
        except Exception:
            pass
        return
    try:
        from fastapi.responses import JSONResponse

        if not isinstance(response, JSONResponse):
            table.drop(key)
            return
        if not 200 <= response.status_code < 400:
            # Never hold errors: a replayed 429/5xx would mask recovery
            # (rebalance, failover, fresh retry-after). Retries rerun live.
            table.drop(key)
            return
        table.complete(
            key,
            response.status_code,
            bytes(response.body),
            dedup.DedupTable.store_headers(response.headers),
        )
        if not entry.timed_out:
            return
        request = kwargs["request"]
        try:
            _record_usage(request, kwargs["ingress"], kwargs["model"], response)
        except Exception as exc:
            logger.debug("dedup background usage failed: %r", exc)
        try:
            _record_conversation(
                response,
                kwargs["session_tracker"],
                kwargs["secret_key"],
                kwargs["session_id"],
                None,
            )
        except Exception as exc:
            logger.debug("dedup background conversation failed: %r", exc)
        try:
            _warm_new_conversation(
                response,
                kwargs["is_new_conversation"],
                kwargs["egress"],
                kwargs["client"],
                kwargs["url"],
                kwargs["headers"],
                kwargs["model"],
                kwargs["session_id"],
                kwargs["trace_id"],
            )
        except Exception as exc:
            logger.debug("dedup background warm failed: %r", exc)
    except Exception as exc:
        logger.debug("dedup background finish failed: %r", exc)


async def _publish_providers(request) -> None:
    """Best-effort immediate providers push (track-time/refetch changes).

    Release paths use throttled publish (high-frequency coalescing);
    track-time and health-change pushes bypass the throttle so the UI
    converges without polling.
    """
    try:
        hub = getattr(request.app.state, "admin_hub", None)
        publish = getattr(hub, "publish", None)
        if callable(publish):
            await publish("providers")
    except Exception:
        pass


def _record_conversation(
    response: Response,
    session_tracker,
    secret_key: str | None,
    session_id: str,
    stream_id: str | None = None,
) -> None:
    """Remember downstream response ids so chained continuations reuse this session."""
    if session_tracker is None or secret_key is None:
        return
    response_id = stream_id
    if response_id is None and isinstance(response, JSONResponse):
        if response.status_code >= 400:
            return
        try:
            payload = json.loads(response.body.decode())
        except Exception:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("id"), str):
            response_id = payload["id"]
    # Passthrough streams reuse upstream ids the tap never parses, so only
    # translate legs (explicit stream_id) and JSON bodies map here.
    if response_id is not None:
        session_tracker.remember(secret_key, "chain:" + response_id, session_id)


def _warm_new_conversation(
    response: Response,
    is_new_conversation: bool,
    egress: str,
    client,
    url: str,
    headers: dict,
    model: str,
    session_id: str,
    trace_id: str,
) -> None:
    """Background a title warming call for a brand-new responses conversation."""
    if not is_new_conversation or egress != "responses" or not sessions.SESSION_WARMING:
        return
    if not isinstance(response, StreamingResponse):
        if not isinstance(response, JSONResponse) or response.status_code >= 400:
            return
    elif getattr(response, "status_code", 200) >= 400:
        return

    async def _run() -> None:
        try:
            await sessions.warm_session(
                client, url, headers, model, session_id, trace_id
            )
        except BaseException as exc:
            logger.debug("[%s] session warm task ended: %r", trace_id, exc)

    asyncio.create_task(_run())


async def run(request: Request, settings: Settings, ingress: str) -> Response:
    trace_id = new_trace_id()
    body = await parse_body(request)
    if isinstance(body, JSONResponse):
        return body
    try:
        req: RequestIR = FROM[ingress](body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": {"message": str(exc)}})
    if not req.model:
        req = with_model(req, getattr(settings, DEFAULT_MODEL_ATTR[ingress]))
    requested = req.model
    req = with_model(req, resolve_alias(req.model, settings.model_aliases))
    egress = pick(req.model, ingress)
    genuine = is_genuine_opencode(request.headers)
    # Client tool notice (model-facing collision handling): only run()
    # knows genuine-vs-anonymous, so it computes the notice — but the
    # instructions-append itself lives in to_zen_responses (single
    # implementation, exercised by its unit tests). Genuine keeps exact
    # fidelity: no notice computed, kwarg stays "".
    notice = ""
    if egress == "responses" and not settings.zen_api_key and not genuine:
        # Plain-function legs get the upstream-`shell` directive (live
        # spark A/B, 2026-10-06: naming `shell`+`command` fills cleanly
        # with zero steers; naming the client `exec_command`+`cmd`
        # emits `{}` x3 — the client name has no wire schema since
        # outbound is genuine-12-only). Nested legs keep the tight
        # sketches: a bare "call `shell`" sentence would mis-teach
        # where tools run through the exec orchestrator channel.
        from llms.proxy.client_tools import has_nested_exec_channel

        notice = build_tool_notice(
            req.tools,
            GENUINE_TOOL_NAMES,
            shell_alias=not has_nested_exec_channel(req.tools),
        )
    outbound = (
        to_zen_responses(
            req,
            tool_notice=notice,
            family=_family_for(request, ingress, req, requested),
        )
        if egress == "responses"
        else TO[egress](req)
    )
    affinity = getattr(request.state, "affinity", None)
    secret_key = getattr(request.state, "secret_key", None)
    # Upstream session: per-conversation simulation on the responses leg
    # (previous_response_id chains, codex thread headers), stable per-key
    # fallback everywhere else. Zen's free-tier gate requires body
    # prompt_cache_key to equal the x-opencode-session header (both
    # well-formed ses_ IDs); mint one id here and share it with headers.
    # Chat has no such field/gate; messages needs a real API key instead.
    session_id = stable_session_id(settings.zen_api_key)
    is_new_conversation = False
    session_tracked = False
    session_tracker = getattr(request.app.state, "sessions", None)
    if session_tracker is not None and ingress == "responses":
        ref = sessions.conversation_ref(ingress, body, request.headers)
        if ref is None:
            # No trackable ref: reuse a session reserved by an earlier
            # identical attempt (client retry after a dedup 429) so the
            # retry matches the in-flight hash and its cache affinity.
            reserved = _reserved_session(request, ingress, req.model, body)
            if reserved is not None:
                session_id = reserved
            else:
                session_id = sessions.mint_session_id()
                _remember_reservation(request, ingress, req.model, body, session_id)
                is_new_conversation = True
        else:
            hit = session_tracker.lookup(secret_key, ref)
            if hit is not None:
                session_id = hit
                session_tracked = True
            elif not ref.startswith("chain:"):
                session_id = sessions.mint_session_id()
                session_tracker.remember(secret_key, ref, session_id)
                is_new_conversation = True
    genuine = is_genuine_opencode(request.headers)
    if egress == "responses":
        outbound["prompt_cache_key"] = session_id
        # Anonymous free tier matches the genuine tool set (bisected
        # live: full set passes; bare/renamed sets 403). Outbound is
        # genuine-12 ONLY: client extras never ride the wire (any one
        # can fail the upstream validator and 400 the whole request —
        # seen live as `tools[12].description` length on a codex
        # session). Genuine keeps exact fidelity (passthrough, no
        # notice); keyed operators likewise. The client tool notice
        # (model-facing collision handling, appended after client
        # instructions by to_zen_responses) is computed above from the
        # full set — req.tools already includes deferred
        # `additional_tools` names (dissolved in from_responses) — and
        # the steer paths below still classify against req.tools.
        if not settings.zen_api_key and not genuine:
            _with_genuine_tools(outbound)
    elif egress == "chat" and not settings.zen_api_key and not genuine:
        _ensure_chat_system(outbound)
    bucket = bucket_for(
        affinity,
        req.model,
        settings.num_buckets,
        secret_key,
        # Tracked continuations only: a tracker hit means a real ongoing
        # conversation, so its turns pin to one bucket (cache stays warm)
        # while distinct conversations spread across slots. Fresh mints
        # stay on the stable hash — no identity exists yet to pin, and
        # this keeps untracked traffic (and its tests) deterministic.
        session_id if session_tracked else None,
    )
    table = request.app.state.bucket_table
    slot = table.slot_for(bucket)
    log_ingress(
        trace_id,
        request.url.path,
        {
            "model": req.model,
            "requested_model": requested,
            "ingress": ingress,
            "egress": egress,
            "affinity": affinity,
            "bucket": bucket,
            "slot": slot,
        },
    )
    # Compat family detection (Task 1): record the downstream UA verbatim
    # so per-family probes reveal what each harness actually sends. The
    # relay captures body only, so the proxy log is the UA source. INFO
    # (not debug): the proxy runs at INFO in probe and prod alike.
    logger.info(
        "[%s] ingress UA=%r ingress=%s",
        trace_id,
        request.headers.get("user-agent", ""),
        ingress,
    )
    headers = build_zen_headers(settings, session_id=session_id)
    url = settings.zen_base_url.rstrip("/") + ENDPOINT_PATH[egress]
    log_upstream(trace_id, url, headers, outbound)
    egress_provider = request.app.state.egress
    started = time.monotonic()
    provider_id: str | None = None
    kind: str = "noproxy"
    warp_egress = None
    via_warp: dict | None = None
    warp_idx: int | None = None
    registry = getattr(request.app.state, "providers", None)
    resolve = getattr(egress_provider, "resolve", None)
    if callable(resolve):
        provider_id, kind, warp_egress = resolve(req.model, bucket)
        if kind == "queued":
            # Main-queue hold: warp capacity exists for this model but
            # every candidate is an at-quota probe. Hold the connection
            # open and re-resolve until a probe releases (falls through
            # to the warp block below) or the budget lapses (429).
            # Keepalive is the open connection itself: wakeups every
            # queue_keepalive_s bound the silent gap. Waiters are never
            # in-flight tracked (track sits below), so no release is
            # owed on any exit from this loop. A disconnected client
            # aborts the hold at once (nothing tracked, just return) —
            # otherwise long holds forward upstream to a ghost behind
            # idle timeouts (Cloudflare 120s per spec §5).
            _waited = 0.0
            # No `or` defaults here: an explicit 0.0 budget must degrade
            # immediately (0.0 or 600.0 would read as 600.0 and hang the
            # waiter for the full budget). Floors only stop busy-spin
            # (step) and negative budgets.
            _step = max(0.01, float(getattr(settings, "queue_keepalive_s", 15.0)))
            _budget = max(0.0, float(getattr(settings, "queue_wait_s", 600.0)))
            while _waited < _budget and kind == "queued":
                _sleep = min(_step, _budget - _waited)
                await asyncio.sleep(_sleep)
                _waited += _sleep
                try:
                    if await request.is_disconnected():
                        return JSONResponse(
                            status_code=499,
                            content={"error": {"message": "client disconnected"}},
                        )
                except Exception:
                    pass
                provider_id, kind, warp_egress = resolve(req.model, bucket)
            if kind == "queued":
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "message": "queue wait exceeded",
                            "type": "queue_timeout",
                        }
                    },
                    headers={"retry-after": str(int(_budget))},
                )
        if kind == "warp" and warp_egress is not None:
            provider = None
            if registry is not None:
                for p in registry.load():
                    if p.id == provider_id:
                        provider = p
                        break
            if registry is not None and provider is not None:
                _before = registry.runtime(provider.id).health.fetched_at
                await registry.refresh_health(provider)
                if registry.runtime(provider.id).health.fetched_at != _before:
                    # The refresh actually moved health (not a TTL no-op):
                    # push so admin viewers converge without polling or a
                    # debug click (stale red-until-debug regression).
                    await _publish_providers(request)
                # Re-resolve after the refresh: a slot that flipped ready
                # during this very poll must seed the egress's SOCKS ports
                # before client_for runs, or the request fails open despite
                # a connected tunnel (live finding: final status.json showed
                # ready=true while the request went direct).
                provider_id, kind, warp_egress = resolve(req.model, bucket)
                if kind != "warp" or warp_egress is None:
                    provider_id, via_warp = None, None
                sync = getattr(egress_provider, "sync_bucket_slots", None)
                if callable(sync):
                    sync(table)
            if kind != "warp" or warp_egress is None:
                client = egress_provider.client_for(bucket, slot)
            else:
                try:
                    client = warp_egress.client_for(bucket, slot)
                except RuntimeError:
                    # No ready SOCKS exits: fail open to direct.
                    logger.info(
                        "[%s] warp provider %s has no ready exits; failing open",
                        trace_id,
                        provider_id,
                    )
                    provider_id, via_warp = None, None
                    client = egress_provider.client_for(bucket, slot)
                else:
                    # Slot-spread egress: the request's slot pins to one
                    # ready exit (slot % ready), so warp_idx is the slot's
                    # position in the ready spread — not a pool pin.
                    ports = warp_egress.ready_ports()
                    warp_idx = (
                        ports.index(warp_egress.pick_port(slot)) if ports else None
                    )
                    via_warp = {
                        "provider_id": provider_id or "",
                        "warp_idx": warp_idx,
                        "socks_port": warp_egress.pick_port(slot),
                    }
        else:
            client = egress_provider.client_for(bucket, slot)
    else:
        client = egress_provider.client_for(bucket, slot)
    # In-flight accounting starts once resolve() assigns a provider, and
    # the count change pushes immediately: the per-request refresh below
    # is TTL-gated (usually a no-op), so without this the Busy tick would
    # only surface on the release publish after the response — or never,
    # if release coalesces into the cooldown window.
    # Cooldown-expiry observation: a leftover retry_until whose window
    # lapsed enters ready-probation here (concurrency 1) instead of
    # slipping straight to ready (derive stays pure, so this gate is the
    # only live observer of expiry).
    if registry is not None and provider_id:
        _arm_probation_on_expiry(registry, provider_id)
    _inflight_track(request, provider_id)
    # Post-track quota re-check: two requests can both resolve to the
    # same probation provider between resolve() and track (two awaits
    # sit between them: refresh_health, _publish_providers). The second
    # arrival must not join the armed probe flight at concurrency 2 —
    # release its slot and re-enter the queued hold instead.
    if registry is not None and provider_id:
        try:
            _rt = registry.runtime(provider_id)
            if bool(getattr(_rt, "probation", False)) and _rt.in_flight > 1:
                _inflight_release(request, provider_id)
                provider_id, kind, warp_egress = None, "queued", None
        except Exception:
            pass
    if kind == "queued" and provider_id is None:
        # Bounced by the post-track quota re-check: re-enter the hold
        # loop above is behind us, so hold inline with the same budget
        # semantics (disconnect abort, degrade with the actual budget).
        _waited = 0.0
        _step = max(0.01, float(getattr(settings, "queue_keepalive_s", 15.0)))
        _budget = max(0.0, float(getattr(settings, "queue_wait_s", 600.0)))
        _resolve = getattr(egress_provider, "resolve", None)
        while _waited < _budget and kind == "queued":
            _sleep = min(_step, _budget - _waited)
            await asyncio.sleep(_sleep)
            _waited += _sleep
            try:
                if await request.is_disconnected():
                    return JSONResponse(
                        status_code=499,
                        content={"error": {"message": "client disconnected"}},
                    )
            except Exception:
                pass
            if callable(_resolve):
                provider_id, kind, warp_egress = _resolve(req.model, bucket)
            if kind == "warp" and warp_egress is not None:
                break
        if kind == "queued":
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "queue wait exceeded",
                        "type": "queue_timeout",
                    }
                },
                headers={"retry-after": str(int(_budget))},
            )
        if kind != "warp" or warp_egress is None:
            client = egress_provider.client_for(bucket, slot)
        else:
            try:
                client = warp_egress.client_for(bucket, slot)
            except RuntimeError:
                logger.info(
                    "[%s] warp provider %s has no ready exits; failing open",
                    trace_id,
                    provider_id,
                )
                provider_id, via_warp = None, None
                client = egress_provider.client_for(bucket, slot)
            else:
                ports = warp_egress.ready_ports()
                warp_idx = ports.index(warp_egress.pick_port(slot)) if ports else None
                via_warp = {
                    "provider_id": provider_id or "",
                    "warp_idx": warp_idx,
                    "socks_port": warp_egress.pick_port(slot),
                }
        if registry is not None and provider_id:
            _arm_probation_on_expiry(registry, provider_id)
        _inflight_track(request, provider_id)
    await _publish_providers(request)
    if registry is not None:
        # Pool-dry shed: direct also ratelimited and a warp bounce in
        # flight. Otherwise fall through to normal routing — resolve()
        # already skipped cycling warps, so the next request fails over
        # (fail open to direct as usual).
        shed, shed_provider = _shed_if_pool_dry(registry, req.model, trace_id)
        if shed is not None:
            _inflight_release(request, provider_id)
            _record_usage(request, ingress, req.model, shed)
            await registry.runtime(shed_provider or "").record(
                RecentRequest(
                    ts=time.time(),
                    model=req.model,
                    status=429,
                    ms=(time.monotonic() - started) * 1000.0,
                    warp_idx=None,
                    error="pool dry: warp cycling, direct limited",
                )
            )
            return shed
    tracker = getattr(request.app.state, "usage", None)
    usage_secret = getattr(request.state, "secret_key", None)
    stream_usage_cb = None
    if (
        outbound.get("stream") is True
        and tracker is not None
        and usage_secret is not None
    ):

        def stream_usage_cb(
            done, _tracker=tracker, _key=usage_secret, _model=req.model
        ):
            from llms.proxy.ir import StreamDone as _StreamDone

            if isinstance(done, _StreamDone):
                _tracker.record(
                    _key,
                    _model,
                    done.input_tokens,
                    done.output_tokens,
                    done.cached_tokens,
                    done.reasoning_tokens,
                    count_request=False,
                )
                # Stow the parser outcome for stream settle: a fully
                # consumed StreamingResponse always carries HTTP 200, so
                # settle cannot tell truncation/failure from success by
                # status alone. The sink runs at stream end on the same
                # request, before the wrap's exhaustion callback.
                try:
                    request.state.stream_outcome = done.status
                except Exception:
                    pass
                # Streaming passthrough legs hand the client the upstream
                # response id, which comes back as previous_response_id:
                # chain it to this session now (stream end) so the
                # follow-up turn reuses the session and the prompt cache
                # stays warm. (JSON bodies record at response time in
                # _record_conversation; translate legs mint downstream ids
                # recorded synchronously there.)
                if done.response_id and session_tracker is not None:
                    try:
                        session_tracker.remember(
                            _key, "chain:" + done.response_id, session_id
                        )
                    except Exception as exc:
                        logger.debug(
                            "[%s] session chain remember failed: %r",
                            trace_id,
                            exc,
                        )

    synthesize = (
        _synthesize_for(ingress, egress, req.model)
        if outbound.get("stream") is not True and not settings.zen_api_key
        else None
    )
    synthesize = (
        _synthesize_for(ingress, egress, req.model)
        if outbound.get("stream") is not True and not settings.zen_api_key
        else None
    )

    async def _do_forward():
        # Streaming steer runs only for non-genuine responses-leg
        # clients that declared tools: the fold needs client_names to
        # detect undeclared calls. Genuine opencode passes through
        # untouched (no shaping, no steer); other legs keep legacy
        # streaming.
        # Fold-vs-live observability (2026-10-07): the streaming fold
        # only runs when every one of these holds; a bare-patch turn
        # that never folds reproduces as "arm idle" with no other
        # trace. Log the inputs once per request so a dead arm leaves
        # a verdict (which gate failed), not a mystery. logger.info
        # (not print): uvicorn captures stdout per worker, but the
        # logging pipeline writes to the same stream as every other
        # proxy line — grep one place, not two.
        _client_names = {t.name for t in req.tools if t.name}
        _family = _family_for(request, ingress, req, requested)
        logger.info(
            "[%s] steer gates genuine=%s egress=%s stream=%s names=%s family=%s",
            trace_id,
            genuine,
            egress,
            outbound.get("stream"),
            sorted(_client_names),
            _family,
        )
        # Codex exec is NON-streaming downstream (live 2026-10-07: every
        # codex turn arrives stream=None): the streaming fold never runs
        # for it — the synthesize path (_steer_genuine_calls below) is
        # the arm's live leg. A stream=True gate here would silently
        # disable the arm for the very client it was built for.
        _steer_stream = (
            not genuine
            and egress == "responses"
            and outbound.get("stream") is True
            and bool(_client_names)
        )
        # Compat family (Task 1 detector): downstream UA prefix first,
        # else leg+tool-shape fallback. Unknown degrades to today's
        # behavior (`*` all-family rows only — never a wrong-family
        # rewrite). Genuine opencode keeps exact fidelity regardless.
        # (Computed once above for the steer-gates observability line.)
        _response = await forward(
            client,
            url,
            headers,
            outbound,
            trace_id,
            convert=_convert_for(ingress, egress, req.model),
            translate_dialects=_stream_for(ingress, egress, req.model),
            synthesize_json=synthesize,
            via_warp=via_warp,
            # Usage is sniffed from upstream (egress-dialect) bytes; identical
            # for passthrough (ingress == egress) and required for translate.
            stream_ingress=egress if outbound.get("stream") is True else None,
            stream_usage_sink=stream_usage_cb,
            client_names=_client_names or None,
            steer_streaming=_steer_stream,
            client_tool_defs=req.tools,
            genuine_names=() if genuine else GENUINE_TOOL_NAMES,
            family=_family,
        )
        if synthesize is not None and egress == "responses":
            # Steer any call the client cannot execute (undeclared, or
            # owned-but-invalid args) back: with client tools list them,
            # otherwise demand a direct answer. Genuine opencode keeps
            # exact fidelity — no convert, no re-serialize (see run()).
            _response, _ = await _steer_genuine_calls(
                _response,
                client=client,
                url=url,
                headers=headers,
                outbound=outbound,
                synthesize=synthesize,
                trace_id=trace_id,
                client_names=_client_names,
                client_tools=() if genuine else req.tools,
                genuine_names=() if genuine else GENUINE_TOOL_NAMES,
                family=_family,
            )
        return _response

    # Slow-request dedup (non-streaming only): a request outrunning
    # MAX_TIMEOUT gets 429 + Retry-After while its upstream work
    # continues in the background; a retry with a matching hash claims
    # the held response (evicting it) or 429s again while running.
    dedup_table = getattr(request.app.state, "dedup", None)
    dedup_key = None
    # Any exception escaping the forward/dedup await region below
    # (CancelledError on client disconnect, upstream errors) must release
    # the track() slot: the tail release is then unreachable. The handler
    # covers only this region and re-raises — a blanket post-track
    # try/finally would release the streaming path immediately and defeat
    # the deferred _wrap_inflight release.
    try:
        if dedup_table is not None and not req.stream:
            dedup_key = dedup.request_hash(
                secret_key, affinity, ingress, req.model, body, session_id
            )
            _entry = dedup_table.lookup(dedup_key)
            if _entry is not None:
                if _entry.done:
                    response = dedup.replay_response(
                        _entry.status, _entry.body, _entry.headers
                    )
                    dedup_table.drop(dedup_key)
                else:
                    response = dedup.inflight_response()
                # No upstream leg ran for this waiter: release the track()
                # slot before returning (same rationale as the shed and
                # synthetic-dedup returns).
                _inflight_release(request, provider_id)
                elapsed_ms = (time.monotonic() - started) * 1000.0
                return response
            _timeout = float(settings.max_timeout_s)
            if _timeout > 0:
                _task = asyncio.create_task(_do_forward())
                _entry = dedup_table.track(dedup_key, _task)
                _task.add_done_callback(
                    lambda t: _dedup_background_finish(
                        t,
                        table=dedup_table,
                        key=dedup_key,
                        entry=_entry,
                        request=request,
                        ingress=ingress,
                        model=req.model,
                        session_tracker=session_tracker,
                        secret_key=secret_key,
                        session_id=session_id,
                        is_new_conversation=is_new_conversation,
                        egress=egress,
                        client=client,
                        url=url,
                        headers=headers,
                        trace_id=trace_id,
                    )
                )
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(_task), timeout=_timeout
                    )
                except TimeoutError:
                    _entry.timed_out = True
                    # The upstream leg keeps running detached (shielded);
                    # the background finish never owns the slot (it would
                    # pin Busy for the full upstream duration), so the
                    # foreground releases here.
                    _inflight_release(request, provider_id)
                    response = dedup.inflight_response()
                    elapsed_ms = (time.monotonic() - started) * 1000.0
                    return response
            else:
                response = await _do_forward()
        else:
            response = await _do_forward()
    except BaseException:
        _inflight_release(request, provider_id)
        raise
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if (
        isinstance(response, JSONResponse)
        and response.headers.get(dedup.DEDUP_HEADER) is not None
    ):
        # Synthetic dedup response (inflight 429): skip every tail
        # side-effect so it can never read as provider congestion — but
        # release the track() slot first (no upstream leg ran for it).
        _inflight_release(request, provider_id)
        return response
    _note_free_tier_error(response, settings, trace_id)
    stream_id = None
    if (
        isinstance(response, StreamingResponse)
        and _stream_for(ingress, egress, req.model) is not None
    ):
        stream_id = {
            "responses": new_resp_id,
            "chat": new_chat_id,
            "messages": new_msg_id,
        }[ingress](trace_id)
    _record_conversation(response, session_tracker, secret_key, session_id, stream_id)
    _warm_new_conversation(
        response,
        is_new_conversation,
        egress,
        client,
        url,
        headers,
        req.model,
        session_id,
        trace_id,
    )
    outcome, retry_after = _outcome_of(response)
    if outcome == "ratelimited":
        new_slot = table.note_ratelimited(bucket, retry_after)
        logger.info(
            "[%s] bucket %s ratelimited, moved slot %s -> %s",
            trace_id,
            bucket,
            slot,
            new_slot,
        )
        registry = getattr(request.app.state, "providers", None)
        if registry is not None:
            reason = ""
            if isinstance(response, JSONResponse):
                try:
                    reason = (
                        json.loads(response.body.decode())
                        .get("error", {})
                        .get("message", "")
                    )
                except Exception:
                    reason = ""
            if provider_id:
                registry.runtime(provider_id).note_ratelimited(retry_after, reason)
                _maybe_auto_cycle(request, provider_id, via_warp, trace_id)
            else:
                # Direct path (fail-open or noproxy-routed): record on the
                # noproxy runtime so the pool-dry gate can see direct's
                # ratelimit. Without this the gate would never fire.
                direct_id = next(
                    (
                        p.id
                        for p in _providers_serving(registry, req.model)
                        if p.kind == "noproxy"
                    ),
                    "noproxy",
                )
                registry.runtime(direct_id).note_ratelimited(retry_after, reason)
            # The 429 backoff feeds the providers SSE status (yellow dot):
            # push so admin viewers converge without polling.
            await get_hub(request).publish("providers")
            # Fast-failover hint: pool min-retry over providers serving this
            # model ("when the next request is ok" per the 429 memo). ~1s
            # floor so the client retries fast onto the failover provider.
            if isinstance(response, JSONResponse):
                providers = _providers_serving(registry, req.model)
                wait = math.ceil(_min_retry_in(registry, providers))
                hint = max(1, min(60, int(wait)))
                response.headers["retry-after"] = str(hint)
                logger.info(
                    "[%s] provider %s ratelimited, retry-after hint %s",
                    trace_id,
                    provider_id or "direct",
                    hint,
                )
    _record_usage(request, ingress, req.model, response)
    if isinstance(response, StreamingResponse):
        return _wrap_inflight(
            request,
            response,
            provider_id,
            model=req.model,
            warp_idx=warp_idx,
            started=started,
        )
    _inflight_release(request, provider_id)
    if registry is not None and provider_id:
        _settle_probation(request, registry, provider_id, response)
        status = response.status_code if hasattr(response, "status_code") else 0
        error = ""
        if isinstance(response, JSONResponse) and status >= 400:
            try:
                error = (
                    json.loads(response.body.decode())
                    .get("error", {})
                    .get("message", "")
                )
            except Exception:
                error = ""
        await registry.runtime(provider_id).record(
            RecentRequest(
                ts=time.time(),
                model=req.model,
                status=status,
                ms=elapsed_ms,
                warp_idx=warp_idx,
                error=str(error)[:200],
            )
        )
    return response


def _arm_probation_on_expiry(registry, provider_id: str | None) -> bool:
    """Enter ready-probation when a recorded cooldown just expired.

    The only live observer of retry expiry (derive_lifecycle stays pure,
    so direct `retry_until = 0.0` test manipulation keeps working): a
    leftover non-zero `retry_until` whose `retry_in()` reached 0 means the
    cooldown lapsed without a request noticing — zero it and arm
    probation so the next flight probes at concurrency 1 instead of
    slipping straight to ready. Active backoffs (retry_in > 0) and
    never-limited providers (retry_until == 0) are untouched, preserving
    the Task 4 discipline that only 429/reconnect/expiry mutates
    retry_until. Never raises; returns True when it armed.
    """
    try:
        if registry is None or not provider_id:
            return False
        rt = registry.runtime(provider_id)
        if float(getattr(rt, "retry_until", 0.0) or 0.0) == 0.0:
            return False
        if rt.retry_in() > 0:
            return False
        if bool(getattr(rt, "probation", False)):
            return False
        rt.retry_until = 0.0
        rt.retry_reason = ""
        rt.probation = True
        return True
    except Exception:
        return False


def _settle_probation(request, registry, provider_id: str, response) -> None:
    """Promote or demote a probation provider on its probe outcome.

    Success (2xx) promotes probation -> ready (unbounded). Probe 429
    returns to ratelimited: note_ratelimited already applied the 60s
    default / retry-after value in the outcome block above, so only the
    flag clears here. Non-429 errors leave probation armed: the single
    flight stays the test, and the next request re-probes.

    Streams: a fully-consumed StreamingResponse always carries HTTP 200,
    so settle consults the parser outcome stowed on request.state by
    stream_usage_cb (failed/incomplete stays armed; only completed
    promotes). Absent outcome (untracked stream legs) falls back to
    status, preserving the old behavior.
    """
    try:
        rt = registry.runtime(provider_id)
        if not bool(getattr(rt, "probation", False)):
            return
        status = response.status_code if hasattr(response, "status_code") else 0
        outcome = None
        try:
            outcome = getattr(getattr(request, "state", None), "stream_outcome", None)
        except Exception:
            outcome = None
        if isinstance(response, StreamingResponse) and outcome is not None:
            if outcome == "completed" and 200 <= status < 300:
                rt.probation = False
            return
        if 200 <= status < 300 or status == 429:
            rt.probation = False
    except Exception:
        pass


def _providers_serving(registry, model: str) -> list:
    """Enabled providers serving a model (best-effort; [] when unknown)."""
    try:
        return [p for p in registry.load() if p.enabled and p.serves(model)]
    except Exception:
        return []


def _shed_if_pool_dry(
    registry, model: str, trace_id: str
) -> tuple[JSONResponse | None, str | None]:
    """Escalating 429 only when the provider pool is dry — else (None, None).

    Dry = direct (noproxy) also ratelimited AND ≥1 warp provider serving
    the model is mid-restart, with no serving provider currently usable
    (nothing available, but a bounce in flight means capacity is coming
    back soon). Anything else falls through to normal routing: resolve()
    already skipped cycling warps, so the next request fails over (fail
    open to direct as usual).

    Returns the shed response plus the cycling provider id (for the
    recent-request ring) when shedding.
    """
    providers = _providers_serving(registry, model)
    warps = [p for p in providers if p.kind == "warp"]
    cycling = [p for p in warps if registry.runtime(p.id).cycling]
    if not cycling:
        return None, None
    for p in providers:
        rt = registry.runtime(p.id)
        if p.kind == "noproxy":
            if rt.retry_in() <= 0:
                return None, None
        elif not rt.cycling and any(w.ready for w in rt.health.exits):
            return None, None
    rt = registry.runtime(cycling[0].id)
    rt.cycle_hits += 1
    retry_after = min(60, 5 + (rt.cycle_hits - 1))
    logger.info(
        "[%s] pool dry (warp %s cycling, direct limited), shedding 429 "
        "(hit %s, retry-after %s)",
        trace_id,
        cycling[0].id,
        rt.cycle_hits,
        retry_after,
    )
    return (
        JSONResponse(
            status_code=429,
            content={"error": {"message": f"warp provider {cycling[0].id} cycling"}},
            headers={"retry-after": str(retry_after)},
        ),
        cycling[0].id,
    )


def _min_retry_in(registry, providers: list) -> float:
    """Seconds until the next serving provider is expected usable (≥0)."""
    waits: list[float] = []
    for p in providers:
        rt = registry.runtime(p.id)
        if p.kind == "noproxy":
            waits.append(rt.retry_in())
            continue
        ready = any(w.ready for w in rt.health.exits)
        if ready or not rt.cycling:
            waits.append(0.0)
        else:
            # Cycling with no ready exit: the bounce (or its 45s bring-up)
            # is the soonest this provider recovers; cap the estimate so a
            # wedged bounce doesn't pin the hint.
            waits.append(45.0)
    return min(waits) if waits else 0.0


def _auto_cycle_cooldown_s(request: Request) -> float:
    settings = getattr(request.app.state, "settings", None)
    return float(getattr(settings, "warp_auto_cycle_cooldown_s", 300) or 300)


def _maybe_auto_cycle(
    request: Request, provider_id: str, via_warp: dict | None, trace_id: str
) -> None:
    """Bounce a ratelimited warp exit in the background (best-effort).

    Only fires for requests that actually rode warp (via_warp carries the
    exit's SOCKS port); guarded by per-provider cooldown + in-flight dedup.
    Never raises — the request path must not fail because the bounce did.
    """
    import asyncio
    import time

    try:
        if not via_warp or via_warp.get("socks_port") is None:
            return
        registry = getattr(request.app.state, "providers", None)
        if registry is None:
            return
        provider = next((p for p in registry.load() if p.id == provider_id), None)
        if provider is None or provider.kind != "warp":
            return
        rt = registry.runtime(provider_id)
        now = time.monotonic()
        if rt.cycling:
            logger.info(
                "[%s] warp provider %s already cycling, skipping auto-cycle",
                trace_id,
                provider_id,
            )
            return
        cooldown = _auto_cycle_cooldown_s(request)
        if now - rt.last_auto_cycle < cooldown:
            logger.info(
                "[%s] warp provider %s auto-cycle on cooldown (%.0fs left)",
                trace_id,
                provider_id,
                cooldown - (now - rt.last_auto_cycle),
            )
            return
        pool = registry.pool_for(provider)
        if pool is None:
            return
        port = via_warp["socks_port"]
        inst = next((w for w in pool.instances if w.socks_port == port), None)
        if inst is None:
            logger.warning(
                "[%s] warp provider %s auto-cycle: no slot on port %s",
                trace_id,
                provider_id,
                port,
            )
            return
        rt.last_auto_cycle = now
        rt.cycling = True
        rt.cycle_hits = 0
        bounce_epoch = rt.retry_epoch
        bounce_gen = rt.gen
        logger.info(
            "[%s] warp provider %s exit %s (port %s) ratelimited, cycling",
            trace_id,
            provider_id,
            inst.idx,
            port,
        )

        async def _bounce_and_clear() -> None:
            bounced_ok = False
            try:
                result = await pool.bounce_exit(inst.idx)
                bounced_ok = isinstance(result, dict) and bool(result.get("ok", False))
                logger.info(
                    "[%s] warp provider %s exit %s bounce done: %s",
                    trace_id,
                    provider_id,
                    inst.idx,
                    result,
                )
            except Exception as exc:
                logger.warning(
                    "[%s] warp provider %s exit %s bounce failed: %r",
                    trace_id,
                    provider_id,
                    inst.idx,
                    exc,
                )
            rt.cycling = False
            rt.cycle_hits = 0
            if rt.gen != bounce_gen:
                # Superseded mid-bounce (disable/delete): the pool is
                # gone on purpose. refresh_health would ensure_pool a
                # brand-new pool for a dead provider and resurrect
                # daemons for it — skip the refresh, just converge SSE.
                # (Unconditional clears, not finally: a return in
                # finally would swallow bounce exceptions.)
                logger.info(
                    "[%s] warp provider %s auto-cycle superseded "
                    "(gen %s -> %s), skipping post-cycle refresh",
                    trace_id,
                    provider_id,
                    bounce_gen,
                    rt.gen,
                )
                try:
                    await get_hub(request).publish("providers")
                except Exception:
                    pass
                return
            try:
                await registry.refresh_health(provider, force=True)
            except Exception as exc:
                logger.warning(
                    "[%s] warp provider %s post-cycle refresh failed: %r",
                    trace_id,
                    provider_id,
                    exc,
                )
            if bounced_ok and rt.retry_epoch == bounce_epoch:
                # Fresh circuits: drop the 429 backoff so the provider
                # rejoins immediately instead of sitting out max(60s,
                # retry-after). Guarded by epoch: a sibling 429 that
                # landed mid-bounce bumped the epoch and must survive.
                # A still-bad exit keeps its backoff and the cooldown
                # gates the next bounce.
                rt.retry_until = 0.0
                rt.retry_reason = ""
                # Bounce-ok is a becoming-ready path: arm probation so
                # the next request probes at concurrency 1 (promotion
                # happens on first success).
                rt.probation = True
                logger.info(
                    "[%s] warp provider %s exit %s backoff cleared",
                    trace_id,
                    provider_id,
                    inst.idx,
                )
                # Backoff cleared: push the fresh (green) status to SSE.
                await get_hub(request).publish("providers")

        rt.cycle_task = asyncio.create_task(_bounce_and_clear())
    except Exception as exc:
        logger.warning(
            "[%s] warp provider %s auto-cycle trigger failed: %r",
            trace_id,
            provider_id,
            exc,
        )


def _record_usage(
    request: Request, ingress: str, model: str, response: Response
) -> None:
    from llms.proxy.ir import StreamDone
    from llms.proxy.usage import extract_usage

    tracker = getattr(request.app.state, "usage", None)
    secret_key = getattr(request.state, "secret_key", None)
    if tracker is None or secret_key is None:
        return
    if isinstance(response, StreamingResponse):
        # Translate path: usage arrives via stream_usage_sink when the
        # heartbeat generator finishes (see forward.translate_streaming).
        done = getattr(response, "stream_usage", None)
        if isinstance(done, StreamDone):
            tracker.record(
                secret_key,
                model,
                done.input_tokens,
                done.output_tokens,
                done.cached_tokens,
                done.reasoning_tokens,
            )
            return
        # Passthrough: the tap in forward.py records tokens synchronously
        # when the stream exhausts (request counted here, tokens merged on
        # completion without double-counting). Nothing more to do at
        # response-build time; fall through to the request-only record.
        tracker.record(secret_key, model, None, None, None, None)
        return
    if not isinstance(response, JSONResponse) or response.status_code >= 400:
        return
    try:
        payload = json.loads(response.body.decode())
    except Exception:
        payload = None
    if not isinstance(payload, dict):
        return
    in_tokens, out_tokens, cached_tokens, reasoning_tokens = extract_usage(
        ingress, payload
    )
    tracker.record(
        secret_key, model, in_tokens, out_tokens, cached_tokens, reasoning_tokens
    )
