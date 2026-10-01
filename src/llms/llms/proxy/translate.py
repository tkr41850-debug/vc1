from __future__ import annotations

import logging
from dataclasses import replace

from llms.proxy.ir import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    ImageBlock,
    LlmMessage,
    LlmParams,
    OpaqueBlock,
    RequestIR,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolDef,
    ToolResultBlock,
)

logger = logging.getLogger("zen_proxy")

_CHAT_ROLES = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL)
_RESPONSES_ROLES = ("system", "developer", "user", "assistant")

EFFORT_BUDGETS = {"low": 1024, "medium": 4096, "high": 16384}


def effort_for_budget(budget: int) -> str:
    if budget <= 1024:
        return "low"
    if budget <= 8192:
        return "medium"
    return "high"


def _structured_from_chat(format: object) -> dict | None:
    if not isinstance(format, dict):
        return None
    if format.get("type") == "json_schema" and isinstance(
        format.get("json_schema"), dict
    ):
        schema = format["json_schema"]
        return {
            "name": schema.get("name"),
            "schema": schema.get("schema"),
            "strict": bool(schema.get("strict", False)),
        }
    if format.get("type") == "json_object":
        return {"name": None, "schema": None, "strict": False}
    return None


def _structured_from_responses(format: object) -> dict | None:
    if not isinstance(format, dict):
        return None
    if format.get("type") == "json_schema":
        return {
            "name": format.get("name"),
            "schema": format.get("schema"),
            "strict": bool(format.get("strict", False)),
        }
    if format.get("type") == "json_object":
        return {"name": None, "schema": None, "strict": False}
    return None


def _params_from_chat(body: dict) -> LlmParams:
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    effort = body.get("reasoning_effort")
    thinking = body.get("thinking")
    if (
        effort is None
        and isinstance(thinking, dict)
        and thinking.get("type") == "enabled"
    ):
        effort = "medium"
    parallel = body.get("parallel_tool_calls")
    return LlmParams(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=max_tokens,
        stop=body.get("stop"),
        frequency_penalty=body.get("frequency_penalty"),
        presence_penalty=body.get("presence_penalty"),
        reasoning_effort=str(effort) if effort is not None else None,
        parallel_tool_calls=None if parallel is None else bool(parallel),
        structured_output=_structured_from_chat(body.get("response_format")),
    )


def _text_of(content) -> str:
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _responses_output_to_blocks(output) -> tuple:
    """Split a function_call_output payload into (text, images).

    Output may be a string, or an array of input_text / input_image /
    input_file parts (Codex view_image results). Previously arrays were
    flattened with str(), handing the model Python-repr garbage instead
    of viewable images. Unknown parts are ignored.
    """
    if output is None:
        return "", []
    if isinstance(output, str):
        return output, []
    if isinstance(output, dict):
        output = [output]
    if isinstance(output, list):
        texts: list = []
        images: list = []
        for part in output:
            if not isinstance(part, dict):
                continue
            kind = part.get("type")
            if kind == "input_text":
                texts.append(str(part.get("text", "")))
            elif kind == "input_image":
                url = str(part.get("image_url") or "")
                fid = str(part.get("file_id") or "")
                if url or fid:
                    images.append(ImageBlock(url, fid))
            elif kind == "input_file":
                label = part.get("filename") or part.get("file_id") or "unnamed"
                texts.append(f"[attached file: {label}]")
        return "".join(texts), images
    return _text_of(output), []


def _chat_part_to_block(part: dict):
    if not isinstance(part, dict):
        raise ValueError("unsupported chat content part: not an object")  # noqa: TRY004
    kind = part.get("type")
    if kind == "text":
        return TextBlock(part.get("text", ""))
    if kind == "image_url":
        ref = part.get("image_url", "")
        url = ref.get("url") if isinstance(ref, dict) else ref
        return ImageBlock(str(url))
    if kind in ("reasoning_content", "reasoning"):
        return ThinkingBlock(str(part.get("text", part.get("reasoning_content", ""))))
    raise ValueError(f"unsupported chat content part: {kind}")


def _canonical_tool_choice(choice) -> str | dict | None:
    """Normalize a per-dialect tool_choice into IR form.

    IR keeps the chat wire shape (OpenAI-style): "auto" | "required" |
    {"name": ...} | {"type": "function", "function": {...}}. Each egress
    renderer converts back to its own dialect (to_zen_chat passes it
    through, to_zen_responses unwraps function form, to_zen_messages maps
    to type any/tool/auto). Without this, a choice arriving on one dialect
    leaks verbatim onto another where it is invalid or misread.
    """
    if choice is None or isinstance(choice, str):
        return choice
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type", "auto")
    if kind == "tool":
        return {"name": choice.get("name", "")}
    if kind == "any":
        return "required"
    if kind == "auto":
        return "auto"
    fn = choice.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return {"name": fn["name"]}
    if choice.get("name"):
        return {"name": choice["name"]}
    return "auto"


def _chat_tool_to_ir(t: dict) -> ToolDef:
    fn = t.get("function", {})
    if not isinstance(fn, dict):
        fn = {}
    params = fn.get("parameters", {})
    return ToolDef(
        str(fn.get("name", "")),
        str(fn.get("description", "")),
        dict(params or {}),
    )


def from_chat(body: dict) -> RequestIR:
    messages: list[LlmMessage] = []
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            raise ValueError("unsupported chat message: not an object")  # noqa: TRY004
        role = msg.get("role")
        if role not in _CHAT_ROLES:
            raise ValueError(f"unsupported chat role: {role}")
        if role == ROLE_TOOL:
            content = msg.get("content")
            images: list = []
            if isinstance(content, list):
                # Image parts nested in tool content ride alongside the
                # text result (same shape as other legs).
                text = "".join(
                    str(p.get("text", ""))
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "image_url":
                        ref = p.get("image_url", "")
                        url = ref.get("url") if isinstance(ref, dict) else ref
                        if url:
                            images.append(ImageBlock(str(url)))
            else:
                text = _text_of(content)
            messages.append(
                LlmMessage(
                    role=ROLE_TOOL,
                    blocks=(
                        ToolResultBlock(
                            str(msg.get("tool_call_id", "")),
                            text,
                        ),
                        *images,
                    ),
                )
            )
            continue
        blocks: list = []
        content = msg.get("content")
        if isinstance(msg.get("reasoning_content"), str) and msg["reasoning_content"]:
            blocks.append(ThinkingBlock(msg["reasoning_content"]))
        if isinstance(content, str):
            blocks.append(TextBlock(content))
        elif isinstance(content, list):
            blocks.extend(_chat_part_to_block(p) for p in content)
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                raise ValueError("unsupported tool call: not an object")  # noqa: TRY004
            fn = call.get("function", {})
            if not isinstance(fn, dict):
                fn = {}
            if call.get("type", "function") != "function":
                raise ValueError(f"unsupported tool call type: {call.get('type')}")
            blocks.append(
                ToolCallBlock(
                    str(call.get("id", "")),
                    str(fn.get("name", "")),
                    str(fn.get("arguments", "")),
                )
            )
        messages.append(LlmMessage(role=role, blocks=tuple(blocks)))
    tools = tuple(
        _chat_tool_to_ir(t)
        for t in body.get("tools", [])
        if isinstance(t, dict) and t.get("type", "function") == "function"
    )
    return RequestIR(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=_canonical_tool_choice(body.get("tool_choice")),
        stream=body.get("stream") is True,
        params=_params_from_chat(body),
    )


def _responses_content_to_blocks(content, role: str = "user") -> list:
    blocks: list = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("unsupported responses content part: not an object")  # noqa: TRY004
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            blocks.append(TextBlock(part.get("text", "")))
        elif kind == "input_image":
            url = str(part.get("image_url") or "")
            file_id = str(part.get("file_id") or "")
            # file_id-only references (Files API) keep the id; neither
            # present means a meaningless part — drop it rather than
            # emitting an empty image_url upstream.
            if url or file_id:
                blocks.append(ImageBlock(url, file_id))
        elif kind == "input_file":
            # File bytes were never proxied; keep the turn alive with a
            # named placeholder instead of 400ing the session.
            label = part.get("filename") or part.get("file_id") or "unnamed"
            blocks.append(TextBlock(f"[attached file: {label}]"))
        # Unknown future part kinds are dropped (fail-open: a 400 here
        # would break the whole session; upstream validates fidelity).
    return blocks


def _responses_tool_to_ir(t: dict) -> ToolDef:
    if t.get("type", "function") == "function":
        params = t.get("parameters", {})
        return ToolDef(
            str(t.get("name", "")),
            str(t.get("description", "")),
            dict(params or {}),
        )
    params = t.get("parameters", {})
    tool_def = ToolDef(
        str(t.get("name", "")),
        str(t.get("description", "")),
        dict(params or {}),
        kind=str(t.get("type", "")),
        options={
            k: v
            for k, v in t.items()
            if k not in ("type", "name", "description", "parameters")
        },
    )
    return tool_def


def from_responses(body: dict) -> RequestIR:
    messages: list[LlmMessage] = []
    if body.get("instructions"):
        messages.append(
            LlmMessage(role=ROLE_SYSTEM, blocks=(TextBlock(str(body["instructions"])),))
        )
    raw = body.get("input", "")
    items = raw if isinstance(raw, list) else [raw]
    extra_tools: list = []
    for item in items:
        if isinstance(item, str):
            messages.append(LlmMessage(role=ROLE_USER, blocks=(TextBlock(item),)))
            continue
        if not isinstance(item, dict):
            raise ValueError(  # noqa: TRY004
                f"unsupported responses input item: {type(item).__name__}"
            )
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = ROLE_SYSTEM
            if role not in _RESPONSES_ROLES and role != ROLE_SYSTEM:
                raise ValueError(f"unsupported responses role: {role}")
            raw_content = item.get("content") or []
            if isinstance(raw_content, str):
                blocks = [TextBlock(raw_content)]
            else:
                blocks = _responses_content_to_blocks(raw_content)
            messages.append(LlmMessage(role=role, blocks=tuple(blocks)))
        elif kind == "function_call":
            call_id = str(item.get("call_id", item.get("id", "")))
            messages.append(
                LlmMessage(
                    role=ROLE_ASSISTANT,
                    blocks=(
                        ToolCallBlock(
                            call_id,
                            str(item.get("name", "")),
                            str(item.get("arguments", "")),
                        ),
                    ),
                )
            )
        elif kind == "function_call_output":
            text, images = _responses_output_to_blocks(item.get("output"))
            messages.append(
                LlmMessage(
                    role=ROLE_TOOL,
                    blocks=(
                        ToolResultBlock(str(item.get("call_id", "")), text),
                        *images,
                    ),
                )
            )
        elif kind == "reasoning":
            texts = []
            for part in (item.get("summary") or []) + (item.get("content") or []):
                if part.get("type") in ("summary_text", "reasoning_text", "text"):
                    texts.append(part.get("text", ""))
            if texts:
                messages.append(
                    LlmMessage(
                        role=ROLE_ASSISTANT, blocks=(ThinkingBlock("".join(texts)),)
                    )
                )
        elif kind == "additional_tools":
            # Codex deferred-tool definitions ride as input items, a shape
            # Zen rejects outright ("input[0] did not match any supported
            # type"). Dissolve the nested definitions into top-level tools
            # (the same tools the model sees, in the shape Zen validates)
            # and drop the item itself.
            for t in item.get("tools", []) or []:
                if isinstance(t, dict):
                    extra_tools.append(_responses_tool_to_ir(t))
        else:
            # Server-side history items (web_search_call, file_search_call,
            # computer_call(_output), item_reference, ...) have no IR
            # semantics here. Codex echoes prior outputs back verbatim;
            # 400ing breaks the session, so preserve them for the
            # responses leg (dropped on chat/messages legs). Upstream
            # remains the validator for truly invalid items.
            messages.append(
                LlmMessage(role=ROLE_ASSISTANT, blocks=(OpaqueBlock(dict(item)),))
            )
    tools = tuple(
        _responses_tool_to_ir(t) for t in body.get("tools", []) if isinstance(t, dict)
    ) + tuple(extra_tools)
    return RequestIR(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=_canonical_tool_choice(body.get("tool_choice")),
        stream=body.get("stream") is True,
        params=LlmParams(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_output_tokens"),
            reasoning_effort=(
                str(body["reasoning"].get("effort"))
                if isinstance(body.get("reasoning"), dict)
                and body["reasoning"].get("effort")
                else None
            ),
            parallel_tool_calls=(
                None
                if body.get("parallel_tool_calls") is None
                else bool(body.get("parallel_tool_calls"))
            ),
            structured_output=(
                _structured_from_responses(body.get("text", {}).get("format"))
                if isinstance(body.get("text"), dict)
                else None
            ),
        ),
    )


def _render_text(blocks: tuple) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def to_zen_chat(req: RequestIR) -> dict:
    body: dict = {"model": req.model, "messages": []}
    for msg in req.messages:
        if msg.role == ROLE_TOOL:
            if any(isinstance(b, ImageBlock) for b in msg.blocks):
                # Chat tool messages are text-only: images nested in tool
                # results cannot ride this leg and are dropped here (they
                # survive on responses/messages legs).
                logger.debug("chat egress dropping image in tool result")
            body["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": next(
                        (
                            b.call_id
                            for b in msg.blocks
                            if isinstance(b, ToolResultBlock)
                        ),
                        "",
                    ),
                    "content": "".join(
                        b.output for b in msg.blocks if isinstance(b, ToolResultBlock)
                    ),
                }
            )
            continue
        out: dict = {"role": msg.role}
        parts: list = []
        calls: list = []
        results: list = []
        thinking: list = []
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ThinkingBlock):
                thinking.append(b.text)
            elif isinstance(b, ImageBlock):
                if b.url:
                    parts.append({"type": "image_url", "image_url": {"url": b.url}})
                elif b.file_id:
                    # Chat completions cannot reference Files-API ids;
                    # keep the turn visible instead of dropping it.
                    parts.append(
                        {"type": "text", "text": f"[attached file: {b.file_id}]"}
                    )
            elif isinstance(b, ToolCallBlock):
                calls.append(
                    {
                        "id": b.call_id,
                        "type": "function",
                        "function": {"name": b.name, "arguments": b.wire_arguments()},
                    }
                )
            elif isinstance(b, ToolResultBlock):
                results.append(
                    {"role": "tool", "tool_call_id": b.call_id, "content": b.output}
                )
        if (
            len(parts) == 1
            and parts[0]["type"] == "text"
            and not calls
            and not results
            and not thinking
        ):
            out["content"] = parts[0]["text"]
        elif parts or calls:
            out["content"] = parts if parts else None
        elif thinking:
            out["content"] = None
        else:
            out = None
        if out is not None:
            if calls:
                out["tool_calls"] = calls
            if thinking:
                out["reasoning_content"] = "\n".join(thinking)
            body["messages"].append(out)
        body["messages"].extend(results)
    # Chat completions only support function tools: non-function tools
    # (web_search, ...) are dropped instead of raising (was an
    # unhandled 500 for responses/messages clients routed to chat).
    # Logged: the backend cannot execute server-side tools, so a
    # dropped web_search means the model may answer from memory.
    if req.tools:
        dropped = sorted({t.kind for t in req.tools if t.kind != "function"})
        if dropped:
            logger.debug("chat egress dropping non-function tools: %s", dropped)
        function_tools = [t for t in req.tools if t.kind == "function"]
        if function_tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in function_tools
            ]
    if req.tool_choice is not None:
        body["tool_choice"] = req.tool_choice
    params = req.params
    if params.temperature is not None:
        body["temperature"] = params.temperature
    if params.top_p is not None:
        body["top_p"] = params.top_p
    if params.max_tokens is not None:
        body["max_tokens"] = params.max_tokens
    if params.frequency_penalty is not None:
        body["frequency_penalty"] = params.frequency_penalty
    if params.presence_penalty is not None:
        body["presence_penalty"] = params.presence_penalty
    if params.stop is not None:
        body["stop"] = params.stop
    if params.reasoning_effort is not None:
        body["reasoning_effort"] = params.reasoning_effort
    if params.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = params.parallel_tool_calls
    if params.structured_output is not None:
        body["response_format"] = _chat_format_from_ir(params.structured_output)
    if req.stream:
        body["stream"] = True
    return body


def _chat_format_from_ir(structured: dict) -> dict:
    if structured.get("schema") is not None:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": structured.get("name") or "response",
                "schema": structured["schema"],
                "strict": bool(structured.get("strict", False)),
            },
        }
    return {"type": "json_object"}


def _responses_format_from_ir(structured: dict) -> dict:
    if structured.get("schema") is not None:
        return {
            "type": "json_schema",
            "name": structured.get("name") or "response",
            "schema": structured["schema"],
            "strict": bool(structured.get("strict", False)),
        }
    return {"type": "json_object"}


def _responses_tool_from_ir(t: ToolDef) -> dict:
    if t.kind == "function":
        return {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
    if t.kind.startswith("web_search"):
        return {"type": "web_search"}
    # Other built-in tools (file_search, computer, mcp, namespace, ...):
    # forward the definition as-is (fail-open; upstream validates).
    # description rides along ALWAYS (even empty): Zen 400s server tools
    # with a missing description key (seen live with codex namespace
    # containers) — its validator reads like pydantic "field required",
    # which an empty string satisfies but an absent key does not.
    tool: dict = {"type": t.kind, "description": t.description or ""}
    if t.name:
        tool["name"] = t.name
    if t.parameters:
        tool["parameters"] = t.parameters
    tool.update(t.options)
    return tool


def to_zen_responses(req: RequestIR) -> dict:
    body: dict = {"model": req.model, "input": []}
    systems = [
        b.text
        for m in req.messages
        if m.role == ROLE_SYSTEM
        for b in m.blocks
        if isinstance(b, TextBlock)
    ]
    if systems:
        body["instructions"] = "\n".join(systems)
    for msg in req.messages:
        if msg.role == ROLE_SYSTEM:
            continue
        if msg.role == ROLE_TOOL:
            images = [b for b in msg.blocks if isinstance(b, ImageBlock)]
            if images:
                # Tool results carrying images (Codex view_image): the
                # output array keeps text alongside viewable image parts
                # instead of flattening everything to a string.
                out: list = []
                text = "".join(
                    b.output for b in msg.blocks if isinstance(b, ToolResultBlock)
                )
                if text:
                    out.append({"type": "input_text", "text": text})
                for img in images:
                    if img.url:
                        out.append({"type": "input_image", "image_url": img.url})
                    elif img.file_id:
                        out.append({"type": "input_image", "file_id": img.file_id})
                body["input"].append(
                    {
                        "type": "function_call_output",
                        "call_id": next(
                            (
                                b.call_id
                                for b in msg.blocks
                                if isinstance(b, ToolResultBlock)
                            ),
                            "",
                        ),
                        "output": out,
                    }
                )
                continue
            for b in msg.blocks:
                if isinstance(b, ToolResultBlock):
                    body["input"].append(
                        {
                            "type": "function_call_output",
                            "call_id": b.call_id,
                            "output": b.output,
                        }
                    )
            continue
        content = []
        text_type = "output_text" if msg.role == ROLE_ASSISTANT else "input_text"
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                content.append({"type": text_type, "text": b.text})
            elif isinstance(b, ThinkingBlock):
                body["input"].append(
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": b.text}],
                    }
                )
            elif isinstance(b, ImageBlock):
                if b.url:
                    content.append({"type": "input_image", "image_url": b.url})
                elif b.file_id:
                    content.append({"type": "input_image", "file_id": b.file_id})
                # Neither: unresolvable reference — drop the part rather
                # than emitting an empty image_url upstream.
            elif isinstance(b, ToolCallBlock):
                body["input"].append(
                    {
                        "type": "function_call",
                        "call_id": b.call_id,
                        "name": b.name,
                        "arguments": b.wire_arguments(),
                    }
                )
            elif isinstance(b, ToolResultBlock):
                body["input"].append(
                    {
                        "type": "function_call_output",
                        "call_id": b.call_id,
                        "output": b.output,
                    }
                )
            elif isinstance(b, OpaqueBlock) and not b.part:
                # Top-level history items round-trip verbatim; part-level
                # (messages-dialect) opaques have no responses equivalent.
                body["input"].append(dict(b.item))
        if content:
            body["input"].append(
                {"type": "message", "role": msg.role, "content": content}
            )
    if req.tools:
        body["tools"] = [_responses_tool_from_ir(t) for t in req.tools]
    if req.tool_choice is not None:
        # Responses mirrors the chat wire shape (string or function-form
        # dict); a messages-shaped {"type": "tool"} choice canonicalized
        # at ingress into {"name": ...} would otherwise leak through.
        choice = req.tool_choice
        if isinstance(choice, dict) and choice.get("type") == "tool":
            choice = {"name": choice.get("name", "")}
        body["tool_choice"] = choice
    if req.params.temperature is not None:
        body["temperature"] = req.params.temperature
    if req.params.top_p is not None:
        body["top_p"] = req.params.top_p
    if req.params.max_tokens is not None:
        body["max_output_tokens"] = req.params.max_tokens
    if req.params.reasoning_effort is not None:
        body["reasoning"] = {"effort": req.params.reasoning_effort}
    if req.params.parallel_tool_calls is not None:
        body["parallel_tool_calls"] = req.params.parallel_tool_calls
    if req.params.structured_output is not None:
        body["text"] = {
            "format": _responses_format_from_ir(req.params.structured_output)
        }
    if req.stream:
        body["stream"] = True
    return body


def with_model(req: RequestIR, model: str) -> RequestIR:
    return replace(req, model=model)


def responses_output_to_ir_messages(output: list) -> tuple:
    messages: list = []
    for item in output:
        kind = item.get("type")
        if kind == "message":
            texts = [
                p.get("text", "")
                for p in (item.get("content") or [])
                if p.get("type") == "output_text"
            ]
            if texts:
                messages.append(
                    LlmMessage(role=ROLE_ASSISTANT, blocks=(TextBlock("".join(texts)),))
                )
        elif kind == "reasoning":
            texts = []
            for part in (item.get("summary") or []) + (item.get("content") or []):
                if part.get("type") in ("summary_text", "reasoning_text", "text"):
                    texts.append(part.get("text", ""))
            if texts:
                messages.append(
                    LlmMessage(
                        role=ROLE_ASSISTANT, blocks=(ThinkingBlock("".join(texts)),)
                    )
                )
    for item in output:
        kind = item.get("type")
        if kind == "function_call":
            messages.append(
                LlmMessage(
                    role=ROLE_ASSISTANT,
                    blocks=(
                        ToolCallBlock(
                            str(item.get("call_id", item.get("id", ""))),
                            str(item.get("name", "")),
                            str(item.get("arguments", "")),
                        ),
                    ),
                )
            )
        elif kind not in ("message", "reasoning"):
            messages.append(
                LlmMessage(role=ROLE_ASSISTANT, blocks=(OpaqueBlock(dict(item)),))
            )
    return tuple(messages)


def messages_content_to_ir_blocks(content: list) -> tuple:
    import json as _json

    blocks: list = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            blocks.append(TextBlock(part.get("text", "")))
        elif kind == "tool_use":
            blocks.append(
                ToolCallBlock(
                    str(part.get("id", "")),
                    str(part.get("name", "")),
                    _json.dumps(part.get("input", {})),
                )
            )
        elif kind in ("thinking", "redacted_thinking"):
            blocks.append(
                ThinkingBlock(str(part.get("thinking", part.get("data", ""))))
            )
        else:
            # Upstream server blocks (server_tool_use,
            # web_search_tool_result, ...) and any future block: keep as
            # part-level opaques instead of 502ing the convert leg.
            blocks.append(OpaqueBlock(dict(part), part=True))
    return tuple(blocks)


def ir_messages_to_messages_content(messages: tuple) -> list:
    import json as _json

    content: list = []
    for msg in messages:
        if msg.role != ROLE_ASSISTANT:
            continue
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                content.append({"type": "text", "text": b.text})
            elif isinstance(b, ThinkingBlock):
                content.append({"type": "thinking", "thinking": b.text})
            elif isinstance(b, ToolCallBlock):
                try:
                    arguments = _json.loads(b.arguments or "{}")
                except Exception:
                    arguments = {"_raw": b.arguments}
                content.append(
                    {
                        "type": "tool_use",
                        "id": b.call_id,
                        "name": b.name,
                        "input": arguments,
                    }
                )
    return content


def ir_messages_to_responses_output(messages: tuple) -> list:
    output: list = []
    for msg in messages:
        if msg.role != ROLE_ASSISTANT:
            continue
        texts = "".join(b.text for b in msg.blocks if isinstance(b, TextBlock))
        if texts:
            output.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": texts, "annotations": []}
                    ],
                }
            )
        for b in msg.blocks:
            if isinstance(b, ThinkingBlock):
                output.append(
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": b.text}],
                    }
                )
            elif isinstance(b, ToolCallBlock):
                output.append(
                    {
                        "type": "function_call",
                        "call_id": b.call_id,
                        "name": b.name,
                        "arguments": b.wire_arguments(),
                    }
                )
            elif isinstance(b, OpaqueBlock):
                output.append(dict(b.item))
    return output


def _messages_text_of(content) -> str:
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            texts.append(part.get("text", ""))
        elif isinstance(part, dict) and part.get("type") == "tool_result":
            inner = part.get("content", "")
            texts.append(inner if isinstance(inner, str) else _messages_text_of(inner))
    return "".join(texts)


def _messages_source_to_image(source: dict) -> ImageBlock | None:
    """ImageBlock from a messages image source, or None when unresolvable."""
    if not isinstance(source, dict):
        return None
    if source.get("type") == "url":
        url = str(source.get("url", ""))
        return ImageBlock(url) if url else None
    if source.get("type") == "base64":
        return ImageBlock(
            f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
        )
    if source.get("type") == "file_id":
        fid = str(source.get("file_id", ""))
        return ImageBlock("", fid) if fid else None
    return None


def _messages_images_of(content) -> list:
    """ImageBlocks for image parts nested in tool_result content."""
    images: list = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                img = _messages_source_to_image(part.get("source", {}))
                if img is not None:
                    images.append(img)
    return images


def _cache_ttl(part: dict) -> str | None:
    """Anthropic cache breakpoint type from a content/tool block, if set."""
    cc = part.get("cache_control")
    if isinstance(cc, dict) and isinstance(cc.get("type"), str):
        return cc["type"]
    return None


def _with_cache(block: dict, ttl: str | None) -> dict:
    if ttl:
        block["cache_control"] = {"type": ttl}
    return block


def _messages_block_to_ir(part: dict):
    if not isinstance(part, dict):
        raise ValueError("unsupported messages content block: not an object")  # noqa: TRY004
    kind = part.get("type")
    if kind == "text":
        return TextBlock(part.get("text", ""), _cache_ttl(part))
    if kind == "image":
        source = part.get("source", {})
        if isinstance(source, dict) and source.get("type") == "url":
            return ImageBlock(str(source.get("url", "")), cache=_cache_ttl(part))
        if isinstance(source, dict) and source.get("type") == "base64":
            return ImageBlock(
                f"data:{source.get('media_type', '')};base64,{source.get('data', '')}",
                cache=_cache_ttl(part),
            )
        if isinstance(source, dict) and source.get("type") == "file_id":
            # Files-API reference: round-trips verbatim on the messages
            # leg via part-level opaque; dropped on other legs.
            return OpaqueBlock(dict(part), part=True)
        raise ValueError(f"unsupported messages image source: {source}")
    if kind == "tool_use":
        import json as _json

        return ToolCallBlock(
            str(part.get("id", "")),
            str(part.get("name", "")),
            _json.dumps(part.get("input", {})),
            _cache_ttl(part),
        )
    if kind == "tool_result":
        content = part.get("content", "")
        ttl = _cache_ttl(part)
        if ttl is None and isinstance(content, list):
            # Claude Code often pins the breakpoint on an inner content
            # block instead of the tool_result itself — inherit it.
            for inner in content:
                if isinstance(inner, dict):
                    ttl = _cache_ttl(inner)
                    if ttl is not None:
                        break
        return [
            ToolResultBlock(
                str(part.get("tool_use_id", "")), _messages_text_of(content), ttl
            ),
            # Images nested in tool results (Claude Code Read of an image
            # file) ride alongside as image parts; without them the model
            # receives an empty result and stays blind.
            *_messages_images_of(content),
        ]
    if kind == "thinking":
        return ThinkingBlock(str(part.get("thinking", "")), _cache_ttl(part))
    if kind == "redacted_thinking":
        return ThinkingBlock(str(part.get("data", "")), _cache_ttl(part))
    # Server-side history blocks (server_tool_use, web_search_tool_result,
    # code_execution_tool_result, ...): Claude Code echoes prior turns
    # back verbatim; 400ing breaks the session, so they ride through as
    # part-level opaques (verbatim on the messages leg, dropped
    # elsewhere). Upstream validates truly invalid blocks.
    return OpaqueBlock(dict(part), part=True)


def from_messages(body: dict) -> RequestIR:
    messages: list[LlmMessage] = []
    system = body.get("system", "")
    if isinstance(system, str):
        if system:
            messages.append(LlmMessage(role=ROLE_SYSTEM, blocks=(TextBlock(system),)))
    elif isinstance(system, list):
        # Anthropic system arrays carry per-block cache breakpoints —
        # keep one TextBlock per text part instead of collapsing.
        blocks = [
            TextBlock(str(p.get("text", "")), _cache_ttl(p))
            for p in system
            if isinstance(p, dict) and p.get("type", "text") == "text"
        ]
        if blocks:
            messages.append(LlmMessage(role=ROLE_SYSTEM, blocks=tuple(blocks)))
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            raise ValueError("unsupported messages message: not an object")  # noqa: TRY004
        role = msg.get("role")
        if role not in (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM):
            raise ValueError(f"unsupported messages role: {role}")
        content = msg.get("content", "")
        if isinstance(content, str):
            blocks = [TextBlock(content)] if content else []
        elif isinstance(content, list):
            blocks = []
            for p in content:
                block = _messages_block_to_ir(p)
                # tool_result with nested images expands to several blocks.
                blocks.extend(block if isinstance(block, list) else [block])
        else:
            blocks = []
        messages.append(LlmMessage(role=role, blocks=tuple(blocks)))
    tools = tuple(
        ToolDef(
            str(
                t.get("name", "")
                or (
                    "web_search"
                    if str(t.get("type", "")).startswith("web_search")
                    else ""
                )
            ),
            str(t.get("description", "")),
            dict(t.get("input_schema", {}) or {}),
            kind=str(t.get("type", "function")),
            options={
                k: v
                for k, v in t.items()
                if k not in ("type", "name", "description", "input_schema")
            },
        )
        for t in body.get("tools", [])
        if isinstance(t, dict)
    )
    choice = _canonical_tool_choice(body.get("tool_choice"))
    stop = body.get("stop_sequences")
    effort: str | None = None
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        try:
            effort = effort_for_budget(int(thinking.get("budget_tokens", 4096)))
        except (TypeError, ValueError):
            effort = "medium"
    return RequestIR(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=choice,
        stream=body.get("stream") is True,
        params=LlmParams(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_tokens"),
            stop=stop,
            reasoning_effort=effort,
        ),
    )


def _messages_tool_from_ir(t: ToolDef) -> dict:
    if t.kind == "function":
        tool = {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        # Passthrough extras (cache_control breakpoints ride here).
        tool.update(t.options)
        return tool
    tool = {"type": t.kind, "name": t.name}
    if t.description:
        tool["description"] = t.description
    if t.parameters:
        tool["input_schema"] = t.parameters
    tool.update(t.options)
    return tool


def _messages_image_part(b: ImageBlock) -> dict | None:
    """Encode image bytes for the messages leg (Claude source shapes).

    Data URLs split back into base64 sources — the url source only
    accepts http(s), so stuffing data: bytes there was rejected
    upstream. file_id references round-trip natively. Empty images
    (unresolvable references) are dropped by the caller.
    """
    if b.url.startswith("data:"):
        header, _, data = b.url.partition(",")
        if ";base64" in header and data:
            media = header[len("data:") :].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media,
                    "data": data,
                },
            }
    if b.url:
        return {"type": "image", "source": {"type": "url", "url": b.url}}
    if b.file_id:
        return {"type": "image", "source": {"type": "file_id", "file_id": b.file_id}}
    return None


def to_zen_messages(req: RequestIR) -> dict:
    import json as _json

    body: dict = {"model": req.model, "messages": []}
    systems = [
        (b.text, b.cache)
        for m in req.messages
        if m.role == ROLE_SYSTEM
        for b in m.blocks
        if isinstance(b, TextBlock)
    ]
    if len(systems) == 1 and not systems[0][1]:
        body["system"] = systems[0][0]
    elif systems:
        # Multiple blocks or breakpoints: keep the array shape so
        # cache_control survives (a joined string would drop it).
        body["system"] = [
            _with_cache({"type": "text", "text": text}, ttl) for text, ttl in systems
        ]
    for msg in req.messages:
        if msg.role == ROLE_SYSTEM:
            continue
        role = ROLE_USER if msg.role == ROLE_TOOL else msg.role
        parts: list = []
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                parts.append(_with_cache({"type": "text", "text": b.text}, b.cache))
            elif isinstance(b, ThinkingBlock):
                parts.append(
                    _with_cache({"type": "thinking", "thinking": b.text}, b.cache)
                )
            elif isinstance(b, ImageBlock):
                part = _messages_image_part(b)
                if part is not None:
                    parts.append(_with_cache(part, b.cache))
            elif isinstance(b, ToolCallBlock):
                try:
                    arguments = _json.loads(b.arguments) if b.arguments else {}
                except Exception:
                    arguments = {"_raw": b.arguments}
                parts.append(
                    _with_cache(
                        {
                            "type": "tool_use",
                            "id": b.call_id,
                            "name": b.name,
                            "input": arguments,
                        },
                        b.cache,
                    )
                )
            elif isinstance(b, ToolResultBlock):
                parts.append(
                    _with_cache(
                        {
                            "type": "tool_result",
                            "tool_use_id": b.call_id,
                            "content": b.output,
                        },
                        b.cache,
                    )
                )
            elif isinstance(b, OpaqueBlock) and b.part:
                # Part-level (messages-dialect) history round-trips
                # verbatim; item-level (responses-dialect) opaques have no
                # messages equivalent and fall into the empty-parts guard.
                parts.append(dict(b.item))
            # OpaqueBlock (web_search_call et al.): no messages-leg
            # equivalent; dropped below via the empty-parts guard.
        if not parts:
            continue
        if len(parts) == 1 and parts[0]["type"] == "text":
            content = parts[0]["text"]
        else:
            content = parts
        body["messages"].append({"role": role, "content": content})
    if req.tools:
        body["tools"] = [_messages_tool_from_ir(t) for t in req.tools]
    if req.tool_choice is not None:
        choice = req.tool_choice
        if choice == "required":
            body["tool_choice"] = {"type": "any"}
        elif isinstance(choice, dict) and choice.get("type") == "tool":
            # Already messages-shaped (round-trip): pass through verbatim.
            body["tool_choice"] = choice
        elif isinstance(choice, dict) and "name" in choice:
            body["tool_choice"] = {"type": "tool", "name": choice["name"]}
        else:
            body["tool_choice"] = {"type": "auto"}
    params = req.params
    body["max_tokens"] = params.max_tokens if params.max_tokens is not None else 1024
    if params.stop is not None:
        # Responses has no stop parameter; the messages leg does.
        body["stop_sequences"] = (
            [params.stop] if isinstance(params.stop, str) else list(params.stop)
        )
    if params.temperature is not None:
        body["temperature"] = params.temperature
    if params.top_p is not None:
        body["top_p"] = params.top_p
    if params.reasoning_effort is not None:
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": EFFORT_BUDGETS.get(params.reasoning_effort, 4096),
        }
    if req.stream:
        body["stream"] = True
    return body
