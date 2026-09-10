from __future__ import annotations

from dataclasses import replace

from llms.proxy.ir import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    ImageBlock,
    LlmMessage,
    LlmParams,
    RequestIR,
    TextBlock,
    ThinkingBlock,
    ToolCallBlock,
    ToolDef,
    ToolResultBlock,
)

_CHAT_ROLES = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL)
_RESPONSES_ROLES = ("system", "developer", "user", "assistant")

EFFORT_BUDGETS = {"low": 1024, "medium": 4096, "high": 16384}


def effort_for_budget(budget: int) -> str:
    if budget <= 1024:
        return "low"
    if budget <= 8192:
        return "medium"
    return "high"


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
    )


def _text_of(content) -> str:
    if content is None:
        return ""
    return content if isinstance(content, str) else str(content)


def _chat_part_to_block(part: dict):
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


def from_chat(body: dict) -> RequestIR:
    messages: list[LlmMessage] = []
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in _CHAT_ROLES:
            raise ValueError(f"unsupported chat role: {role}")
        if role == ROLE_TOOL:
            messages.append(
                LlmMessage(
                    role=ROLE_TOOL,
                    blocks=(
                        ToolResultBlock(
                            str(msg.get("tool_call_id", "")),
                            _text_of(msg.get("content")),
                        ),
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
        for call in msg.get("tool_calls", []):
            fn = call.get("function", {})
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
        ToolDef(
            str(t.get("function", {}).get("name", "")),
            str(t.get("function", {}).get("description", "")),
            dict(t.get("function", {}).get("parameters", {})),
        )
        for t in body.get("tools", [])
        if t.get("type", "function") == "function"
    )
    return RequestIR(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        stream=body.get("stream") is True,
        params=_params_from_chat(body),
    )


def _responses_content_to_blocks(content, role: str = "user") -> list:
    blocks: list = []
    for part in content:
        kind = part.get("type")
        if kind in ("input_text", "output_text"):
            blocks.append(TextBlock(part.get("text", "")))
        elif kind == "input_image":
            blocks.append(ImageBlock(str(part.get("image_url", ""))))
        else:
            raise ValueError(f"unsupported responses content part: {kind}")
    return blocks


def from_responses(body: dict) -> RequestIR:
    messages: list[LlmMessage] = []
    if body.get("instructions"):
        messages.append(
            LlmMessage(role=ROLE_SYSTEM, blocks=(TextBlock(str(body["instructions"])),))
        )
    raw = body.get("input", "")
    items = raw if isinstance(raw, list) else [raw]
    for item in items:
        if isinstance(item, str):
            messages.append(LlmMessage(role=ROLE_USER, blocks=(TextBlock(item),)))
            continue
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = ROLE_SYSTEM
            if role not in _RESPONSES_ROLES and role != ROLE_SYSTEM:
                raise ValueError(f"unsupported responses role: {role}")
            raw_content = item.get("content", [])
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
            messages.append(
                LlmMessage(
                    role=ROLE_TOOL,
                    blocks=(
                        ToolResultBlock(
                            str(item.get("call_id", "")), _text_of(item.get("output"))
                        ),
                    ),
                )
            )
        elif kind == "reasoning":
            texts = []
            for part in item.get("summary", []) + item.get("content", []):
                if part.get("type") in ("summary_text", "reasoning_text", "text"):
                    texts.append(part.get("text", ""))
            if texts:
                messages.append(
                    LlmMessage(
                        role=ROLE_ASSISTANT, blocks=(ThinkingBlock("".join(texts)),)
                    )
                )
        else:
            raise ValueError(f"unsupported responses input item: {kind}")
    tools = tuple(
        ToolDef(
            str(t.get("name", "")),
            str(t.get("description", "")),
            dict(t.get("parameters", {})),
        )
        for t in body.get("tools", [])
        if t.get("type") == "function"
    )
    return RequestIR(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=body.get("tool_choice"),
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
        ),
    )


def _render_text(blocks: tuple) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def to_zen_chat(req: RequestIR) -> dict:
    body: dict = {"model": req.model, "messages": []}
    for msg in req.messages:
        if msg.role == ROLE_TOOL:
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
                parts.append({"type": "image_url", "image_url": {"url": b.url}})
            elif isinstance(b, ToolCallBlock):
                calls.append(
                    {
                        "id": b.call_id,
                        "type": "function",
                        "function": {"name": b.name, "arguments": b.arguments},
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
    if req.tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in req.tools
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
    if req.stream:
        body["stream"] = True
    return body


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
                content.append({"type": "input_image", "image_url": b.url})
            elif isinstance(b, ToolCallBlock):
                body["input"].append(
                    {
                        "type": "function_call",
                        "call_id": b.call_id,
                        "name": b.name,
                        "arguments": b.arguments,
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
        if content:
            body["input"].append(
                {"type": "message", "role": msg.role, "content": content}
            )
    if req.tools:
        body["tools"] = [
            {
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in req.tools
        ]
    if req.tool_choice is not None:
        body["tool_choice"] = req.tool_choice
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
                for p in item.get("content", [])
                if p.get("type") == "output_text"
            ]
            if texts:
                messages.append(
                    LlmMessage(role=ROLE_ASSISTANT, blocks=(TextBlock("".join(texts)),))
                )
        elif kind == "reasoning":
            texts = []
            for part in item.get("summary", []) + item.get("content", []):
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
    return tuple(messages)


def messages_content_to_ir_blocks(content: list) -> tuple:
    import json as _json

    blocks: list = []
    for part in content:
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
        else:
            raise ValueError(f"unsupported messages response block: {kind}")
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
                        "arguments": b.arguments,
                    }
                )
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


def _messages_block_to_ir(part: dict):
    kind = part.get("type")
    if kind == "text":
        return TextBlock(part.get("text", ""))
    if kind == "image":
        source = part.get("source", {})
        if isinstance(source, dict) and source.get("type") == "url":
            return ImageBlock(str(source.get("url", "")))
        if isinstance(source, dict) and source.get("type") == "base64":
            return ImageBlock(
                f"data:{source.get('media_type', '')};base64,{source.get('data', '')}"
            )
        raise ValueError(f"unsupported messages image source: {source}")
    if kind == "tool_use":
        import json as _json

        return ToolCallBlock(
            str(part.get("id", "")),
            str(part.get("name", "")),
            _json.dumps(part.get("input", {})),
        )
    if kind == "tool_result":
        return ToolResultBlock(
            str(part.get("tool_use_id", "")), _messages_text_of(part.get("content", ""))
        )
    if kind == "thinking":
        return ThinkingBlock(str(part.get("thinking", "")))
    if kind == "redacted_thinking":
        return ThinkingBlock(str(part.get("data", "")))
    raise ValueError(f"unsupported messages content block: {kind}")


def from_messages(body: dict) -> RequestIR:
    messages: list[LlmMessage] = []
    system = body.get("system", "")
    if system:
        messages.append(
            LlmMessage(role=ROLE_SYSTEM, blocks=(TextBlock(_messages_text_of(system)),))
        )
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in (ROLE_USER, ROLE_ASSISTANT, ROLE_SYSTEM):
            raise ValueError(f"unsupported messages role: {role}")
        content = msg.get("content", "")
        if isinstance(content, str):
            blocks = [TextBlock(content)]
        else:
            blocks = [_messages_block_to_ir(p) for p in content]
        messages.append(LlmMessage(role=role, blocks=tuple(blocks)))
    tools = tuple(
        ToolDef(
            str(t.get("name", "")),
            str(t.get("description", "")),
            dict(t.get("input_schema", {})),
        )
        for t in body.get("tools", [])
    )
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        kind = choice.get("type", "auto")
        tool_choice = {"auto": "auto", "any": "required"}.get(kind, choice)
        if kind == "tool":
            tool_choice = {"name": choice.get("name", "")}
    else:
        tool_choice = choice
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
        tool_choice=tool_choice,
        stream=body.get("stream") is True,
        params=LlmParams(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_tokens"),
            reasoning_effort=effort,
        ),
    )


def to_zen_messages(req: RequestIR) -> dict:
    import json as _json

    body: dict = {"model": req.model, "messages": []}
    systems = [
        b.text
        for m in req.messages
        if m.role == ROLE_SYSTEM
        for b in m.blocks
        if isinstance(b, TextBlock)
    ]
    if systems:
        body["system"] = "\n".join(systems)
    for msg in req.messages:
        if msg.role == ROLE_SYSTEM:
            continue
        role = ROLE_USER if msg.role == ROLE_TOOL else msg.role
        parts: list = []
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                parts.append({"type": "text", "text": b.text})
            elif isinstance(b, ThinkingBlock):
                parts.append({"type": "thinking", "thinking": b.text})
            elif isinstance(b, ImageBlock):
                parts.append({"type": "image", "source": {"type": "url", "url": b.url}})
            elif isinstance(b, ToolCallBlock):
                try:
                    arguments = _json.loads(b.arguments) if b.arguments else {}
                except Exception:
                    arguments = {"_raw": b.arguments}
                parts.append(
                    {
                        "type": "tool_use",
                        "id": b.call_id,
                        "name": b.name,
                        "input": arguments,
                    }
                )
            elif isinstance(b, ToolResultBlock):
                parts.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": b.call_id,
                        "content": b.output,
                    }
                )
        if len(parts) == 1 and parts[0]["type"] == "text":
            content = parts[0]["text"]
        else:
            content = parts
        body["messages"].append({"role": role, "content": content})
    if req.tools:
        body["tools"] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in req.tools
        ]
    if req.tool_choice is not None:
        choice = req.tool_choice
        if choice == "required":
            body["tool_choice"] = {"type": "any"}
        elif isinstance(choice, dict) and "name" in choice:
            body["tool_choice"] = {"type": "tool", "name": choice["name"]}
        else:
            body["tool_choice"] = {"type": "auto"}
    params = req.params
    body["max_tokens"] = params.max_tokens if params.max_tokens is not None else 1024
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
