from __future__ import annotations

from dataclasses import replace

from proxy.ir import (
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_TOOL,
    ROLE_USER,
    ImageBlock,
    LlmMessage,
    LlmParams,
    LlmRequest,
    TextBlock,
    ToolCallBlock,
    ToolDef,
    ToolResultBlock,
)

_CHAT_ROLES = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_TOOL)
_RESPONSES_ROLES = ("system", "developer", "user", "assistant")


def _params_from_chat(body: dict) -> LlmParams:
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens"))
    return LlmParams(
        temperature=body.get("temperature"),
        top_p=body.get("top_p"),
        max_tokens=max_tokens,
        stop=body.get("stop"),
        frequency_penalty=body.get("frequency_penalty"),
        presence_penalty=body.get("presence_penalty"),
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
    raise ValueError(f"unsupported chat content part: {kind}")


def from_chat(body: dict) -> LlmRequest:
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
    return LlmRequest(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        stream=body.get("stream") is True,
        params=_params_from_chat(body),
    )


def _responses_content_to_blocks(content) -> list:
    blocks: list = []
    for part in content:
        kind = part.get("type")
        if kind == "input_text":
            blocks.append(TextBlock(part.get("text", "")))
        elif kind == "input_image":
            blocks.append(ImageBlock(str(part.get("image_url", ""))))
        else:
            raise ValueError(f"unsupported responses content part: {kind}")
    return blocks


def from_responses(body: dict) -> LlmRequest:
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
    return LlmRequest(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=body.get("tool_choice"),
        stream=body.get("stream") is True,
        params=LlmParams(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_output_tokens"),
        ),
    )


def _render_text(blocks: tuple) -> str:
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


def to_zen_chat(req: LlmRequest) -> dict:
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
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                parts.append({"type": "text", "text": b.text})
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
        if len(parts) == 1 and parts[0]["type"] == "text" and not calls:
            out["content"] = parts[0]["text"]
        elif parts:
            out["content"] = parts
        else:
            out["content"] = None
        if calls:
            out["tool_calls"] = calls
        body["messages"].append(out)
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
    if req.stream:
        body["stream"] = True
    return body


def to_zen_responses(req: LlmRequest) -> dict:
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
        for b in msg.blocks:
            if isinstance(b, TextBlock):
                content.append({"type": "input_text", "text": b.text})
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
    if req.stream:
        body["stream"] = True
    return body


def with_model(req: LlmRequest, model: str) -> LlmRequest:
    return replace(req, model=model)


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
    raise ValueError(f"unsupported messages content block: {kind}")


def from_messages(body: dict) -> LlmRequest:
    messages: list[LlmMessage] = []
    system = body.get("system", "")
    if system:
        messages.append(
            LlmMessage(role=ROLE_SYSTEM, blocks=(TextBlock(_messages_text_of(system)),))
        )
    for msg in body.get("messages", []):
        role = msg.get("role")
        if role not in (ROLE_USER, ROLE_ASSISTANT):
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
    return LlmRequest(
        model=str(body.get("model", "")),
        messages=tuple(messages),
        tools=tools,
        tool_choice=tool_choice,
        stream=body.get("stream") is True,
        params=LlmParams(
            temperature=body.get("temperature"),
            top_p=body.get("top_p"),
            max_tokens=body.get("max_tokens"),
        ),
    )


def to_zen_messages(req: LlmRequest) -> dict:
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
    if req.stream:
        body["stream"] = True
    return body
