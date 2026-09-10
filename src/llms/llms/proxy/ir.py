from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ImageBlock:
    url: str


@dataclass(frozen=True)
class ToolCallBlock:
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResultBlock:
    call_id: str
    output: str


@dataclass(frozen=True)
class LlmMessage:
    role: str
    blocks: tuple = ()


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LlmParams:
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    reasoning_effort: str | None = None
    parallel_tool_calls: bool | None = None


@dataclass(frozen=True)
class LlmRequest:
    model: str
    messages: tuple = ()
    tools: tuple = ()
    tool_choice: str | dict | None = None
    stream: bool = False
    params: LlmParams = field(default_factory=LlmParams)


ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolArgsDelta:
    call_id: str
    name: str
    args_chunk: str


@dataclass(frozen=True)
class StreamDone:
    status: str
    has_tool_calls: bool = False
