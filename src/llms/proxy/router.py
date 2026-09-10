from __future__ import annotations

RESPONSES_PREFIXES = ("muse-spark", "gpt-", "grok-")

CHAT_PREFIXES = (
    "deepseek",
    "kimi",
    "glm-",
    "minimax",
    "mimo",
    "big-pickle",
    "ling-",
    "nemotron",
    "qwen3-coder",
    "trinity",
    "ring-",
    "north-",
    "hy3",
)

MESSAGES_PREFIXES = ("claude-", "qwen")

ENDPOINT_PATH = {
    "responses": "/responses",
    "chat": "/chat/completions",
    "messages": "/messages",
}


def normalize_model(model: str) -> str:
    name = model.strip().lower().removeprefix("opencode/")
    return name


def pick(model: str, ingress: str) -> str:
    name = normalize_model(model)
    if name.startswith(MESSAGES_PREFIXES) and not name.startswith("qwen3-coder"):
        return "messages"
    if name.startswith(RESPONSES_PREFIXES):
        return "responses"
    if name.startswith(CHAT_PREFIXES):
        return "chat"
    return ingress
