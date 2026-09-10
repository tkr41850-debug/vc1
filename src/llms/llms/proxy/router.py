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

FREE_RESPONSES_MODELS = (
    "muse-spark-1.3-contributor-free",
    "muse-spark-1.2-contributor-free",
)

FREE_CHAT_MODELS = (
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "ling-3.0-flash-fin-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "big-pickle",
)

FREE_MODELS = FREE_RESPONSES_MODELS + FREE_CHAT_MODELS


def normalize_model(model: str) -> str:
    name = model.strip().lower().removeprefix("opencode/")
    return name


def resolve_alias(model: str, aliases: tuple = ()) -> str:
    name = normalize_model(model)
    for pattern, target in aliases:
        pattern = pattern.strip().lower()
        if pattern.endswith("*") and name.startswith(pattern[:-1]):
            return target
        if pattern == name:
            return target
    return model


def pick(model: str, ingress: str) -> str:
    name = normalize_model(model)
    if name.startswith(MESSAGES_PREFIXES) and not name.startswith("qwen3-coder"):
        return "messages"
    if name.startswith(RESPONSES_PREFIXES):
        return "responses"
    if name.startswith(CHAT_PREFIXES):
        return "chat"
    return ingress
