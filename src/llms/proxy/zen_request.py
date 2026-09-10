from __future__ import annotations

from proxy.config import Settings

ALLOWED_FIELDS = (
    "model",
    "input",
    "instructions",
    "stream",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_output_tokens",
    "temperature",
    "top_p",
    "top_logprobs",
    "truncation",
    "background",
    "reasoning",
    "text",
    "metadata",
    "previous_response_id",
)


def build_zen_request(body: dict, settings: Settings) -> dict:
    request = {k: body[k] for k in ALLOWED_FIELDS if k in body}
    if not request.get("model"):
        request["model"] = settings.default_model
    return request
