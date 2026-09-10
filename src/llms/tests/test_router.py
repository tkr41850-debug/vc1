from __future__ import annotations

import pytest

from llms.proxy.router import normalize_model, pick


@pytest.mark.parametrize(
    "model,expected",
    [
        ("muse-spark-1.3-contributor-free", "responses"),
        ("opencode/muse-spark-1.3", "responses"),
        ("gpt-5.5", "responses"),
        ("GROK-4.5", "responses"),
        ("deepseek-v4-flash", "chat"),
        ("opencode/deepseek-v4-flash-free", "chat"),
        ("kimi-k2.6", "chat"),
        ("glm-5.1", "chat"),
        ("mimo-v2.5-free", "chat"),
        ("big-pickle", "chat"),
        ("claude-sonnet-4-5", "messages"),
        ("opencode/claude-opus-4-5", "messages"),
        ("qwen3.7-plus", "messages"),
        ("qwen3-coder", "chat"),
    ],
)
def test_pick_known_models(model, expected):
    assert pick(model, "chat") == expected
    assert pick(model, "responses") == expected


def test_pick_unknown_falls_back_to_ingress():
    assert pick("future-model-99", "chat") == "chat"
    assert pick("future-model-99", "responses") == "responses"


def test_normalize_strips_prefix_and_case():
    assert normalize_model("opencode/GPT-5.5") == "gpt-5.5"
    assert normalize_model("  mimo-v2.5-free ") == "mimo-v2.5-free"
