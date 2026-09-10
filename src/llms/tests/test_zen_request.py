from __future__ import annotations

from proxy.zen_request import build_zen_request
from tests.conftest import make_settings


def test_drops_unknown_harness_fields():
    body = {
        "model": "muse-spark-1.3-contributor-free",
        "input": "hi",
        "harness_session": "abc",
        "custom_retries": 3,
    }
    request = build_zen_request(body, make_settings())
    assert request == {"model": "muse-spark-1.3-contributor-free", "input": "hi"}


def test_applies_default_model():
    request = build_zen_request({"input": "hi"}, make_settings())
    assert request["model"] == "muse-spark-1.3-contributor-free"


def test_keeps_known_responses_fields():
    body = {
        "model": "muse-spark-1.3-contributor-free",
        "instructions": "be brief",
        "input": "hi",
        "stream": False,
        "temperature": 0.5,
        "max_output_tokens": 64,
    }
    assert build_zen_request(body, make_settings()) == body
