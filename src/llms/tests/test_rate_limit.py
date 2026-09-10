from __future__ import annotations

from llms.proxy.rate_limit import classify


def test_ok_on_success():
    assert classify(200, {"output": []}) == ("ok", None)


def test_ratelimited_on_429_with_retry_after():
    assert classify(429, {}, {"Retry-After": "5"}) == ("ratelimited", 5.0)


def test_ratelimited_on_zen_free_usage_error():
    payload = {
        "type": "error",
        "error": {"type": "FreeUsageLimitError", "message": "limit"},
    }
    assert classify(400, payload) == ("ratelimited", None)


def test_ratelimited_on_quota_message():
    assert classify(400, {"error": {"message": "Too many requests today"}}) == (
        "ratelimited",
        None,
    )


def test_error_on_5xx():
    assert classify(500, {"error": {"message": "boom"}}) == ("error", None)
    assert classify(502, None) == ("error", None)


def test_ok_on_client_errors():
    assert classify(400, {"error": {"message": "bad model"}}) == ("ok", None)
    assert classify(401, {"error": {"message": "bad key"}}) == ("ok", None)


def test_bad_retry_after_header_tolerated():
    assert classify(429, {}, {"retry-after": "soon"}) == ("ratelimited", None)
