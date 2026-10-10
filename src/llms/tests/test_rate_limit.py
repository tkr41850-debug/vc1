from __future__ import annotations

import pytest

from llms.proxy.rate_limit import classify


def test_ok_on_success():
    assert classify(200, {"output": []}) == ("ok", None)


@pytest.mark.parametrize(
    ("status", "payload", "headers", "expected"),
    [
        (429, {}, {"Retry-After": "5"}, ("ratelimited", 5.0)),
        (
            400,
            {
                "type": "error",
                "error": {"type": "FreeUsageLimitError", "message": "limit"},
            },
            None,
            ("ratelimited", None),
        ),
        (
            400,
            {"error": {"message": "Too many requests today"}},
            None,
            ("ratelimited", None),
        ),
        (
            400,
            {"detail": "Rate limit exceeded"},
            None,
            ("ratelimited", None),
        ),
        (
            400,
            {"error": {"detail": "quota exhausted"}},
            None,
            ("ratelimited", None),
        ),
    ],
    ids=["retry-after", "free-usage-error", "quota-message", "detail", "error-detail"],
)
def test_ratelimited_classify(status, payload, headers, expected):
    assert classify(status, payload, headers) == expected


def test_error_on_5xx():
    assert classify(500, {"error": {"message": "boom"}}) == ("error", None)
    assert classify(502, None) == ("error", None)


def test_ok_on_client_errors():
    assert classify(400, {"error": {"message": "bad model"}}) == ("ok", None)
    assert classify(401, {"error": {"message": "bad key"}}) == ("ok", None)


def test_bad_retry_after_header_tolerated():
    assert classify(429, {}, {"retry-after": "soon"}) == ("ratelimited", None)
