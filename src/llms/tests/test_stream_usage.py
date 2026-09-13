from __future__ import annotations

import pytest

from llms.proxy.ir import StreamDone
from llms.proxy.stream_translate import parse_chat_sse


def _responses_lines(in_tok=7, out_tok=3, cached=2, reasoning=1) -> list[str]:
    import json as _json

    def ev(payload: dict) -> str:
        # Compact separators keep every data line short: the ASGI test
        # transport truncates over-long lines in flight, but the tap sees
        # full upstream bytes either way.
        return "data: " + _json.dumps(payload, separators=(",", ":"))

    return [
        ev({"type": "response.output_text.delta", "delta": "hi"}),
        "",
        # Short keys AND short values: the ASGI test transport truncates
        # over-long data lines in flight (production TCP has no such cap).
        # Only the token/detail shape matters here — full field-name
        # coverage lives in test_translate_response.py.
        ev(
            {
                "type": "response.completed",
                "response": {
                    "id": "r",
                    "status": "completed",
                    "usage": {
                        "in": in_tok,
                        "o": out_tok,
                        "in_d": {"c": cached},
                        "o_d": {"r": reasoning},
                    },
                },
            }
        ),
        "",
    ]


def _messages_lines(in_tok=7, out_tok=3, cached=4) -> list[str]:
    import json as _json

    def data(payload: dict) -> str:
        return "data: " + _json.dumps(payload, separators=(",", ":"))

    return [
        "event: message_delta",
        data(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"in": in_tok, "o": out_tok, "cr": cached},
            }
        ),
        "",
        "event: message_stop",
        data({"type": "message_stop"}),
        "",
    ]


def _chat_lines_with_usage(in_tok=9, out_tok=5, cached=3) -> list[str]:
    import json as _json

    def ev(payload: dict) -> str:
        return "data: " + _json.dumps(payload, separators=(",", ":"))

    short_usage = {"p": in_tok, "c": out_tok, "p_det": {"cached": cached}}
    return [
        ev({"choices": [{"delta": {"content": "hi"}}]}),
        "",
        # Short keys: the ASGI test transport truncates over-long data
        # lines in flight (production TCP has no such cap).
        ev({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "",
        ev({"choices": [{"delta": {}}], "usage": dict(short_usage)}),
        "",
        "data: [DONE]",
        "",
    ]


def _chat_lines_without_usage() -> list[str]:
    import json as _json

    return [
        "data: " + _json.dumps({"choices": [{"delta": {"content": "hi"}}]}),
        "",
        "data: " + _json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        "",
        "data: [DONE]",
        "",
    ]


def test_parse_responses_sse_carries_usage_details():
    from llms.proxy.stream_translate import parse_responses_sse

    dones = [
        d for d in parse_responses_sse(_responses_lines()) if isinstance(d, StreamDone)
    ]
    assert len(dones) == 1
    done = dones[0]
    assert (done.input_tokens, done.output_tokens) == (7, 3)
    assert (done.cached_tokens, done.reasoning_tokens) == (2, 1)


def test_parse_messages_sse_carries_cache_read():
    from llms.proxy.stream_translate import parse_messages_sse

    dones = [
        d for d in parse_messages_sse(_messages_lines()) if isinstance(d, StreamDone)
    ]
    assert len(dones) == 1
    done = dones[0]
    assert (done.input_tokens, done.output_tokens) == (7, 3)
    assert done.cached_tokens == 4
    assert done.reasoning_tokens is None


def test_parse_chat_sse_carries_usage_chunk():
    dones = [
        d for d in parse_chat_sse(_chat_lines_with_usage()) if isinstance(d, StreamDone)
    ]
    assert len(dones) == 1
    done = dones[0]
    assert (done.input_tokens, done.output_tokens) == (9, 5)
    assert done.cached_tokens == 3


def test_parse_chat_sse_usage_inside_choice():
    import json as _json

    lines = [
        "data: "
        + _json.dumps(
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                        "usage": {"prompt_tokens": 9, "completion_tokens": 5},
                    }
                ]
            },
            separators=(",", ":"),
        ),
        "",
    ]
    dones = [d for d in parse_chat_sse(lines) if isinstance(d, StreamDone)]
    assert len(dones) == 1
    # Usage inside the choice object is not a documented OpenAI shape;
    # the parser only honors top-level usage chunks.
    assert (dones[0].input_tokens, dones[0].output_tokens) == (None, None)


def test_parse_chat_sse_without_usage_stays_none():
    dones = [
        d
        for d in parse_chat_sse(_chat_lines_without_usage())
        if isinstance(d, StreamDone)
    ]
    assert len(dones) == 1
    done = dones[0]
    assert done.input_tokens is None
    assert done.output_tokens is None


def _stream_app(monkeypatch, tmp_path, upstream_body: bytes, ingress: str, model: str):
    import httpx
    from fastapi.testclient import TestClient

    from llms.proxy.buckets import BucketTable
    from llms.proxy.egress import DirectEgress
    from llms.proxy.main import create_app
    from llms.proxy.store import ApiKey, Store
    from tests.conftest import TEST_HEADERS, TEST_SECRET, make_settings

    async def handler(request: httpx.Request) -> httpx.Response:
        # One SSE frame per chunk with short data lines: this keeps every
        # line under the ASGI test transport's per-message cap (long lines
        # are truncated in flight there; production TCP has no such cap and
        # the tap reassembles split frames regardless).
        frames = upstream_body.split(b"\n\n")
        assert all(len(f) < 200 for f in frames), "fixture lines must stay short"

        async def chunks():
            for f in frames:
                yield f + b"\n\n"

        return httpx.Response(
            200,
            content=chunks(),
            headers={"content-type": "text/event-stream"},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    Store(data_dir=tmp_path).save_keys([ApiKey(key=TEST_SECRET)])
    app = create_app(make_settings(data_dir=str(tmp_path)))
    app.state.egress = DirectEgress(client)
    app.state.bucket_table = BucketTable(num_buckets=1024, num_slots=1)
    tc = TestClient(app)
    return tc, TEST_HEADERS


def _drain(text: str) -> None:
    # Consume the whole SSE body so the usage tap fires.
    assert "text/event-stream" in text or "data:" in text


@pytest.mark.parametrize(
    "ingress,path,model,lines,expect",
    [
        (
            "responses",
            "v1/responses",
            "muse-spark-1.3-contributor-free",
            None,  # built below
            (7, 3, 2, 1),
        ),
        ("messages", "v1/messages", "claude-haiku-4-5", None, (7, 3, 4, None)),
        ("chat", "v1/chat/completions", "mimo-v2.5-free", None, (9, 5, 3, None)),
    ],
)
def test_stream_records_usage_end_to_end(
    monkeypatch, tmp_path, ingress, path, model, lines, expect
):
    if ingress == "responses":
        raw = _responses_lines()
        body = {"model": model, "input": "hi", "stream": True}
    elif ingress == "messages":
        raw = _messages_lines()
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": True,
        }
    else:
        raw = _chat_lines_with_usage()
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
    tc, headers = _stream_app(
        monkeypatch, tmp_path, "\n".join(raw).encode(), ingress, model
    )
    with tc:
        with tc.stream("POST", f"/{path}", json=body, headers=headers) as r:
            assert r.status_code == 200, r.text
            chunks = b"".join(r.iter_bytes())
            assert b"data:" in chunks
        # The usage tap records synchronously at stream exhaustion, inside
        # the response lifecycle — assert inline while the portal is live.
        from tests.conftest import TEST_SECRET

        snap = tc.app.state.usage.snapshot()["keys"][TEST_SECRET]
        assert snap["requests"] == 1
        assert snap["input_tokens"] == expect[0]
        assert snap["output_tokens"] == expect[1]
        assert snap["cached_tokens"] == (expect[2] or 0)
        assert snap["reasoning_tokens"] == (expect[3] or 0)


def test_stream_without_upstream_usage_records_request_only(monkeypatch, tmp_path):
    raw = _chat_lines_without_usage()
    tc, headers = _stream_app(
        monkeypatch, tmp_path, "\n".join(raw).encode(), "chat", "mimo-v2.5-free"
    )
    with (
        tc,
        tc.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "mimo-v2.5-free",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            headers=headers,
        ) as r,
    ):
        assert r.status_code == 200, r.text
        chunks = b"".join(r.iter_bytes())
        assert b"data:" in chunks
    from tests.conftest import TEST_SECRET

    snap = tc.app.state.usage.snapshot()["keys"][TEST_SECRET]
    assert snap["requests"] == 1
    assert snap["input_tokens"] == 0
