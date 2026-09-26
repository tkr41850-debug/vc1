from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx

from llms.proxy.forward import translate_streaming
from llms.proxy.stream_translate import (
    STREAM_EMITTERS,
    STREAM_PARSERS,
    SseFramer,
    translate_lines,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> list[str]:
    return (FIXTURES / name).read_text(errors="replace").splitlines()


def _incremental(
    ingress: str, egress: str, lines: list[str], trace_id: str = "t", model: str = "m"
) -> bytes:
    framer = SseFramer()
    parser = STREAM_PARSERS[ingress]()
    emitter = STREAM_EMITTERS[egress](trace_id, model)
    out: list[bytes] = []
    for line in lines:
        for payload in framer.feed(line):
            for delta in parser.feed_payload(payload):
                out.extend(emitter.feed_delta(delta))
    for payload in framer.finish():
        for delta in parser.feed_payload(payload):
            out.extend(emitter.feed_delta(delta))
    done = parser.finish()
    if done is not None:
        out.extend(emitter.feed_delta(done))
    return b"".join(out)


def _batch(
    ingress: str, egress: str, lines: list[str], trace_id: str = "t", model: str = "m"
) -> bytes:
    return b"".join(translate_lines(ingress, egress, lines, trace_id, model))


CHAT_LINES = [
    'data: {"choices":[{"delta":{"content":"hi"}}]}',
    "",
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "",
    "data: [DONE]",
    "",
]

MESSAGES_LINES = [
    "event: content_block_start",
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "",
    "event: content_block_delta",
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
    "",
    "event: content_block_stop",
    'data: {"type":"content_block_stop","index":0}',
    "",
    "event: message_delta",
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}',
    "",
    "event: message_stop",
    'data: {"type":"message_stop"}',
    "",
]


def test_incremental_matches_batch_all_pairs():
    cases = [
        ("responses", _read_fixture("responses_incomplete.sse")),
        ("responses", _read_fixture("responses_rich_prefix.sse")),
        ("chat", CHAT_LINES),
        ("messages", MESSAGES_LINES),
    ]
    pairs = [
        ("responses", "chat"),
        ("responses", "messages"),
        ("chat", "responses"),
        ("chat", "messages"),
        ("messages", "responses"),
        ("messages", "chat"),
    ]
    for ingress, lines in cases:
        for from_d, to_d in pairs:
            if from_d != ingress:
                continue
            assert _incremental(from_d, to_d, lines) == _batch(from_d, to_d, lines), (
                from_d,
                to_d,
            )


def test_incremental_fixture_content():
    out = _incremental(
        "responses", "chat", _read_fixture("responses_rich_prefix.sse")
    ).decode()
    assert "tool_calls" in out
    assert out.rstrip().endswith("data: [DONE]")
    out_msg = _incremental(
        "responses", "messages", _read_fixture("responses_rich_prefix.sse")
    ).decode()
    assert "message_start" in out_msg
    assert "message_stop" in out_msg


def test_translate_streams_before_upstream_ends():
    """First translated chunk must arrive well before a slow upstream ends.

    Drives translate_streaming directly: Starlette's TestClient coalesces
    streamed bodies, so end-to-end timing can't be observed through it.
    """
    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        async def chunks():
            await asyncio.sleep(0.4)
            yield b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
            await asyncio.sleep(0.4)
            yield (
                b'data: {"type":"response.completed","response":{"status":"completed",'
                b'"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}}\n\n'
            )

        return httpx.Response(
            200, content=chunks(), headers={"content-type": "text/event-stream"}
        )

    async def run() -> tuple[float | None, float, int]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            req = client.build_request("POST", "http://x/y", json={})
            upstream = await client.send(req, stream=True)
            t0 = time.monotonic()
            first_at = None
            n = 0
            async for chunk in translate_streaming(
                upstream, "chat", "responses", "trace1", "m", "responses", None
            ):
                n += 1
                if first_at is None and b'"content": "hello"' in chunk:
                    first_at = time.monotonic() - t0
            return first_at, time.monotonic() - t0, n

    first_at, total, n = asyncio.run(run())
    assert n > 0
    assert first_at is not None, "no content chunk streamed"
    assert total > 0.6, "upstream stalls were not honored"
    assert first_at < total - 0.15, (
        f"first chunk ({first_at:.2f}s) did not beat upstream end ({total:.2f}s)"
    )
