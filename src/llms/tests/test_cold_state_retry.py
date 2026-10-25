"""Cold-state 400 escape semantics (forward._send_with_cold_retry).

Zen's anonymous inference materializes session state asynchronously and
converts replayed assistant turns against it, so a follow-up turn can be
rejected with `[invalid_value] messages[N]: content must not be an
empty array` until the state settles (minutes in the tails). The helper
re-probes the identical request within bounds, then flattens the input
to user-only messages — a shape zen converts without state. Other 400s
relay at once.
"""

import asyncio
import json

import httpx

from llms.proxy import forward

_MARKER = (
    "Upstream request failed: [invalid_value] messages[3]: "
    "content must not be an empty array"
)


class _Scripted:
    """MockTransport handler replaying a status/message script."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.bodies = []

    def handler(self, request):
        self.calls += 1
        self.bodies.append(json.loads(request.content))
        status, message = self.script[min(self.calls - 1, len(self.script) - 1)]
        if status == 400:
            return httpx.Response(status, json={"error": {"message": message}})
        return httpx.Response(status, json={"ok": True})


def _client(script):
    scripted = _Scripted(script)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(scripted.handler),
        base_url="https://example.test",
    )
    return client, scripted


def _run(client, body):
    def factory(req_body):
        return client.build_request("POST", "/x", json=req_body)

    return asyncio.run(forward._send_with_cold_retry(client, body, factory, "trace"))


def _replay_body():
    return {
        "model": "glm-5.3-flash-free",
        "stream": True,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "plan"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "write",
                "arguments": '{"path": "a.txt"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "wrote it",
            },
        ],
    }


def test_cold_state_400_probes_then_flattens(monkeypatch):
    monkeypatch.setattr(forward, "COLD_STATE_RETRY_DELAYS_S", (0.01,))
    client, scripted = _client([(400, _MARKER), (400, _MARKER), (200, "")])
    resp = _run(client, _replay_body())
    assert resp.status_code == 200
    # 1 original + 1 faithful probe + 1 flattened resend.
    assert scripted.calls == 3
    # The probe kept the faithful body; the final send is flattened.
    assert scripted.bodies[0] == scripted.bodies[1]
    final_input = scripted.bodies[2]["input"]
    assert [it["type"] for it in final_input] == ["message", "message"]
    note = final_input[-1]["content"][0]["text"]
    assert "assistant tool call write" in note
    assert "tool result: wrote it" in note
    assert "assistant thinking: plan" in note
    # Untouched top-level fields survive the flatten.
    assert scripted.bodies[2]["stream"] is True


def test_fast_state_flip_keeps_faithful_body(monkeypatch):
    monkeypatch.setattr(forward, "COLD_STATE_RETRY_DELAYS_S", (0.01, 0.01))
    client, scripted = _client([(400, _MARKER), (200, "")])
    resp = _run(client, _replay_body())
    assert resp.status_code == 200
    assert scripted.calls == 2
    assert scripted.bodies[0] == scripted.bodies[1]


def test_no_probes_flatten_immediately(monkeypatch):
    monkeypatch.setattr(forward, "COLD_STATE_RETRY_DELAYS_S", ())
    client, scripted = _client([(400, _MARKER), (200, "")])
    resp = _run(client, _replay_body())
    assert resp.status_code == 200
    assert scripted.calls == 2
    assert [it["type"] for it in scripted.bodies[1]["input"]] == [
        "message",
        "message",
    ]


def test_exhausted_flatten_relays_final_400(monkeypatch):
    monkeypatch.setattr(forward, "COLD_STATE_RETRY_DELAYS_S", (0.01,))
    client, scripted = _client([(400, _MARKER)])
    resp = _run(client, _replay_body())
    assert resp.status_code == 400
    # 1 original + 1 probe + 1 flattened attempt, then relay.
    assert scripted.calls == 3
    body = asyncio.run(resp.aread())
    assert b"empty array" in body


def test_unrelated_400_not_retried(monkeypatch):
    monkeypatch.setattr(forward, "COLD_STATE_RETRY_DELAYS_S", (0.01, 0.01))
    client, scripted = _client([(400, "Invalid credential")])
    resp = _run(client, _replay_body())
    assert resp.status_code == 400
    assert scripted.calls == 1
    assert b"Invalid credential" in asyncio.run(resp.aread())


def test_clean_200_single_shot(monkeypatch):
    monkeypatch.setattr(forward, "COLD_STATE_RETRY_DELAYS_S", (0.01,))
    client, scripted = _client([(200, "")])
    resp = _run(client, _replay_body())
    assert resp.status_code == 200
    assert scripted.calls == 1


def test_messages_only_body_not_flattened(monkeypatch):
    monkeypatch.setattr(forward, "COLD_STATE_RETRY_DELAYS_S", (0.01,))
    client, scripted = _client([(400, _MARKER)])
    body = {"input": [{"type": "message", "role": "user", "content": []}]}
    resp = _run(client, body)
    # Nothing to flatten: probes exhaust, then relay without a third send.
    assert resp.status_code == 400
    assert scripted.calls == 2


def test_flatten_preserves_order_and_other_items():
    body = _replay_body()
    body["input"].insert(
        2,
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "do it"}],
        },
    )
    flat = forward._flatten_for_cold_state(body)
    roles = [it["role"] for it in flat["input"]]
    # Messages keep their order; the inlined note lands last.
    assert roles == ["user", "user", "user"]
    assert flat["input"][1]["content"][0]["text"] == "do it"
    assert "assistant thinking: plan" in flat["input"][-1]["content"][0]["text"]
    # Original body untouched.
    assert len(body["input"]) == 5
