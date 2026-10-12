"""Relay-mode tests (local llms -> upstream llms as opencode traffic).

When UPSTREAM_RELAY_URL + UPSTREAM_RELAY_KEY are set, the local gateway must
forward the ANONYMOUS-shaped upstream request (genuine opencode wire:
Bearer public, opencode UA, ses_ session ids, genuine-12 tools) to the
upstream llms INSTEAD of opencode.ai/zen — and authenticate to the
upstream llms as a genuine opencode client.

Credential rule: the operator's relay key (sk-...) authenticates the
local->upstream hop on the standard Authorization header (the upstream
is an llms gateway: it expects sk- there). The upstream's own
middleware resolves it, strips affinity, and forwards to ITS pipeline
— which re-derives the anonymous opencode shape for its own upstream
hop. The local hop must therefore send the request EXACTLY as a
genuine opencode client would (anonymous tool injection, canonical
instructions, ses_ session), because the upstream's
`is_genuine_opencode` UA check decides passthrough: a non-opencode UA
gets the upstream's anonymous shaping applied ON TOP of the local
shaping (double notice, double tool injection).

Probe-vs-test split: hermetic tests assert the local hop's URL, auth,
and wire shape via MockTransport; live quota behavior lives in
scripts/, not here.
"""

from __future__ import annotations


def _relay_settings(**overrides):
    from tests.conftest import make_settings

    base = {
        "zen_base_url": "https://opencode.ai/zen/v1",
        "zen_api_key": "",
        "upstream_relay_url": "https://upstream.example",
        "upstream_relay_key": "sk-relay-test",
    }
    base.update(overrides)
    return make_settings(**base)


def test_relay_url_points_at_upstream_when_configured():
    """Relay configured: upstream URL is the relay endpoint, not zen's."""
    from llms.proxy.relay import relay_url_for

    settings = _relay_settings()
    assert relay_url_for(settings, "responses") == "https://upstream.example/responses"
    assert (
        relay_url_for(settings, "chat") == "https://upstream.example/chat/completions"
    )


def test_relay_url_falls_back_to_zen_without_config():
    """No relay config: URL is the direct zen endpoint (today's behavior)."""
    from llms.proxy.relay import relay_url_for

    settings = _relay_settings(upstream_relay_url="", upstream_relay_key="")
    assert (
        relay_url_for(settings, "responses") == "https://opencode.ai/zen/v1/responses"
    )


def test_relay_sends_sk_on_local_hop_not_upstream():
    """Local->upstream hop carries the relay sk- key; zen hop shape untouched."""
    from llms.proxy.relay import relay_headers_for

    settings = _relay_settings()
    local_headers = relay_headers_for(settings, "ses_abc123")
    # The relay key authenticates to the upstream (an llms gateway expecting sk-).
    assert local_headers["Authorization"] == "Bearer sk-relay-test"
    # ...but the opencode identity headers ride along so the upstream sees a
    # genuine-shaped client and passes through without re-shaping.
    assert local_headers["User-Agent"].startswith("opencode/")
    assert local_headers["x-opencode-session"] == "ses_abc123"


def test_relay_requires_both_url_and_key():
    """Half-configured relay is a config error, never a silent direct."""
    import pytest

    from llms.proxy.relay import relay_url_for

    with pytest.raises(ValueError):
        relay_url_for(_relay_settings(upstream_relay_key=""), "responses")
    with pytest.raises(ValueError):
        relay_url_for(_relay_settings(upstream_relay_url=""), "responses")


def test_relay_end_to_end_hits_upstream_not_zen(app_client=None):
    """Full pipeline with relay config: the upstream POST goes to the
    relay transport (relay URL + relay key), carrying the anonymous
    opencode-shaped body."""
    import httpx

    from llms.proxy.buckets import BucketTable
    from llms.proxy.config import Settings
    from llms.proxy.egress import DirectEgress
    from llms.proxy.keys import reset_cache
    from llms.proxy.main import create_app
    from llms.proxy.store import ApiKey, Store
    from tests.conftest import TEST_SECRET

    seen: dict = {}

    def relay_handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        try:
            import json as _json

            seen["json"] = _json.loads(request.content.decode())
        except Exception:
            seen["json"] = None
        # The anonymous leg always streams upstream and folds SSE back
        # into one JSON body, so the mock speaks SSE (not JSON).
        body = (
            'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
            'data: {"type":"response.completed"}\n\n'
        )
        return httpx.Response(
            200,
            content=body.encode(),
            headers={"content-type": "text/event-stream"},
        )

    from fastapi.testclient import TestClient

    relay_client = httpx.AsyncClient(
        transport=httpx.MockTransport(relay_handler),
        base_url="https://upstream.example",
    )
    reset_cache()
    import tempfile
    from pathlib import Path

    tmp = tempfile.mkdtemp()
    Store(data_dir=Path(tmp)).save_keys([ApiKey(key=TEST_SECRET, label="test")])
    from tests.conftest import make_settings as _mk

    settings: Settings = _mk(
        data_dir=tmp,
        zen_base_url="https://opencode.ai/zen/v1",
        zen_api_key="",
        upstream_relay_url="https://upstream.example",
        upstream_relay_key="sk-relay-test",
    )
    app = create_app(settings)
    app.state.egress = DirectEgress(relay_client)
    app.state.bucket_table = BucketTable(num_buckets=1024, num_slots=1)
    with TestClient(app) as tc:
        r = tc.post(
            "/v1/responses",
            json={"model": "muse-spark-1.3-contributor-free", "input": "hi"},
            headers={"Authorization": f"Bearer {TEST_SECRET}"},
        )
    assert r.status_code == 200, r.text[:500]
    assert seen["url"].startswith("https://upstream.example/")
    assert seen["headers"].get("authorization") == "Bearer sk-relay-test"
    assert seen["headers"].get("user-agent", "").startswith("opencode/")
    assert r.json()["output"][0]["content"][0]["text"] == "hello"
