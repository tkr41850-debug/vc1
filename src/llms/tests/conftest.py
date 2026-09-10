from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from llms.proxy.buckets import BucketTable
from llms.proxy.config import Settings
from llms.proxy.egress import DirectEgress
from llms.proxy.main import create_app


def make_settings(**overrides) -> Settings:
    base = {
        "zen_base_url": "https://opencode.ai/zen/v1",
        "zen_api_key": "",
        "opencode_version": "1.18.4",
        "opencode_client": "cli",
        "opencode_project": "global",
        "default_model": "muse-spark-1.3-contributor-free",
        "default_chat_model": "muse-spark-1.3-contributor-free",
        "default_messages_model": "claude-haiku-4-5",
        "allow_client_keys": False,
        "num_buckets": 1024,
        "num_slots": 8,
        "slot_cooldown_s": 60.0,
        "egress_mode": "direct",
        "model_aliases": (),
        "vsp_base_url": "",
        "vsp_token": "",
        "port": 8789,
        "request_timeout_s": 30.0,
        "data_dir": "",
        "static_dir": "",
        "github_client_id": "",
        "github_client_secret": "",
        "github_redirect_uri": "http://localhost:8789/api/admin/callback",
        "admin_github_users": (),
        "admin_session_secret": "test-secret",
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture()
def upstream_seen():
    return {}


@pytest.fixture()
def mock_upstream(upstream_seen):
    def handler(request: httpx.Request) -> httpx.Response:
        upstream_seen["url"] = str(request.url)
        upstream_seen["headers"] = dict(request.headers)
        try:
            import json as _json

            upstream_seen["json"] = _json.loads(request.content.decode())
        except Exception:
            upstream_seen["json"] = None
        mode = upstream_seen.pop("mode", "default")
        if str(request.url).endswith("/messages") and mode == "default":
            return httpx.Response(
                200,
                json={
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [{"type": "text", "text": "hello"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                },
            )
        if mode == "stream":
            if str(request.url).endswith("/chat/completions"):
                body = (
                    'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    "data: [DONE]\n\n"
                )
            else:
                body = (
                    'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
                    'data: {"inference-cost":123}\n\n'
                    'data: {"type":"response.completed"}\n\n'
                )
            return httpx.Response(
                200,
                content=body.encode(),
                headers={"content-type": "text/event-stream"},
            )
        if mode == "upstream_error":
            return httpx.Response(
                401,
                json={
                    "type": "error",
                    "error": {"type": "AuthError", "message": "bad key"},
                },
            )
        if str(request.url).endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "model": "mimo-v2.5-free",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "object": "response",
                "status": "completed",
                "model": "muse-spark-1.3-contributor-free",
                "output_text": "hello",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "hello", "annotations": []}
                        ],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        transport=transport, base_url="https://opencode.ai/zen/v1"
    )
    yield client, upstream_seen


TEST_SECRET = "sk-test"
TEST_HEADERS = {"Authorization": f"Bearer {TEST_SECRET}"}


def build_app_client(settings, mock_client, seed_key: str | None = None):
    from llms.proxy.store import ApiKey, Store

    if seed_key is not None:
        from pathlib import Path

        Store(data_dir=Path(settings.data_dir)).save_keys(
            [ApiKey(key=seed_key, label="test")]
        )
    app = create_app(settings)
    app.state.egress = DirectEgress(mock_client)
    app.state.bucket_table = BucketTable(num_buckets=1024, num_slots=1)
    return TestClient(app)


@pytest.fixture()
def app_client(mock_upstream, tmp_path):
    client, seen = mock_upstream
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        yield tc, seen


@pytest.fixture()
def admin_client(mock_upstream, tmp_path):
    from llms.proxy.auth import require_admin

    client, seen = mock_upstream
    with build_app_client(
        make_settings(data_dir=str(tmp_path)), client, seed_key=TEST_SECRET
    ) as tc:
        tc.app.dependency_overrides[require_admin] = lambda: "test-admin"
        yield tc, seen
