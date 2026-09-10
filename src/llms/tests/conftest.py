from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy.config import Settings
from proxy.main import create_app


def make_settings(**overrides) -> Settings:
    base = {
        "zen_base_url": "https://opencode.ai/zen/v1",
        "zen_api_key": "",
        "opencode_version": "1.18.4",
        "opencode_client": "cli",
        "opencode_project": "global",
        "default_model": "muse-spark-1.3-contributor-free",
        "port": 8789,
        "request_timeout_s": 30.0,
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
        if mode == "stream":
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
        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "object": "response",
                "model": "muse-spark-1.3-contributor-free",
                "output_text": "hello",
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(
        transport=transport, base_url="https://opencode.ai/zen/v1"
    )
    yield client, upstream_seen


@pytest.fixture()
def app_client(mock_upstream):
    client, seen = mock_upstream
    app = create_app(make_settings())
    app.state.upstream_client = client
    with TestClient(app) as tc:
        yield tc, seen
