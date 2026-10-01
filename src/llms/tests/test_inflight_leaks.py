"""In-flight accounting must not leak on post-track early exits.

Regression tests for the cancelled-request leak: every path that leaves
run() after _inflight_track() must release the slot — early 429 returns,
replay returns, and exceptions (incl. CancelledError) from the forward
leg. Hermetic: drives pipeline.run() directly with stubbed app state.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from types import SimpleNamespace

import httpx

import llms.proxy.pipeline as pipeline
from llms.proxy import dedup
from llms.proxy.buckets import BucketTable
from llms.proxy.config import Settings
from llms.proxy.providers import ProviderRegistry
from llms.proxy.zen_headers import stable_session_id

MODEL = "deepseek-v4-flash-free"  # chat ingress == chat egress (no translate/steer)
PROVIDER = "w1"


class _WarpEgress:
    """Minimal warp egress: ready exit serving a stub httpx client."""

    def __init__(self, client):
        self._client = client

    def client_for(self, bucket, slot):
        return self._client

    def ready_ports(self):
        return [40001]

    def pick_port(self, slot):
        return 40001


class _Egress:
    """resolve() pins a warp provider so provider_id survives run().

    (A "noproxy" kind would reset provider_id to None before tracking, so
    the test would exercise nothing.)
    """

    def __init__(self, client):
        self._warp = _WarpEgress(client)

    def resolve(self, model, bucket=0):
        return (PROVIDER, "warp", self._warp)

    def client_for(self, bucket, slot):
        return self._warp.client_for(bucket, slot)


def _make_request(app, body: dict):
    from starlette.requests import Request

    body_bytes = json.dumps(body).encode()

    async def receive():
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "app": app,
    }
    return Request(scope, receive)


def _setup(tmp, **settings_overrides):
    settings = Settings(data_dir=tmp, **settings_overrides)
    registry = ProviderRegistry(data_dir=tmp, settings=settings)
    client = httpx.AsyncClient()
    app = SimpleNamespace(
        state=SimpleNamespace(
            bucket_table=BucketTable(),
            egress=_Egress(client),
            providers=registry,
            usage=None,
            sessions=None,
            dedup=None,
            admin_hub=None,
        )
    )
    body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    return settings, registry, client, _make_request(app, body), body, app


def test_cancelled_json_request_releases_inflight():
    """CancelledError from forward() must not leave in_flight stuck at 1."""

    async def _cancelled_forward(*args, **kwargs):
        raise asyncio.CancelledError()

    with tempfile.TemporaryDirectory() as tmp:
        settings, registry, client, request, _, _ = _setup(tmp)
        orig, pipeline.forward = pipeline.forward, _cancelled_forward
        try:
            try:
                asyncio.run(pipeline.run(request, settings, "chat"))
            except asyncio.CancelledError:
                pass  # client disconnect propagates; accounting must not leak
            else:
                raise AssertionError("expected CancelledError to propagate")
        finally:
            pipeline.forward = orig
            asyncio.run(client.aclose())
        assert registry.runtime(PROVIDER).in_flight == 0, (
            f"in_flight leaked: {registry.runtime(PROVIDER).in_flight}"
        )


def test_forward_exception_releases_inflight():
    """A plain exception from forward() must release the tracked slot."""

    async def _boom_forward(*args, **kwargs):
        raise RuntimeError("upstream blew up")

    with tempfile.TemporaryDirectory() as tmp:
        settings, registry, client, request, _, _ = _setup(tmp)
        orig, pipeline.forward = pipeline.forward, _boom_forward
        try:
            try:
                asyncio.run(pipeline.run(request, settings, "chat"))
            except RuntimeError:
                pass
            else:
                raise AssertionError("expected RuntimeError to propagate")
        finally:
            pipeline.forward = orig
            asyncio.run(client.aclose())
        assert registry.runtime(PROVIDER).in_flight == 0, (
            f"in_flight leaked: {registry.runtime(PROVIDER).in_flight}"
        )


def test_timeout_synthetic_429_releases_inflight():
    """TimeoutError -> dedup inflight 429 must release the tracked slot."""

    async def _slow_forward(*args, **kwargs):
        from fastapi.responses import JSONResponse

        await asyncio.sleep(30)
        return JSONResponse(status_code=200, content={"id": "never"})

    with tempfile.TemporaryDirectory() as tmp:
        settings, registry, client, request, _, app = _setup(tmp, max_timeout_s=0.05)
        app.state.dedup = dedup.DedupTable()
        orig, pipeline.forward = pipeline.forward, _slow_forward
        try:
            response = asyncio.run(pipeline.run(request, settings, "chat"))
        finally:
            pipeline.forward = orig
            asyncio.run(client.aclose())
        assert response.status_code == 429
        assert registry.runtime(PROVIDER).in_flight == 0, (
            f"in_flight leaked: {registry.runtime(PROVIDER).in_flight}"
        )


def test_dedup_replay_hit_releases_inflight():
    """Completed-dedup replay return must release the tracked slot."""

    async def _must_not_run(*args, **kwargs):
        raise AssertionError("replay hit must not reach the upstream leg")

    with tempfile.TemporaryDirectory() as tmp:
        settings, registry, client, request, body, app = _setup(tmp)
        table = dedup.DedupTable()
        app.state.dedup = table
        key = dedup.request_hash(None, None, "chat", MODEL, body, stable_session_id(""))
        table.complete(key, 200, json.dumps({"id": "held"}).encode(), {})
        orig, pipeline.forward = pipeline.forward, _must_not_run
        try:
            response = asyncio.run(pipeline.run(request, settings, "chat"))
        finally:
            pipeline.forward = orig
            asyncio.run(client.aclose())
        assert response.status_code == 200
        assert response.headers.get(dedup.DEDUP_HEADER) == "hit"
        assert registry.runtime(PROVIDER).in_flight == 0, (
            f"in_flight leaked: {registry.runtime(PROVIDER).in_flight}"
        )


def test_dedup_inflight_waiter_releases_inflight():
    """A live (not-done) dedup entry returns 429 and releases the slot."""

    async def _must_not_run(*args, **kwargs):
        raise AssertionError("inflight waiter must not reach the upstream leg")

    with tempfile.TemporaryDirectory() as tmp:
        settings, registry, client, request, body, app = _setup(tmp)
        table = dedup.DedupTable()
        app.state.dedup = table
        key = dedup.request_hash(None, None, "chat", MODEL, body, stable_session_id(""))

        async def _plant():
            fut = asyncio.get_event_loop().create_future()
            table.track(key, fut)
            return fut

        loop = asyncio.new_event_loop()
        try:
            holder = loop.run_until_complete(_plant())
            orig, pipeline.forward = pipeline.forward, _must_not_run
            try:
                response = loop.run_until_complete(
                    pipeline.run(request, settings, "chat")
                )
            finally:
                pipeline.forward = orig
                loop.run_until_complete(client.aclose())
                if not holder.done():
                    holder.cancel()
                loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()
        assert response.status_code == 429
        assert registry.runtime(PROVIDER).in_flight == 0, (
            f"in_flight leaked: {registry.runtime(PROVIDER).in_flight}"
        )
