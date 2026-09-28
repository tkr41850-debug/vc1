from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time

import httpx
import itsdangerous
import pytest
import uvicorn

from tests.conftest import make_settings

STREAMS = (
    ("/ui/keys/stream", "/api/admin/keys", "keys"),
    ("/ui/models/stream", "/api/admin/models", "models"),
    ("/ui/providers/stream", "/api/admin/providers", "providers"),
)


def _session_cookie(secret: str = "test-secret") -> str:
    signer = itsdangerous.TimestampSigner(secret)
    raw = base64.b64encode(json.dumps({"admin_user": "tester"}).encode())
    return signer.sign(raw).decode()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server(tmp_path_factory):
    """Real uvicorn over a socket: TestClient buffers whole SSE bodies, so
    infinite streams can only be tested against a live server. Never read a
    stream to EOF — take 1-2 frames, then exit the context (which closes)."""
    from llms.proxy.keys import reset_cache
    from llms.proxy.main import create_app
    from llms.proxy.store import ApiKey, Store

    data_dir = tmp_path_factory.mktemp("sse-data")
    static_dir = tmp_path_factory.mktemp("sse-static")
    (static_dir / "index.html").write_text("<html>sse-spa</html>")
    reset_cache()
    Store(data_dir=data_dir).save_keys([ApiKey(key="sk-stream-seed", label="seed")])
    settings = make_settings(
        data_dir=str(data_dir),
        static_dir=str(static_dir),
        admin_github_users=("tester",),
        admin_session_secret="test-secret",
    )
    app = create_app(settings)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "uvicorn did not start"
    yield f"http://127.0.0.1:{port}", app
    server.should_exit = True
    thread.join(timeout=20)


def _authed(base: str) -> httpx.AsyncClient:
    client = httpx.AsyncClient(base_url=base, timeout=20.0)
    client.cookies.set("session", _session_cookie())
    return client


async def _read_frame(aiter, buf: str) -> tuple[str, str]:
    """Next complete SSE frame (text up to a blank line)."""
    async for chunk in aiter:
        buf += chunk
        if "\n\n" in buf:
            frame, rest = buf.split("\n\n", 1)
            return frame, rest
    raise AssertionError("stream ended before a full SSE frame arrived")


def _data(frame: str) -> dict:
    assert frame.startswith("data: "), frame[:60]
    return json.loads(frame[len("data: ") :])


@pytest.mark.parametrize(
    "stream_path,rest_path,top_key", STREAMS, ids=["keys", "models", "providers"]
)
async def test_stream_snapshot_matches_rest(
    live_server, stream_path, rest_path, top_key
):
    base, _ = live_server
    async with _authed(base) as c:
        rest = (await c.get(rest_path)).json()
        async with c.stream("GET", stream_path) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            frame, _ = await _read_frame(r.aiter_text(), "")
            assert _data(frame) == rest


async def test_keys_stream_pushes_crud_updates(live_server):
    base, app = live_server
    async with _authed(base) as c:
        async with c.stream("GET", "/ui/keys/stream") as r:
            assert r.status_code == 200
            it = r.aiter_text()
            buf = ""
            frame, buf = await _read_frame(it, buf)
            assert "sk-stream-e2e" not in [k["key"] for k in _data(frame)["keys"]]

            rc = await c.post(
                "/api/admin/keys",
                json={"key": "sk-stream-e2e", "label": "e2e", "enabled": True},
            )
            assert rc.status_code == 201
            frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
            assert "sk-stream-e2e" in [k["key"] for k in _data(frame)["keys"]]

            ru = await c.put("/api/admin/keys/sk-stream-e2e", json={"label": "renamed"})
            assert ru.status_code == 200
            frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
            renamed = next(
                k for k in _data(frame)["keys"] if k["key"] == "sk-stream-e2e"
            )
            assert renamed["label"] == "renamed"

            rd = await c.delete("/api/admin/keys/sk-stream-e2e")
            assert rd.status_code == 200
            frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
            assert "sk-stream-e2e" not in [k["key"] for k in _data(frame)["keys"]]
        # Exiting the stream context closes the connection; one more write
        # wakes the generator so it notices the disconnect and unsubscribes.
        await c.post(
            "/api/admin/keys",
            json={"key": "sk-stream-cleanup", "label": "", "enabled": True},
        )
        await c.delete("/api/admin/keys/sk-stream-cleanup")
        hub = app.state.admin_hub
        for _ in range(200):
            if not hub._subs.get("keys"):
                break
            await asyncio.sleep(0.05)
        assert not hub._subs.get("keys")


async def test_models_stream_pushes_crud_updates(live_server):
    base, _ = live_server
    async with _authed(base) as c, c.stream("GET", "/ui/models/stream") as r:
        assert r.status_code == 200
        it = r.aiter_text()
        buf = ""
        frame, buf = await _read_frame(it, buf)
        assert "sse-stream-model" not in [m["id"] for m in _data(frame)["models"]]

        rc = await c.post(
            "/api/admin/models",
            json={"id": "sse-stream-model", "label": "P", "enabled": True},
        )
        assert rc.status_code == 201
        frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
        assert "sse-stream-model" in [m["id"] for m in _data(frame)["models"]]

        await c.delete("/api/admin/models/sse-stream-model")
        frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
        assert "sse-stream-model" not in [m["id"] for m in _data(frame)["models"]]


async def test_providers_stream_pushes_crud_updates(live_server):
    base, _ = live_server
    async with _authed(base) as c, c.stream("GET", "/ui/providers/stream") as r:
        assert r.status_code == 200
        it = r.aiter_text()
        buf = ""
        frame, buf = await _read_frame(it, buf)
        assert "sse-stream-np" not in [p["id"] for p in _data(frame)["providers"]]

        rc = await c.post(
            "/api/admin/providers",
            json={
                "id": "sse-stream-np",
                "label": "L",
                "kind": "noproxy",
                "models": [],
                "enabled": True,
                "exits": 1,
            },
        )
        assert rc.status_code == 201
        frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
        assert "sse-stream-np" in [p["id"] for p in _data(frame)["providers"]]

        await c.delete("/api/admin/providers/sse-stream-np")
        frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
        assert "sse-stream-np" not in [p["id"] for p in _data(frame)["providers"]]


async def test_stream_heartbeats_are_ping_comments(live_server, monkeypatch):
    from llms.proxy.routes import ui_streams

    # Same process serves the stream (thread), so patching the cadence here
    # applies there too: no CRUD, just wait for a heartbeat comment.
    monkeypatch.setattr(ui_streams, "HEARTBEAT_S", 0.2)
    base, _ = live_server
    async with _authed(base) as c, c.stream("GET", "/ui/models/stream") as r:
        assert r.status_code == 200
        it = r.aiter_text()
        buf = ""
        seen_ping = False
        for _ in range(10):
            frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
            if frame == ": ping":
                seen_ping = True
                break
            assert frame.startswith("data: ")
        assert seen_ping


async def test_streams_reject_unauthenticated(live_server):
    base, _ = live_server
    async with httpx.AsyncClient(base_url=base, timeout=10.0) as c:
        for stream_path, _, _ in STREAMS:
            r = await c.get(stream_path)
            assert r.status_code == 401
            assert r.json() == {"error": {"message": "admin login required"}}


async def test_streams_registered_before_spa_fallback(live_server):
    base, _ = live_server
    async with _authed(base) as c:
        for stream_path, _, _ in STREAMS:
            async with c.stream("GET", stream_path) as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
        # The SPA tab routes still serve the bundle.
        r = await c.get("/ui/keys")
        assert r.status_code == 200
        assert "sse-spa" in r.text
    async with httpx.AsyncClient(
        base_url=base, timeout=10.0, follow_redirects=False
    ) as c:
        # Logged out: SPA pages bounce to login, but streams 401 like admin API.
        r = await c.get("/ui/keys")
        assert r.status_code == 302
        r = await c.get("/ui/keys/stream")
        assert r.status_code == 401
