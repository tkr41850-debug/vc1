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
    ("/api/admin/keys/sse", "/api/admin/keys", "keys"),
    ("/api/admin/models/sse", "/api/admin/models", "models"),
    ("/api/admin/providers/sse", "/api/admin/providers", "providers"),
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
        async with c.stream("GET", "/api/admin/keys/sse") as r:
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
    async with _authed(base) as c, c.stream("GET", "/api/admin/models/sse") as r:
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
    async with _authed(base) as c, c.stream("GET", "/api/admin/providers/sse") as r:
        assert r.status_code == 200
        it = r.aiter_text()
        buf = ""

        async def _until(pred, what):
            # Frames also arrive from the connect-time refresh pass, so
            # scan (skipping pings) instead of asserting on the next frame.
            nonlocal buf
            for _ in range(30):
                frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
                if frame == ": ping":
                    continue
                assert frame.startswith("data: ")
                if pred(_data(frame)["providers"]):
                    return
            raise AssertionError(f"timed out waiting for providers {what}")

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
        await _until(lambda ps: "sse-stream-np" in [p["id"] for p in ps], "create")

        await c.delete("/api/admin/providers/sse-stream-np")
        await _until(lambda ps: "sse-stream-np" not in [p["id"] for p in ps], "delete")


async def test_providers_stream_pushes_reconnect(live_server, monkeypatch):
    """POST .../reconnect publishes, so SSE clients see fresh health/retry
    state without refetching (would hang without the publish)."""

    class _FakePool:
        async def reconnect(self):
            return {"ok": True}

        async def refresh_statuses(self):
            return None

        def snapshot(self):
            return {"error": "", "exits": []}

    base, app = live_server
    registry = app.state.providers

    async def _ensure_pool(provider):
        return _FakePool()

    monkeypatch.setattr(registry, "ensure_pool", _ensure_pool)
    async with _authed(base) as c:
        rc = await c.post(
            "/api/admin/providers",
            json={
                "id": "sse-stream-reconnect",
                "label": "R",
                "kind": "warp",
                "models": [],
                "enabled": True,
                "exits": 1,
            },
        )
        assert rc.status_code == 201
        try:
            async with c.stream("GET", "/api/admin/providers/sse") as r:
                assert r.status_code == 200
                it = r.aiter_text()
                buf = ""
                frame, buf = await _read_frame(it, buf)
                assert "sse-stream-reconnect" in [
                    p["id"] for p in _data(frame)["providers"]
                ]
                rr = await c.post("/api/admin/providers/sse-stream-reconnect/reconnect")
                assert rr.status_code == 200
                frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
                assert "sse-stream-reconnect" in [
                    p["id"] for p in _data(frame)["providers"]
                ]
        finally:
            await c.delete("/api/admin/providers/sse-stream-reconnect")


async def test_sse_ids_do_not_shadow_collection_streams(live_server):
    """A model/provider literally id'd 'sse' must not capture the
    collection routes (same-method overlap would serve JSON, not SSE)."""
    base, _ = live_server
    async with _authed(base) as c:
        rm = await c.post(
            "/api/admin/models", json={"id": "sse", "label": "S", "enabled": True}
        )
        assert rm.status_code == 201
        rp = await c.post(
            "/api/admin/providers",
            json={
                "id": "sse",
                "label": "S",
                "kind": "warp",
                "models": [],
                "enabled": True,
                "exits": 1,
            },
        )
        assert rp.status_code == 201
        try:
            async with c.stream("GET", "/api/admin/models/sse") as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                frame, _ = await _read_frame(r.aiter_text(), "")
                assert "sse" in [m["id"] for m in _data(frame)["models"]]
            async with c.stream("GET", "/api/admin/providers/sse") as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                frame, _ = await _read_frame(r.aiter_text(), "")
                assert "sse" in [p["id"] for p in _data(frame)["providers"]]
            # Item routes for id 'sse' still work alongside the streams.
            ru = await c.put("/api/admin/models/sse", json={"label": "renamed"})
            assert ru.status_code == 200
        finally:
            await c.delete("/api/admin/models/sse")
            await c.delete("/api/admin/providers/sse")


async def test_keys_stream_pushes_out_of_band_file_edits(live_server, monkeypatch):
    """Edits bypassing the CRUD routes (keygen CLI, hand-edited YAML) still
    push: the heartbeat re-stats keys.yaml and snapshots on change."""
    from llms.proxy.routes import admin_streams
    from llms.proxy.store import ApiKey, Store

    monkeypatch.setattr(admin_streams, "HEARTBEAT_S", 0.2)
    base, app = live_server
    from pathlib import Path

    store = Store(data_dir=Path(app.state.settings.data_dir))
    async with _authed(base) as c, c.stream("GET", "/api/admin/keys/sse") as r:
        assert r.status_code == 200
        it = r.aiter_text()
        buf = ""
        frame, buf = await _read_frame(it, buf)
        assert "sk-oob-edit" not in [k["key"] for k in _data(frame)["keys"]]
        keys = store.load_keys()
        keys.append(ApiKey(key="sk-oob-edit", label="oob"))
        store.save_keys(keys)
        try:
            seen = False
            for _ in range(20):
                frame, buf = await asyncio.wait_for(_read_frame(it, buf), 10)
                if frame == ": ping":
                    continue
                assert frame.startswith("data: ")
                if "sk-oob-edit" in [k["key"] for k in _data(frame)["keys"]]:
                    seen = True
                    break
            assert seen
        finally:
            store.save_keys([k for k in store.load_keys() if k.key != "sk-oob-edit"])


async def test_providers_stream_sends_known_first_frame(live_server):
    """First frame is what is known (never waits on refresh)."""
    base, _ = live_server
    async with _authed(base) as c:
        rc = await c.post(
            "/api/admin/providers",
            json={
                "id": "sse-stream-refresh",
                "label": "R",
                "kind": "noproxy",
                "models": [],
                "enabled": True,
                "exits": 1,
            },
        )
        assert rc.status_code == 201
        try:
            async with c.stream("GET", "/api/admin/providers/sse") as r:
                assert r.status_code == 200
                it = r.aiter_text()
                buf = ""
                frame1, buf = await _read_frame(it, buf)
                d1 = _data(frame1)
                assert "sse-stream-refresh" in [p["id"] for p in d1["providers"]]
                # Push-only: no timer refresh pass follows on its own —
                # the next frame arrives only on a CRUD/health/429 push or
                # the heartbeat ping (HEARTBEAT_S cadence).
                frame2, buf = await asyncio.wait_for(_read_frame(it, buf), 20)
                assert frame2.startswith("data: ") or frame2.startswith(": ping")
        finally:
            await c.delete("/api/admin/providers/sse-stream-refresh")


async def test_stream_heartbeats_are_ping_comments(live_server, monkeypatch):
    from llms.proxy.routes import admin_streams

    # Same process serves the stream (thread), so patching the cadence here
    # applies there too: no CRUD, just wait for a heartbeat comment.
    monkeypatch.setattr(admin_streams, "HEARTBEAT_S", 0.2)
    base, _ = live_server
    async with _authed(base) as c, c.stream("GET", "/api/admin/models/sse") as r:
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


async def test_streams_live_beside_spa_routes(live_server):
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
        # Logged out: SPA pages bounce to login, streams 401 like admin API
        # (they live under /api/admin/*, so no gate special-casing).
        r = await c.get("/ui/keys")
        assert r.status_code == 302
        r = await c.get("/api/admin/keys/sse")
        assert r.status_code == 401
        assert r.json() == {"error": {"message": "admin login required"}}


def test_heartbeat_cadences_under_half_cf_budget():
    """All SSE heartbeat cadences stay under 60s (half the CF 120s budget).

    Worst case single miss still delivers at 2x cadence, well under 120s.
    """
    from llms.proxy import forward
    from llms.proxy.routes import admin_streams

    assert admin_streams.HEARTBEAT_S < 60
    assert forward.STREAM_HEARTBEAT_S < 60
    assert admin_streams.HEARTBEAT_S == 15.0
    assert forward.STREAM_HEARTBEAT_S == 30.0
