from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from llms.proxy.admin_hub import get_hub
from llms.proxy.auth import require_admin
from llms.proxy.config import Settings, settings_from_app
from llms.proxy.routes.admin import keys_snapshot, models_snapshot
from llms.proxy.routes.providers import _registry, providers_snapshot
from llms.proxy.store import Store

logger = logging.getLogger("zen_proxy")

router = APIRouter()

# Collection live-update streams for the admin UI. They sit next to the
# REST list endpoints they mirror (same snapshot shape as GET
# /api/admin/{keys,models,providers}) so the browser holds one SSE per
# collection instead of polling GETs. Heartbeat cadence mirrors the
# per-provider recent-request stream
# (GET /api/admin/providers/{id}/stream): SSE comments only, never fake
# events, so proxies don't kill idle connections.
HEARTBEAT_S = 15.0


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _mtimes(paths: list[Path]) -> float:
    """Newest mtime across paths (missing files sort as epoch)."""
    latest = 0.0
    for p in paths:
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            pass
    return latest


async def _collection_stream(
    request: Request,
    topic: str,
    snapshot,
    watch: list[Path] | None = None,
    refresh=None,
) -> StreamingResponse:
    """Snapshot-now, refresh, then live updates.

    The first frame is always what is currently known (pure in-memory
    reads — never waits on refresh), so page load never stalls behind a
    slow poll. When ``refresh`` is given (providers: TTL-gated health
    re-poll), it runs after the first frame and a second frame goes out
    only if the payload actually changed.
    """

    async def gen():
        # Subscribe BEFORE the first snapshot: a CRUD publishing in between
        # stays queued and triggers an immediate second snapshot below,
        # so no write in that window is ever lost.
        q = await get_hub(request).subscribe(topic)
        try:
            try:
                last = _frame(snapshot())
            except Exception as exc:
                # Snapshot work must never kill heartbeats: degrade to a
                # ping so the stream survives past the tunnel idle timeout.
                logger.debug("sse %s snapshot failed: %r", topic, exc)
                yield ": ping\n\n"
                last = None
            if last is not None:
                yield last
            if refresh is not None:
                try:
                    # Providers refresh is bounded below the heartbeat
                    # cadence so a hung poll can never silence pings.
                    await asyncio.wait_for(refresh(), timeout=10.0)
                except Exception as exc:
                    logger.debug("sse %s refresh failed: %r", topic, exc)
                else:
                    try:
                        fresh = _frame(snapshot())
                    except Exception as exc:
                        logger.debug("sse %s snapshot failed: %r", topic, exc)
                    else:
                        if fresh != last:
                            last = fresh
                            yield fresh
            # Out-of-band edits (keygen CLI, hand-edited YAML) bypass the
            # CRUD routes and their publishes; the heartbeat re-stats the
            # files and pushes a fresh snapshot when they moved.
            seen = _mtimes(watch or [])
            while True:
                try:
                    if await request.is_disconnected():
                        break
                except asyncio.CancelledError:
                    raise
                except Exception:
                    break
                try:
                    await asyncio.wait_for(q.get(), timeout=HEARTBEAT_S)
                except asyncio.TimeoutError:
                    try:
                        if watch is not None:
                            now = _mtimes(watch)
                            if now > seen:
                                seen = now
                                try:
                                    last = _frame(snapshot())
                                except asyncio.CancelledError:
                                    raise
                                except Exception as exc:
                                    logger.debug(
                                        "sse %s snapshot failed: %r", topic, exc
                                    )
                                    yield ": ping\n\n"
                                    continue
                                yield last
                                continue
                        yield ": ping\n\n"
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        yield b": ping\n\n"
                    continue
                except asyncio.CancelledError:
                    raise
                except BaseException:
                    # Cancellation-adjacent noise degrades to a ping, never
                    # stream death.
                    yield ": ping\n\n"
                    continue
                try:
                    last = _frame(snapshot())
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("sse %s snapshot failed: %r", topic, exc)
                    yield ": ping\n\n"
                    continue
                yield last
        finally:
            await get_hub(request).unsubscribe(topic, q)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/api/admin/keys/sse")
async def keys_stream(
    request: Request,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = Store(data_dir=Path(settings.data_dir))
    watch = [Path(settings.data_dir) / "keys.yaml"]

    def snapshot() -> dict:
        # Usage is live (in-memory counters), so re-read it on every event;
        # closing over the connect-time snapshot would go stale.
        return keys_snapshot(store, request.app.state.usage.snapshot())

    return await _collection_stream(request, "keys", snapshot, watch)


@router.get("/api/admin/models/sse")
async def models_stream(
    request: Request,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = Store(data_dir=Path(settings.data_dir))
    watch = [Path(settings.data_dir) / "models.yaml"]
    return await _collection_stream(
        request, "models", lambda: models_snapshot(store), watch
    )


@router.get("/api/admin/providers/sse")
async def providers_stream(
    request: Request,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    watch = [Path(settings.data_dir) / "providers.yaml"]

    async def _refresh() -> None:
        # Push-only: no timer refresh. Health refreshes are event-driven
        # (enable/boot task, reconnect task, post-cycle, 429, explicit GET
        # health). Kept as a no-op hook so the first-frame staging stays.
        return None

    return await _collection_stream(
        request, "providers", lambda: providers_snapshot(request), watch, _refresh
    )
