from __future__ import annotations

import asyncio
import threading

from starlette.requests import Request


class AdminHub:
    """asyncio.Queue fan-out for admin collection changes (keys/models/providers).

    CRUD writes in the admin routes publish a notification per topic; the
    /api/admin/*/sse SSE endpoints re-snapshot their collection on each
    notification. Mirrors ProviderRuntime's subscriber fan-out (asyncio only,
    no threads, slow readers are dropped instead of blocking writers).

    The lock is a threading lock (all critical sections are non-blocking),
    so publish/subscribe are safe across event loops — the data plane
    publishes from the app loop while tests subscribe from their own.
    """

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    async def subscribe(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._lock:
            self._subs.setdefault(topic, set()).add(q)
        return q

    async def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs.get(topic, set()).discard(q)

    def publish_nowait(self, topic: str) -> None:
        with self._lock:
            subs = list(self._subs.get(topic, ()))
            dead = []
            for q in subs:
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subs.get(topic, set()).discard(q)

    async def publish(self, topic: str) -> None:
        self.publish_nowait(topic)


def get_hub(request: Request) -> AdminHub:
    """The app-scoped hub (created in create_app; lazily as fallback)."""
    hub = getattr(request.app.state, "admin_hub", None)
    if hub is None:
        hub = AdminHub()
        request.app.state.admin_hub = hub
    return hub
