from __future__ import annotations

import asyncio

from starlette.requests import Request


class AdminHub:
    """asyncio.Queue fan-out for admin collection changes (keys/models/providers).

    CRUD writes in the admin routes publish a notification per topic; the
    /ui/*/stream SSE endpoints re-snapshot their collection on each
    notification. Mirrors ProviderRuntime's subscriber fan-out (asyncio only,
    no threads, slow readers are dropped instead of blocking writers).
    """

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subs.setdefault(topic, set()).add(q)
        return q

    async def unsubscribe(self, topic: str, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subs.get(topic, set()).discard(q)

    async def publish(self, topic: str) -> None:
        async with self._lock:
            subs = list(self._subs.get(topic, ()))
            dead = []
            for q in subs:
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                self._subs.get(topic, set()).discard(q)


def get_hub(request: Request) -> AdminHub:
    """The app-scoped hub (created in create_app; lazily as fallback)."""
    hub = getattr(request.app.state, "admin_hub", None)
    if hub is None:
        hub = AdminHub()
        request.app.state.admin_hub = hub
    return hub
