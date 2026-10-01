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
        # Throttled push (§3): count-only in-flight updates coalesce.
        # First call per quiet window publishes immediately; the rest
        # schedule one trailing publish ~2s out. CRUD/health/retry
        # transitions use publish() and always bypass the throttle.
        self._throttle_pending: dict[str, bool] = {}
        self._throttle_scheduled: dict[str, bool] = {}

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

    async def publish_throttled(self, topic: str, delay_s: float = 2.0) -> None:
        """Coalesced publish for high-frequency count-only updates.

        First call in a quiet window publishes immediately; calls inside
        the window schedule exactly one trailing publish. Never raises.
        """
        immediate = False
        with self._lock:
            if not self._throttle_pending.get(topic, False):
                self._throttle_pending[topic] = True
                immediate = True
            elif not self._throttle_scheduled.get(topic, False):
                self._throttle_scheduled[topic] = True
            else:
                return
        if immediate:
            self.publish_nowait(topic)
            return
        try:
            await asyncio.sleep(delay_s)
        except Exception:
            pass
        with self._lock:
            self._throttle_scheduled[topic] = False
            self._throttle_pending[topic] = False
        self.publish_nowait(topic)


def get_hub(request: Request) -> AdminHub:
    """The app-scoped hub (created in create_app; lazily as fallback)."""
    hub = getattr(request.app.state, "admin_hub", None)
    if hub is None:
        hub = AdminHub()
        request.app.state.admin_hub = hub
    return hub
