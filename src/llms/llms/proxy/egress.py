from __future__ import annotations

from typing import Protocol

import httpx


class EgressProvider(Protocol):
    def num_slots(self) -> int: ...

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient: ...

    async def aclose(self) -> None: ...


class DirectEgress:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    def num_slots(self) -> int:
        return 1

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient:
        return self._client

    async def aclose(self) -> None:
        await self._client.aclose()


class WarpPoolEgress:
    """Route Zen requests through a vsp warp pool via its /fetch relay.

    The pool picks the warp exit internally (its active/healthy instance);
    num_slots mirrors the pool's ready-exit count (refreshed by the registry
    health poll), starting at 0 before the first poll. Zero slots means no
    ready exits: resolve() skips such pools so traffic fails open to direct.
    """

    def __init__(
        self,
        pool_base_url: str,
        token: str = "",
        num_slots: int = 0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.pool_base_url = pool_base_url.rstrip("/")
        self.token = token
        self._num_slots = max(0, num_slots)
        self._client = client
        self._owned = client is None

    def num_slots(self) -> int:
        return self._num_slots

    def set_num_slots(self, n: int) -> None:
        self._num_slots = max(0, int(n))

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient:
        # The httpx client targets the pool; relay wrapping happens in
        # forward.py via fetch_spec()/parse_fetch_result(). Kept on the
        # protocol so BucketTable slot math stays uniform across providers.
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.pool_base_url, timeout=120.0)
            self._owned = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None
            self._owned = False


class ProviderEgress:
    """Composite egress: model -> provider (noproxy direct or warp relay).

    Holds one WarpPoolEgress per warp provider plus the direct client for
    noproxy. Pipeline asks for (model, bucket, slot) and gets the right
    client plus relay metadata.
    """

    def __init__(self, direct: DirectEgress, registry=None) -> None:
        self._direct = direct
        self._registry = registry
        self._warp: dict[str, WarpPoolEgress] = {}

    def num_slots(self) -> int:
        return self._direct.num_slots()

    def resolve(self, model: str):
        """(provider_id|None, kind, WarpPoolEgress|None) for a model.

        Warp providers take precedence over noproxy when both serve the
        model — noproxy is the default fallback, not the first match.
        A warp pool with known-zero ready exits is skipped so traffic
        fails open to direct until exits come up; unknown health is
        treated as eligible so the first request triggers a poll.
        """
        registry = self._registry
        if registry is None:
            return None, "noproxy", None
        providers = registry.load()
        for p in providers:
            if not p.enabled or p.kind != "warp" or not p.serves(model):
                continue
            egress = self._warp.get(p.id)
            if egress is None:
                egress = WarpPoolEgress(p.base_url, p.token)
                saved = registry.ready_exits(p.id)
                if saved is not None:
                    egress.set_num_slots(saved)
                self._warp[p.id] = egress
            registry._egresses = self._warp
            if registry.runtime(p.id).health.fetched_at > 0 and egress.num_slots() == 0:
                continue
            return p.id, "warp", egress
        for p in providers:
            if p.enabled and p.serves(model):
                return p.id, "noproxy", None
        return None, "noproxy", None

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient:
        return self._direct.client_for(bucket, slot)

    def sync_bucket_slots(self, table) -> bool:
        """Resize the bucket table to the live warp pool spread.

        Slots track the largest ready-exit count across enabled warp
        providers (direct-only deploys stay at 1; pools with no ready
        exits contribute 0). Resizing reshuffles bucket placement;
        callers already holding a slot keep it for their in-flight
        request.
        """
        num_slots = 1
        registry = self._registry
        if registry is not None:
            try:
                providers = registry.load()
            except Exception:
                providers = []
            for p in providers:
                if not p.enabled or p.kind != "warp":
                    continue
                egress = self._warp.get(p.id)
                if egress is not None:
                    slots = egress.num_slots()
                else:
                    saved = registry.ready_exits(p.id)
                    slots = saved if saved is not None else 0
                num_slots = max(num_slots, slots)
        return table.set_num_slots(num_slots)

    async def aclose(self) -> None:
        await self._direct.aclose()
        for egress in self._warp.values():
            await egress.aclose()
        self._warp.clear()
