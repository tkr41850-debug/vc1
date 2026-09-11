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
    health poll) so bucket slot math spreads across real exits.
    """

    def __init__(
        self,
        pool_base_url: str,
        token: str = "",
        num_slots: int = 8,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.pool_base_url = pool_base_url.rstrip("/")
        self.token = token
        self._num_slots = max(1, num_slots)
        self._client = client
        self._owned = client is None

    def num_slots(self) -> int:
        return self._num_slots

    def set_num_slots(self, n: int) -> None:
        self._num_slots = max(1, int(n))

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
        """
        registry = self._registry
        if registry is None:
            return None, "noproxy", None
        providers = registry.load()
        provider = None
        for p in providers:
            if p.enabled and p.kind == "warp" and p.serves(model):
                provider = p
                break
        if provider is None:
            for p in providers:
                if p.enabled and p.serves(model):
                    return p.id, "noproxy", None
            return None, "noproxy", None
        egress = self._warp.get(provider.id)
        if egress is None:
            egress = WarpPoolEgress(provider.base_url, provider.token)
            self._warp[provider.id] = egress
        registry._egresses = self._warp
        return provider.id, "warp", egress

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient:
        return self._direct.client_for(bucket, slot)

    async def aclose(self) -> None:
        await self._direct.aclose()
        for egress in self._warp.values():
            await egress.aclose()
        self._warp.clear()
