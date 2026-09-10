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
    def __init__(self, pool_base_url: str, token: str = "", num_slots: int = 8) -> None:
        self.pool_base_url = pool_base_url.rstrip("/")
        self.token = token
        self._num_slots = num_slots

    def num_slots(self) -> int:
        return self._num_slots

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient:
        raise NotImplementedError(
            "warp pool egress not implemented: vsp needs a per-bucket warp selection API"
        )

    async def aclose(self) -> None:
        return None
