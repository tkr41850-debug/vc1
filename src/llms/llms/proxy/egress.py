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
