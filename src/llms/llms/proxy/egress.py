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


def _socks_client(port: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        proxy=f"socks5://127.0.0.1:{port}",
        timeout=120.0,
        trust_env=False,
    )


class WarpSocksEgress:
    """Route Zen requests through local warp SOCKS exits, direct in-process.

    The in-process supervisor (see llms.proxy.warp) owns warp-svc +
    warp-cli; this class dials the pool's ready SOCKS ports from the index
    spread (slot % ready exits). Zero slots means no ready exits:
    resolve() skips such pools so traffic fails open to direct.
    """

    def __init__(
        self,
        provider_id: str,
        num_slots: int = 0,
        socks_ports: tuple[int, ...] = (),
    ) -> None:
        self.provider_id = provider_id
        self._num_slots = max(0, num_slots)
        self._socks_ports = tuple(socks_ports)
        self._clients: dict[int, httpx.AsyncClient] = {}

    def num_slots(self) -> int:
        return self._num_slots

    def set_num_slots(self, n: int) -> None:
        self._num_slots = max(0, int(n))

    def set_socks_ports(self, ports: list[int]) -> None:
        ports = list(ports)
        if ports != list(self._socks_ports):
            self._socks_ports = tuple(ports)

    def pick_port(self, slot: int) -> int | None:
        if not self._socks_ports:
            return None
        return self._socks_ports[slot % len(self._socks_ports)]

    def client_for(self, bucket: int, slot: int) -> httpx.AsyncClient:
        port = self.pick_port(slot)
        if port is None:
            raise RuntimeError(
                f"warp provider {self.provider_id}: no ready SOCKS exits"
            )
        client = self._clients.get(port)
        if client is None:
            client = _socks_client(port)
            self._clients[port] = client
        return client

    async def aclose(self) -> None:
        for client in self._clients.values():
            try:
                await client.aclose()
            except Exception:
                pass
        self._clients.clear()


def _pool_may_recover(registry, provider) -> bool:
    """True when a zero-ready warp pool should stay eligible for traffic.

    Request-driven health refreshes are what promote a late-connecting slot
    to ready — but resolve() skips pools with known-zero ready exits, so a
    pool whose first poll landed mid-boot (daemon handshaking, proxy mode
    just applied) would pin all traffic to direct until some unrelated
    refresh happens. Skip only when every slot's last cached status is
    disconnected; any other state (connecting, unable, never-polled, or
    connected-with-promotion-pending) keeps the pool in the path so the
    next refresh can promote it. A missing pool (or pool without slot
    state) counts as down.
    """
    pool_for = getattr(registry, "pool_for", None)
    pool = pool_for(provider) if callable(pool_for) else None
    instances = getattr(pool, "instances", None) if pool is not None else None
    if not instances:
        return False
    cache = getattr(pool, "status_cache", None) or {}
    for w in instances:
        entry = cache.get(getattr(w, "idx", None), {})
        status = str(entry.get("status", "")).lower() if isinstance(entry, dict) else ""
        if not status.startswith("disconnected"):
            return True
    return False


class ProviderEgress:
    """Composite egress: model -> provider (noproxy direct or warp SOCKS).

    Holds one WarpSocksEgress per warp provider plus the direct client for
    noproxy. Pipeline asks for (model, bucket, slot) and gets the right
    client plus the ready SOCKS port.
    """

    def __init__(self, direct: DirectEgress, registry=None) -> None:
        self._direct = direct
        self._registry = registry
        self._warp: dict[str, WarpSocksEgress] = {}

    def num_slots(self) -> int:
        return self._direct.num_slots()

    def resolve(self, model: str):
        """(provider_id|None, kind, WarpSocksEgress|None) for a model.

        Warp providers take precedence over noproxy when both serve the
        model — noproxy is the default fallback, not the first match.
        A warp pool with known-zero ready exits is skipped so traffic
        fails open to direct — unless the pool may still come up (see
        _pool_may_recover); unknown health is treated as eligible so the
        first request triggers a poll.
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
                egress = WarpSocksEgress(p.id)
                self._warp[p.id] = egress
            registry._egresses = self._warp
            rt = registry.runtime(p.id)
            ready = [w for w in rt.health.exits if w.ready]
            if (
                rt.health.fetched_at > 0
                and not ready
                and not _pool_may_recover(registry, p)
            ):
                continue
            if ready:
                egress.set_num_slots(len(ready))
                egress.set_socks_ports(sorted(w.socks for w in ready))
            elif rt.health.fetched_at <= 0:
                saved = registry.ready_exits(p.id)
                if saved is not None:
                    egress.set_num_slots(saved)
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
