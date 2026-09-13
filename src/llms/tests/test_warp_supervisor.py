from __future__ import annotations

"""Hermetic tests for the in-process warp supervisor (llms.proxy.warp).

All warp-cli/daemon interaction is faked — no binaries, no sockets, no
network. Live behavior is covered by scripts/warp_probe.py.
"""

import asyncio

import pytest

from llms.proxy import warp as warp_mod
from llms.proxy.warp import (
    WarpPool,
    WarpPoolConfig,
    WarpSlot,
    WarpSupervisor,
    logs_dir_for,
    parse_status_output,
    runtime_dir_for,
    state_dir_for,
)


def _pool(provider_id: str = "pool1", exits: int = 3, **overrides) -> WarpPool:
    cfg = WarpPoolConfig(exits=exits, **overrides)
    return WarpPool(provider_id, "/tmp/nope-data", cfg)


def test_parse_status_connected():
    status, reason = parse_status_output("Status update: Connected\nFoo: bar\n")
    assert status == "Connected"
    assert reason == ""


def test_parse_status_disconnected_with_reason():
    out = "Status update: Disconnected\nReason: no network\n"
    status, reason = parse_status_output(out)
    assert status == "Disconnected"
    assert reason == "no network"


def test_connected_ready_rejects_interim_states():
    """The ready matcher needs the Connected line, not a substring.

    Live `warp-cli status` prints interim states ("Connecting ...",
    "Unable ... Registration Missing") whose text contains "connect"; a
    substring check flips ready minutes before the tunnel is up.
    """
    pool = WarpPool("pool1", "/tmp/nope-data", WarpPoolConfig(exits=1))
    assert pool._connected_ready("Status update: Connected\nNetwork: healthy\n")
    assert not pool._connected_ready(
        "Status update: Connecting\nReason: Establishing connection\n"
    )
    assert not pool._connected_ready(
        "Status update: Unable\nReason: Registration Missing\n"
    )
    assert not pool._connected_ready("Status update: Disconnected\n")
    asyncio.run(pool.aclose())


def test_parse_status_falls_back_to_first_line():
    status, _ = parse_status_output("warp-cli not installed\n")
    assert status == "warp-cli not installed"
    assert parse_status_output("") == ("unknown", "")


def test_state_dir_layout():
    assert str(state_dir_for("/data", "pool1", 2)) == "/data/warps/pool1/warp2"
    # Namespaced so supervised daemons never collide with a host warp-svc.
    assert str(runtime_dir_for("pool1", 2)) == "/run/llms-warp-pool1-2"
    assert str(logs_dir_for("pool1", 2)) == "/var/log/llms-warp-pool1-2"


def test_stale_socket_removed_live_socket_trusted(monkeypatch, tmp_path):
    """ensure_daemon probes an orphan socket instead of trusting it.

    A socket with no live tracked daemon is an orphan from a previous gateway
    run: a dead daemon's socket must be removed before spawning, while a live
    foreign daemon's socket must be trusted (no dir churn, no re-spawn).
    """
    pool = _pool()
    slot = WarpSlot(idx=0, socks_port=40001)
    sock = warp_mod.runtime_dir_for("pool1", 0) / "warp_service"
    monkeypatch.setattr(warp_mod.Path, "exists", lambda self: self == sock)
    seen: list = []

    async def go():
        monkeypatch.setattr(
            pool, "_socket_live_sync", lambda s: seen.append(s.idx) or False
        )
        assert await pool.ensure_daemon(slot) is False  # stale, can't unlink real /run
        assert seen == [0]

    asyncio.run(go())

    pool2 = _pool()
    monkeypatch.setattr(pool2, "_socket_live_sync", lambda s: True)
    # Live foreign daemon: early True, no spawn bookkeeping.
    got = asyncio.run(pool2.ensure_daemon(WarpSlot(idx=0, socks_port=40001)))
    assert got is True
    assert pool2._daemons == {}
    asyncio.run(pool2.aclose())


def test_unlink_stale_socket_removes(tmp_path):
    pool = _pool()
    sock = tmp_path / "warp_service"
    sock.write_text("stale")
    assert pool._unlink_stale_socket(sock) is True
    assert not sock.exists()
    assert pool._unlink_stale_socket(sock) is True  # missing_ok path


def test_no_binary_stays_unhealthy_no_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(warp_mod, "warp_cli_available", lambda: False)
    pool = WarpPool("pool1", tmp_path, WarpPoolConfig(exits=2))
    asyncio.run(pool.start())
    try:
        assert pool.binary_error.startswith("warp-cli not installed")
        assert pool.snapshot()["exits"] == []
        asyncio.run(pool.aclose())
        assert asyncio.run(pool.reconnect())["ok"] is False
    finally:
        asyncio.run(pool.aclose())


def test_reconnect_bounces_exits_and_repols(monkeypatch):
    pool = _pool()
    pool.instances = [
        WarpSlot(idx=0, socks_port=40001, ready=True),
        WarpSlot(idx=1, socks_port=40002, ready=True),
        WarpSlot(idx=2, socks_port=40003, ready=False),
    ]
    calls: list = []

    async def fake_cli(slot, *args, timeout=20):
        calls.append((slot.idx, args))
        if args[:1] == ("status",):
            return 0, "Status update: Connected\nNetwork: healthy"
        if args[:2] == ("registration", "show"):
            return 0, "Id: abc"
        return 0, "ok"

    pool._cli = fake_cli  # type: ignore[method-assign]
    for inst in pool.instances:
        (warp_mod.state_dir_for("/tmp/nope-data", "pool1", inst.idx)).mkdir(
            parents=True, exist_ok=True
        )
    ran: list = []
    monkeypatch.setattr(
        warp_mod, "run_cli", lambda *a, **k: ran.append(a[3:]) or (0, "ok")
    )
    result = asyncio.run(pool.reconnect())
    assert result == {
        "ok": True,
        "before": {"ready": 2, "exits": 3},
        "after": {"ready": 3, "exits": 3},
    }
    # Every exit got a bounce disconnect (module-level run_cli) plus the
    # full bring-up (mode + port + connect via pool._cli).
    assert ("disconnect",) in ran
    kinds = [a for _, a in calls]
    assert ("connect",) in kinds
    asyncio.run(pool.aclose())


def test_slow_connect_watcher_marks_ready_late(monkeypatch, tmp_path):
    """A slot that connects after the burst window still flips ready.

    Regression for the live finding where registration + proxy mode applied
    in seconds but the MASQUE tunnel needed ~70s to reach Connected — the
    burst poll (45s) gave up while the daemon was still handshaking.
    """

    async def go():
        pool = WarpPool("pool1", tmp_path, WarpPoolConfig(exits=1))
        slot = WarpSlot(idx=0, socks_port=40001)
        pool.instances.append(slot)
        state = {"phase": "burst"}

        async def fake_cli(s, *args, timeout=20):
            if args[:1] == ("status",):
                # Burst phase: still handshaking; watcher phase: connected.
                if state["phase"] == "burst":
                    return 0, "Status update: Connecting"
                return 0, "Status update: Connected"
            if args[:2] == ("registration", "show"):
                return 0, "Id: abc"
            return 0, "ok"

        pool._cli = fake_cli  # type: ignore[method-assign]
        monkeypatch.setattr(
            pool, "ensure_daemon", lambda s: asyncio.sleep(0, result=True)
        )
        monkeypatch.setattr(pool, "ensure_proxy_mode", lambda s: asyncio.sleep(0))
        monkeypatch.setattr(
            pool, "register_one", lambda s, timeout=60: asyncio.sleep(0, result=True)
        )
        await pool.start()
        try:
            burst = await pool._bring_up(slot, timeout=0.01)
            assert burst is False
            assert slot.ready is False
            # Drive the watcher inline (it would normally run as a task).
            # The watcher only tracks registered slots (registration just
            # succeeded in the burst), so seed reg.json first.
            reg = tmp_path / "warps" / "pool1" / "warp0"
            reg.mkdir(parents=True, exist_ok=True)
            (reg / "reg.json").write_text("{}")
            state["phase"] = "watch"
            await pool._watch_slot(slot, timeout=30)
            assert slot.ready is True
        finally:
            await pool.aclose()

    asyncio.run(go())


def test_refresh_promotes_connected_registered_slot(tmp_path):
    """A registered slot that connected late flips ready on status refresh."""

    async def go():
        pool = WarpPool("pool1", tmp_path, WarpPoolConfig(exits=1))
        slot = WarpSlot(idx=0, socks_port=40001)
        pool.instances.append(slot)
        reg = tmp_path / "warps" / "pool1" / "warp0"
        reg.mkdir(parents=True, exist_ok=True)
        (reg / "reg.json").write_text("{}")

        async def fake_cli(s, *args, timeout=20):
            if args[:1] == ("status",):
                return 0, "Status update: Connected"
            return 0, "ok"

        pool._cli = fake_cli  # type: ignore[method-assign]
        await pool.refresh_statuses()
        assert slot.ready is True
        assert pool.ready_event.is_set()
        await pool.aclose()

    asyncio.run(go())


def test_refresh_does_not_promote_unregistered_slot(tmp_path):
    """An unregistered slot reporting connected stays unready."""

    async def go():
        pool = WarpPool("pool1", tmp_path, WarpPoolConfig(exits=1))
        slot = WarpSlot(idx=0, socks_port=40001)
        pool.instances.append(slot)

        async def fake_cli(s, *args, timeout=20):
            return 0, "Status update: Connected"

        pool._cli = fake_cli  # type: ignore[method-assign]
        await pool.refresh_statuses()
        assert slot.ready is False
        await pool.aclose()

    asyncio.run(go())


def test_reconnect_empty_pool_not_ok():
    pool = _pool()
    result = asyncio.run(pool.reconnect())
    assert result["ok"] is False
    assert result["before"] == {"ready": 0, "exits": 0}
    asyncio.run(pool.aclose())


def test_wait_ready_timeout_returns_none():
    pool = _pool()
    pool.instances = [WarpSlot(idx=0, socks_port=40001, ready=False)]
    assert asyncio.run(pool.wait_ready(timeout=0.01)) is None
    asyncio.run(pool.aclose())


def test_wait_ready_returns_first_healthy():
    async def go() -> WarpSlot | None:
        pool = _pool()
        pool.instances = [
            WarpSlot(idx=0, socks_port=40001, ready=True),
            WarpSlot(idx=1, socks_port=40002, ready=True),
        ]
        pool.ready_event.set()
        got = await pool.wait_ready(timeout=1.0)
        await pool.aclose()
        return got

    got = asyncio.run(go())
    assert got is not None and got.idx == 0


def test_heal_guarded_by_cooldown_and_budget():
    pool = _pool()
    slot = WarpSlot(idx=0, socks_port=40001, ready=False, last_heal=1e9)
    calls: list = []

    async def boom():
        calls.append(True)
        return 0

    pool._budget_wait = lambda: 5.0  # type: ignore[method-assign]
    assert asyncio.run(pool.heal_stale(slot)) is False
    assert calls == []


def test_heal_stale_deletes_and_registers(monkeypatch, tmp_path):
    pool = WarpPool("pool1", tmp_path, WarpPoolConfig(exits=2))
    slot = WarpSlot(idx=0, socks_port=40001, ready=False)
    pool.last_reg_ts = 0.0  # budget spent long ago → budget_wait() == 0
    ran: list = []
    monkeypatch.setattr(
        warp_mod,
        "run_cli",
        lambda *a, **k: ran.append(a) or (0, "ok"),
    )
    assert asyncio.run(pool.heal_stale(slot)) is True
    assert [c[3:] for c in ran] == [
        ("registration", "delete"),
        ("--accept-tos", "registration", "new"),
    ]
    asyncio.run(pool.aclose())


def test_supervisor_ensure_returns_same_pool(monkeypatch, tmp_path):
    monkeypatch.setattr(warp_mod, "warp_cli_available", lambda: False)

    async def go():
        sup = WarpSupervisor(tmp_path)
        a = await sup.ensure("p1")
        b = await sup.ensure("p1")
        assert a is b
        assert sup.get("p1") is a
        assert sup.get("missing") is None
        await sup.drop("p1")
        assert sup.get("p1") is None
        await sup.aclose()

    asyncio.run(go())


def test_poll_ratelimit_sets_error_not_ready():
    pool = _pool()

    async def go():
        slot = WarpSlot(idx=0, socks_port=40001)

        async def fake_cli(s, *args, timeout=20):
            return 0, "Status update: Unable\nReason: too many requests (429)"

        pool._cli = fake_cli  # type: ignore[method-assign]
        ok = await pool.poll_until_connected(slot, timeout=5)
        assert ok is False
        assert slot.last_error.startswith("ratelimited:")
        await pool.aclose()

    asyncio.run(go())


def test_ensure_proxy_mode_sends_explicit_connect():
    """Proxy-mode setup must end with an explicit `connect`.

    The daemon sits in Disconnected(Manual) after a fresh registration;
    SetMode/SetWarpProxyPort alone do not start the handshake — only a
    connect request (or an unrelated always-on flip) does.
    """
    pool = _pool()
    slot = WarpSlot(idx=0, socks_port=40001)
    seen: list = []

    async def fake_cli(s, *args, timeout=20):
        seen.append(args)
        return 0, "ok"

    pool._cli = fake_cli  # type: ignore[method-assign]
    asyncio.run(pool.ensure_proxy_mode(slot))
    kinds = [a for a in seen]
    assert ("mode", "proxy") in kinds
    assert ("proxy", "port", "40001") in kinds
    assert kinds[-1] == ("connect",)
    asyncio.run(pool.aclose())


def test_egress_spreads_slots_over_ready_ports(tmp_path):
    from llms.proxy.egress import WarpSocksEgress

    egress = WarpSocksEgress("pool1", num_slots=2, socks_ports=(40001, 40002))
    assert egress.pick_port(0) == 40001
    assert egress.pick_port(1) == 40002
    assert egress.pick_port(2) == 40001
    c0 = egress.client_for(0, 0)
    assert egress.client_for(7, 0) is c0  # cached per port
    assert egress.client_for(0, 1) is not c0
    asyncio.run(egress.aclose())
    empty = WarpSocksEgress("pool1")
    with pytest.raises(RuntimeError):
        empty.client_for(0, 0)
