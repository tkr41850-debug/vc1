from __future__ import annotations

"""In-process warp supervisor: llms owns its warp-cli datadirs directly.

Adapted from the vsp reference implementation (``~/vsp/app.py``) — llms must
not depend on vsp. One :class:`WarpPool` per warp provider supervises
``warp-svc`` daemons and drives ``warp-cli`` (registration, proxy mode,
connect) for that provider's slots. State lives under the gateway DATA_DIR::

    DATA_DIR/warps/<provider_id>/warp<N>/   # warp-cli STATE_DIRECTORY

Runtime sockets and logs are ephemeral (not in DATA_DIR)::

    /run/llms-warp-<provider_id>-<N>/warp_service
    /var/log/llms-warp-<provider_id>-<N>/

The ``llms-warp-`` namespace keeps supervised daemons off the host
warp-svc's ``/run/cloudflare-warp`` socket. The gateway normally runs as
root (container entrypoint, ``just up`` via sudo-adjacent dirs), so
``warp-svc`` launches directly with the full ``warp_env``; as a non-root
fallback the spawn goes through ``sudo -n env VAR=...`` to preserve the
STATE/RUNTIME/LOGS isolation sudo would otherwise scrub.

When no ``warp-cli`` binary is present the pool stays unhealthy with a clear
error and traffic fails open to direct — no exception, no crash.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("llms.warp")

DEFAULT_SLOTS = 8
DEFAULT_HOLD_TIMEOUT_S = 10.0
DEFAULT_REG_INTERVAL_SEC = 28800
DEFAULT_BOOT_RETRY_SEC = 300
DEFAULT_BASE_SOCKS_PORT = 40001
INITIAL_BURST = 1
STALE_FAIL_THRESHOLD = 3
HEAL_COOLDOWN_SEC = 3600
STATUS_CACHE_SEC = 30


def warp_cli_available() -> bool:
    """True when warp-cli can be launched from this process."""
    return shutil.which("warp-cli") is not None


def parse_status_output(out: str) -> tuple[str, str]:
    """Parse ``warp-cli status`` text into (status, reason)."""
    status, reason = "unknown", ""
    for line in out.splitlines():
        low = line.lower()
        if "status update" in low:
            status = (
                line.split(":", 1)[1].strip()[:60] if ":" in line else line.strip()[:60]
            )
        elif low.startswith("network:"):
            # `warp-cli status` reports connectivity on a second line
            # ("Status update: Connected" + "Network: healthy"); keep it as
            # the reason so health snapshots preserve it.
            reason = line.split(":", 1)[1].strip()[:120]
        elif low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()[:120]
    if status == "unknown":
        first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        status = first[:60] if first else "unknown"
    return status, reason


def state_dir_for(data_dir: str | Path, provider_id: str, idx: int) -> Path:
    return Path(data_dir) / "warps" / provider_id / f"warp{idx}"


def runtime_dir_for(provider_id: str, idx: int) -> Path:
    # Namespaced llms-warp-* so a host warp-svc (systemd, /run/cloudflare-warp)
    # never collides with supervised per-slot daemons.
    return Path(f"/run/llms-warp-{provider_id}-{idx}")


def logs_dir_for(provider_id: str, idx: int) -> Path:
    return Path(f"/var/log/llms-warp-{provider_id}-{idx}")


def _open_log(
    provider_id: str, idx: int, data_dir: str | Path
) -> tuple[object | None, str]:
    """Blocking daemon-log open for ensure_daemon (runs in an executor)."""
    candidates = (
        logs_dir_for(provider_id, idx) / "svc.stdout.log",
        state_dir_for(data_dir, provider_id, idx) / "svc.stdout.log",
    )
    last_err = ""
    for path in candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(path, "ab")  # noqa: SIM115
            return handle, ""
        except OSError as exc:
            last_err = str(exc)
    return None, last_err


def warp_env(provider_id: str, idx: int, data_dir: str | Path) -> dict:
    """Environment isolating one warp-cli slot (state + runtime + logs)."""
    env = dict(os.environ)
    env["STATE_DIRECTORY"] = str(state_dir_for(data_dir, provider_id, idx))
    env["RUNTIME_DIRECTORY"] = str(runtime_dir_for(provider_id, idx))
    env["LOGS_DIRECTORY"] = str(logs_dir_for(provider_id, idx))
    return env


def run_cli(
    provider_id: str, idx: int, data_dir: str | Path, *args: str, timeout: int = 20
) -> tuple[int, str]:
    """Run warp-cli synchronously for one slot; call from an executor thread.

    ``--accept-tos`` is prepended only for subcommands that accept it.
    Zero-arg actions (``connect``/``disconnect``/``status``/``--version``)
    reject any trailing flag, so they are passed through untouched.
    """
    if args[:1] not in (("--accept-tos",), ("connect",), ("disconnect",), ("status",)):
        args = ("--accept-tos", *args)
    try:
        proc = subprocess.run(  # noqa: PLW1510
            ["warp-cli", *args],
            env=warp_env(provider_id, idx, data_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, (proc.stdout + proc.stderr).strip()
    except TimeoutError:
        return 124, "timeout"
    except FileNotFoundError:
        return 127, "warp-cli not installed"
    except Exception as exc:
        return 1, str(exc)


_port_lock = asyncio.Lock()
_allocated_ports: dict[tuple[str, int], int] = {}
_next_port = DEFAULT_BASE_SOCKS_PORT


async def allocate_socks_port(provider_id: str, idx: int, base: int) -> int:
    """Process-wide unique SOCKS port per (provider, slot)."""
    global _next_port
    async with _port_lock:
        key = (provider_id, idx)
        if key not in _allocated_ports:
            _allocated_ports[key] = max(base, _next_port)
            _next_port = _allocated_ports[key] + 1
        return _allocated_ports[key]


@dataclass
class WarpSlot:
    idx: int
    socks_port: int
    ready: bool = False
    last_error: str = ""
    registration_id: str = ""
    fail_count: int = 0
    last_heal: float = 0.0


@dataclass
class WarpPoolConfig:
    slots: int = DEFAULT_SLOTS
    hold_timeout_s: float = DEFAULT_HOLD_TIMEOUT_S
    reg_interval_sec: int = DEFAULT_REG_INTERVAL_SEC
    boot_retry_sec: int = DEFAULT_BOOT_RETRY_SEC
    base_socks_port: int = DEFAULT_BASE_SOCKS_PORT
    protocol: str = "MASQUE"
    masque: str = ""


class WarpPool:
    """Supervises warp-svc + warp-cli for one provider's slots."""

    def __init__(
        self,
        provider_id: str,
        data_dir: str | Path,
        config: WarpPoolConfig | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.data_dir = Path(data_dir)
        self.config = config or WarpPoolConfig()
        self.instances: list[WarpSlot] = []
        self.active: int = 0
        self.ready_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.status_cache: dict[int, dict[str, str]] = {}
        self.last_reg_ts: float = -self.config.reg_interval_sec
        self.binary_error: str = ""
        self._daemons: dict[int, asyncio.subprocess.Process] = {}
        self._tasks: list[asyncio.Task] = []
        self._started = False
        self._closed = False

    # -- lifecycle ----------------------------------------------------

    async def start(self) -> None:
        """Create slots, boot registered ones, spawn scheduler + refresher."""
        if self._started:
            return
        self._started = True
        if not warp_cli_available():
            self.binary_error = (
                "warp-cli not installed; warp provider unavailable "
                "(traffic fails open to direct)"
            )
            log.warning("[%s] %s", self.provider_id, self.binary_error)
            return
        for idx in range(max(0, self.config.slots)):
            port = await allocate_socks_port(
                self.provider_id, idx, self.config.base_socks_port
            )
            self.instances.append(WarpSlot(idx=idx, socks_port=port))
        self._tasks.append(asyncio.create_task(self._registration_scheduler()))
        self._tasks.append(asyncio.create_task(self._status_refresher()))

    async def aclose(self) -> None:
        self._closed = True
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for idx, proc in list(self._daemons.items()):
            if proc.returncode is not None:
                continue
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            # The IPC socket outlives a terminated daemon; a stale socket
            # makes the next ensure_daemon() return True for a dead daemon
            # (warp-cli then fails with "connection refused"). Remove it so
            # the next boot spawns fresh.
            self._unlink_stale_socket(
                runtime_dir_for(self.provider_id, idx) / "warp_service"
            )
        self._daemons.clear()

    # -- health / rotation API (used by ProviderRegistry) -------------

    def healthy(self) -> list[WarpSlot]:
        return [w for w in self.instances if w.ready]

    async def wait_ready(self, timeout: float | None = None) -> WarpSlot | None:
        try:
            await asyncio.wait_for(
                self.ready_event.wait(),
                timeout=self.config.hold_timeout_s if timeout is None else timeout,
            )
        except TimeoutError:
            return None
        healthy = self.healthy()
        if not healthy:
            return None
        for w in healthy:
            if w.idx == self.active:
                return w
        return healthy[0]

    async def rotate(self) -> dict:
        """Advance the active exit to the next healthy slot (local)."""
        async with self.lock:
            order = sorted(self.healthy(), key=lambda w: w.idx)
            nxt = next((w for w in order if w.idx > self.active), None)
            if nxt is None and order:
                nxt = order[0]
            if nxt is None:
                return {"ok": False, "error": "no healthy warps"}
            old = self.active
            self.active = nxt.idx
            loop = asyncio.get_running_loop()

            def _reopen() -> None:
                pid, idx, data_dir = self.provider_id, nxt.idx, self.data_dir
                run_cli(pid, idx, data_dir, "disconnect")
                run_cli(pid, idx, data_dir, "mode", "proxy")
                run_cli(pid, idx, data_dir, "proxy", "port", str(nxt.socks_port))
                run_cli(pid, idx, data_dir, "--accept-tos", "connect")

            await loop.run_in_executor(None, _reopen)
            asyncio.create_task(self._reverify(nxt))
            return {"ok": True, "old": old, "active": self.active}

    async def _reverify(self, inst: WarpSlot) -> None:
        inst.ready = False
        if not self.healthy():
            self.ready_event.clear()
        # A rotate re-runs the full boot (mode + port + connect) and can hit
        # the same slow-connect lag as the initial burst — keep watching past
        # the short window so the exit still flips ready on its own.
        if not await self._bring_up(inst, timeout=45):
            asyncio.create_task(self._watch_slot(inst))

    def snapshot(self) -> dict:
        """Local health snapshot: {active, exits[{idx,ready,status,...}]}."""
        return {
            "active": self.active,
            "error": self.binary_error,
            "exits": [
                {
                    "idx": w.idx,
                    "ready": w.ready,
                    "status": self.status_cache.get(w.idx, {}).get("status", "unknown"),
                    "reason": self.status_cache.get(w.idx, {}).get("reason", ""),
                    "socks": w.socks_port,
                    "registered": self.has_registration(w),
                    "error": w.last_error[-200:],
                }
                for w in self.instances
            ],
        }

    async def debug_config(self) -> dict:
        """Warp-cli metadata for the admin debug surface (no secrets)."""
        loop = asyncio.get_running_loop()
        version = await loop.run_in_executor(None, self._warp_version)
        cfg = self.config
        return {
            "ok": True,
            "managed": "in-process",
            "pool": {
                "provider": self.provider_id,
                "num_slots": len(self.instances),
                "hold_timeout": cfg.hold_timeout_s,
                "reg_interval_sec": cfg.reg_interval_sec,
                "boot_retry_sec": cfg.boot_retry_sec,
                "base_socks_port": cfg.base_socks_port,
                "protocol": cfg.protocol,
                "masque": cfg.masque or "default",
            },
            "build": {"warp_cli": version},
        }

    @staticmethod
    def _warp_version() -> str:
        try:
            proc = subprocess.run(  # noqa: PLW1510
                ["warp-cli", "--accept-tos", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return (proc.stdout + proc.stderr).strip()[:80]
        except Exception as exc:
            return f"unavailable: {exc}"[:80]

    # -- per-slot primitives ------------------------------------------

    def has_registration(self, slot: WarpSlot) -> bool:
        return (
            state_dir_for(self.data_dir, self.provider_id, slot.idx) / "reg.json"
        ).exists()

    def _budget_wait(self) -> float:
        return max(
            0.0, self.config.reg_interval_sec - (time.monotonic() - self.last_reg_ts)
        )

    def _mark_reg(self) -> None:
        self.last_reg_ts = time.monotonic()

    async def _cli(
        self, slot: WarpSlot, *args: str, timeout: int = 20
    ) -> tuple[int, str]:
        loop = asyncio.get_running_loop()
        pid, idx, data_dir = self.provider_id, slot.idx, self.data_dir
        return await loop.run_in_executor(
            None, lambda: run_cli(pid, idx, data_dir, *args, timeout=timeout)
        )

    def _prep_dirs(self, slot: WarpSlot) -> tuple[bool, str]:
        """Blocking dir prep for ensure_daemon (runs in an executor)."""
        for d in (
            state_dir_for(self.data_dir, self.provider_id, slot.idx),
            runtime_dir_for(self.provider_id, slot.idx),
            logs_dir_for(self.provider_id, slot.idx),
        ):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                if os.geteuid() == 0:
                    return False, f"{d} (needs root or pre-created dirs)"
                import subprocess as _sp

                try:
                    _sp.run(
                        ["sudo", "-n", "mkdir", "-p", str(d)],
                        capture_output=True,
                        timeout=10,
                        check=True,
                    )
                except Exception as exc:
                    return False, f"{d}: {exc} (needs root or pre-created dirs)"
        return True, ""

    def _socket_live_sync(self, slot: WarpSlot) -> bool:
        """Blocking liveness probe of a slot socket with no live tracked daemon."""
        _rc, out = run_cli(
            self.provider_id, slot.idx, self.data_dir, "status", timeout=5
        )
        low = out.lower()
        # A live daemon answers status; a dead one leaves its socket file
        # behind and warp-cli fails with "connection refused". Only a positive
        # refusal (or a missing binary) counts as stale — anything ambiguous
        # (timeout, slow boot) is trusted so a starting daemon is never
        # unlinked out from under itself.
        return "refus" not in low and _rc != 127

    def _unlink_stale_socket(self, sock: Path) -> bool:
        """Remove a dead daemon's socket; True when the path is gone."""
        try:
            sock.unlink(missing_ok=True)
        except OSError:
            if os.geteuid() != 0:
                # Supervised runtime dirs are root-owned (sudo-created), so a
                # non-root gateway needs sudo to clean up after a dead daemon.
                try:
                    subprocess.run(  # noqa: PLW1510
                        ["sudo", "-n", "rm", "-f", str(sock)],
                        capture_output=True,
                        timeout=10,
                    )
                except Exception:
                    pass
        return not sock.exists()

    async def ensure_daemon(self, slot: WarpSlot, timeout: float = 30) -> bool:
        sock = runtime_dir_for(self.provider_id, slot.idx) / "warp_service"
        if sock.exists():
            proc = self._daemons.get(slot.idx)
            if proc is not None and proc.returncode is None:
                return True
            # A socket with no live tracked daemon is an orphan from a previous
            # gateway run (runtime dirs key on provider+idx, not data dir) or a
            # stale file from a dead daemon. Probe before trusting it —
            # blindly returning True leaves every warp-cli call failing with
            # "connection refused" and the slot never becomes ready.
            check_loop = asyncio.get_running_loop()
            live = await check_loop.run_in_executor(None, self._socket_live_sync, slot)
            if live:
                return True
            log.warning(
                "[%s] removing stale daemon socket #%s",
                self.provider_id,
                slot.idx,
            )
            if not self._unlink_stale_socket(sock):
                log.warning(
                    "[%s] cannot remove stale socket #%s; skipping spawn",
                    self.provider_id,
                    slot.idx,
                )
                return False
        loop = asyncio.get_running_loop()
        ok, err = await loop.run_in_executor(None, lambda: self._prep_dirs(slot))
        if not ok:
            log.warning("[%s] cannot create dirs: %s", self.provider_id, err)
            return False
        env = warp_env(self.provider_id, slot.idx, self.data_dir)
        logf, log_err = await loop.run_in_executor(
            None, lambda: _open_log(self.provider_id, slot.idx, self.data_dir)
        )
        if logf is None:
            log.warning("[%s] cannot open daemon log: %s", self.provider_id, log_err)
            return False
        # warp-svc binds its IPC socket at "<RUNTIME_DIRECTORY>/warp_service"
        # (absolute path honored; verified live). sudo scrubs the environment,
        # so the non-root fallback re-exports the three isolation vars with
        # `sudo -n env VAR=...`. Daemons run with cwd=state dir as belt and
        # braces; the env values are always absolute.
        state_dir = state_dir_for(self.data_dir, self.provider_id, slot.idx)
        cmd = ["/bin/warp-svc"]
        if os.geteuid() != 0:
            cmd = [
                "sudo",
                "-n",
                "env",
                f"STATE_DIRECTORY={state_dir}",
                f"RUNTIME_DIRECTORY={runtime_dir_for(self.provider_id, slot.idx)}",
                f"LOGS_DIRECTORY={logs_dir_for(self.provider_id, slot.idx)}",
                *cmd,
            ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                cwd=str(state_dir),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=logf,
                stderr=asyncio.subprocess.STDOUT,
            )
        except Exception as exc:
            log.warning(
                "[%s] cannot spawn warp-svc #%s: %s", self.provider_id, slot.idx, exc
            )
            try:
                logf.close()
            except Exception:
                pass
            return False
        self._daemons[slot.idx] = proc
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if proc.returncode is not None:
                log.warning(
                    "[%s] warp-svc #%s exited early", self.provider_id, slot.idx
                )
                return False
            if sock.exists():
                return True
            await asyncio.sleep(0.5)
        log.warning(
            "[%s] warp-svc #%s no socket after %ss",
            self.provider_id,
            slot.idx,
            timeout,
        )
        return False

    async def ensure_proxy_mode(self, slot: WarpSlot) -> None:
        await self._cli(slot, "tunnel", "protocol", "set", self.config.protocol)
        if self.config.masque:
            await self._cli(slot, "tunnel", "masque-options", "set", self.config.masque)
        await self._cli(slot, "mode", "proxy")
        await self._cli(slot, "proxy", "port", str(slot.socks_port))
        # The explicit connect is what lifts the daemon out of
        # Disconnected(Manual) into the MASQUE handshake; without it the
        # tunnel only starts if some other client flips always-on first
        # (live finding: ride2's daemon sat until an unrelated SetAlwaysOn).
        # Note: the bare `connect` subcommand needs no --accept-tos, and
        # run_cli would inject it anyway; pass it bare so the daemon observes
        # exactly `warp-cli connect`. Log rc!=0 — a quiet failure here
        # leaves the slot Disconnected(Manual) forever.
        rc, out = await self._cli(slot, "connect")
        if rc != 0:
            log.warning(
                "[%s] connect failed idx=%s rc=%s: %s",
                self.provider_id,
                slot.idx,
                rc,
                out[:200],
            )
            slot.last_error = f"connect rc={rc}: {out[:200]}"

    def _connected_ready(self, out: str) -> bool:
        """True when a status text reports a live tunnel.

        warp-cli prints two lines ("Status update: Connected" + "Network:
        ...") and the daemon's interim states contain "Connecting"/"Unable",
        so require the Connected line and exclude every non-ready state.
        """
        low = out.lower()
        if "status update: connected" not in low:
            return False
        return not any(
            s in low for s in ("disconnected", "connecting", "unable", "missing")
        )

    async def _fill_registration_id(self, slot: WarpSlot) -> None:
        _rrc, rout = await self._cli(slot, "registration", "show")
        if _rrc == 0:
            for line in rout.splitlines():
                if line.strip().lower().startswith("id:"):
                    slot.registration_id = line.split(":", 1)[1].strip()

    async def poll_until_connected(self, slot: WarpSlot, timeout: float = 60) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            _rc, out = await self._cli(slot, "status")
            low = out.lower()
            if self._connected_ready(out):
                await self._fill_registration_id(slot)
                return True
            if "rate" in low or "429" in low or "too many" in low:
                slot.last_error = f"ratelimited: {out[:200]}"
                return False
            await asyncio.sleep(2)
        slot.last_error = "connect timeout"
        return False

    async def register_one(self, slot: WarpSlot) -> bool:
        rc, out = await self._cli(
            slot, "--accept-tos", "registration", "new", timeout=60
        )
        low = out.lower()
        if rc == 0 or self.has_registration(slot):
            return True
        if "rate" in low or "429" in low or "too many" in low or "limit" in low:
            slot.last_error = f"RATELIMITED: {out[:300]}"
            log.warning(
                "[%s] ratelimit idx=%s: %s", self.provider_id, slot.idx, out[:300]
            )
            return False
        slot.last_error = out[:300]
        log.warning(
            "[%s] register failed idx=%s: %s", self.provider_id, slot.idx, out[:300]
        )
        return "already" in low

    async def heal_stale(self, slot: WarpSlot) -> bool:
        now = time.monotonic()
        if now - slot.last_heal < HEAL_COOLDOWN_SEC:
            return False
        if self._budget_wait() > 0:
            log.info(
                "[%s] heal deferred idx=%s: registration budget spent",
                self.provider_id,
                slot.idx,
            )
            return False

        def _do() -> int:
            pid, idx, data_dir = self.provider_id, slot.idx, self.data_dir
            run_cli(pid, idx, data_dir, "registration", "delete")
            rc, _ = run_cli(
                pid, idx, data_dir, "--accept-tos", "registration", "new", timeout=60
            )
            return rc

        loop = asyncio.get_running_loop()
        rc = await loop.run_in_executor(None, _do)
        slot.last_heal = now
        slot.fail_count = 0
        if rc == 0 or self.has_registration(slot):
            self._mark_reg()
            log.info("[%s] healed stale warp idx=%s", self.provider_id, slot.idx)
            return True
        log.warning("[%s] heal failed idx=%s", self.provider_id, slot.idx)
        return False

    def _sibling_healthy(self, slot: WarpSlot) -> bool:
        return any(o.ready for o in self.instances if o.idx != slot.idx)

    async def _watch_slot(self, slot: WarpSlot, timeout: float = 300) -> None:
        """Keep watching a registered-but-unready slot past any one window.

        The initial burst poll is short (45s) so boot stays fast; on slow
        networks the tunnel needs longer. This watcher drives one extended
        connect wait and marks the slot ready the moment it connects.
        """
        if self._closed or slot.ready or not self.has_registration(slot):
            return
        if not await self.ensure_daemon(slot):
            slot.fail_count += 1
            return
        ok = await self.poll_until_connected(slot, timeout=timeout)
        if ok:
            self._slot_ready(slot)
        else:
            slot.fail_count += 1

    async def boot_one(self, slot: WarpSlot) -> None:
        ok = await self._bring_up(slot)
        if ok:
            return
        await self._maybe_heal(slot)

    # -- background loops ---------------------------------------------

    def _slot_ready(self, slot: WarpSlot) -> None:
        """Mark a slot ready and wake waiters (single funnel for transitions)."""
        slot.ready = True
        slot.fail_count = 0
        if self.healthy():
            self.ready_event.set()

    async def _bring_up(self, slot: WarpSlot, timeout: float = 120) -> bool:
        """One supervised boot: daemon ensured, proxy mode applied, then wait.

        On success the slot is marked ready (waking wait_ready callers);
        on failure the failure counter advances for the stale-heal path.
        """
        if not await self.ensure_daemon(slot):
            slot.last_error = slot.last_error or "daemon not running"
            slot.ready = False
            slot.fail_count += 1
            return False
        await self.ensure_proxy_mode(slot)
        ok = await self.poll_until_connected(slot, timeout=timeout)
        if ok:
            self._slot_ready(slot)
            return True
        slot.fail_count += 1
        return False

    async def _maybe_heal(self, slot: WarpSlot) -> bool:
        """Delete + re-register a stale slot when the cooldown/budget allows."""
        if (
            self.has_registration(slot)
            and slot.fail_count >= STALE_FAIL_THRESHOLD
            and self._sibling_healthy(slot)
            and await self.heal_stale(slot)
        ):
            return await self._bring_up(slot)
        return False

    async def _registration_scheduler(self) -> None:
        registered: set[int] = {
            w.idx for w in self.instances if self.has_registration(w)
        }
        for w in self.instances[:INITIAL_BURST]:
            if self._closed or w.idx in registered:
                continue
            if not await self.ensure_daemon(w):
                continue
            if await self.register_one(w):
                registered.add(w.idx)
                self._mark_reg()
                if not await self._bring_up(w, timeout=45):
                    # The connect phase can lag registration on slow networks;
                    # the refresher loop keeps watching past the burst window.
                    asyncio.create_task(self._watch_slot(w))
        if self.healthy():
            self.ready_event.set()
        for w in list(self.instances):
            if self._closed:
                return
            if w.ready or not self.has_registration(w):
                continue
            asyncio.create_task(self.boot_one(w))

        async def _retry_unready() -> None:
            while not self._closed:
                await asyncio.sleep(self.config.boot_retry_sec)
                for w in self.instances:
                    if self._closed:
                        return
                    if self.has_registration(w) and not w.ready:
                        asyncio.create_task(self.boot_one(w))

        asyncio.create_task(_retry_unready())
        while not self._closed and len(registered) < len(self.instances):
            await asyncio.sleep(self._budget_wait() or self.config.reg_interval_sec)
            nxt = next((w for w in self.instances if w.idx not in registered), None)
            if nxt is None:
                break
            if not await self.ensure_daemon(nxt):
                continue
            if await self.register_one(nxt):
                registered.add(nxt.idx)
                self._mark_reg()
                asyncio.create_task(self.boot_one(nxt))
            else:
                log.warning(
                    "[%s] registration failed idx=%s err=%s; retry next cycle",
                    self.provider_id,
                    nxt.idx,
                    nxt.last_error,
                )

    async def _refresh_one_status(self, slot: WarpSlot) -> None:
        _rc, out = await self._cli(slot, "status", timeout=5)
        status, reason = parse_status_output(out)
        self.status_cache[slot.idx] = {"status": status, "reason": reason}
        # Promotion path: a registered slot whose tunnel connected outside
        # any single wait window (slow networks beat the burst poll; a watcher
        # may be absent after restarts). Flip ready here so wait_ready callers
        # and health snapshots converge without another full boot. Uses the
        # same strict matcher as the connect poll (interim states like
        # "Connecting" contain "connecting", not a Connected line).
        if (
            not slot.ready
            and self._connected_ready(out)
            and self.has_registration(slot)
        ):
            await self._fill_registration_id(slot)
            self._slot_ready(slot)

    async def refresh_statuses(self) -> None:
        """One-shot status poll of every slot (for health refresh)."""
        if not warp_cli_available() or not self.instances:
            return
        await asyncio.gather(
            *(self._refresh_one_status(w) for w in self.instances),
            return_exceptions=True,
        )

    async def _status_refresher(self) -> None:
        while not self._closed:
            try:
                await self.refresh_statuses()
            except Exception as exc:
                log.warning("[%s] status refresh failed: %r", self.provider_id, exc)
            try:
                await asyncio.sleep(STATUS_CACHE_SEC)
            except asyncio.CancelledError:
                return


@dataclass
class WarpSupervisor:
    """Owns one WarpPool per warp provider id."""

    data_dir: str | Path
    pools: dict[str, WarpPool] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def ensure(
        self, provider_id: str, config: WarpPoolConfig | None = None
    ) -> WarpPool:
        async with self.lock:
            pool = self.pools.get(provider_id)
            if pool is None:
                pool = WarpPool(provider_id, self.data_dir, config)
                self.pools[provider_id] = pool
                await pool.start()
            return pool

    def get(self, provider_id: str) -> WarpPool | None:
        return self.pools.get(provider_id)

    async def drop(self, provider_id: str) -> None:
        async with self.lock:
            pool = self.pools.pop(provider_id, None)
        if pool is not None:
            await pool.aclose()

    async def aclose(self) -> None:
        async with self.lock:
            pools = list(self.pools.values())
            self.pools.clear()
        for pool in pools:
            await pool.aclose()
