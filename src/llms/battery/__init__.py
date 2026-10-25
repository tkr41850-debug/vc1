"""Shared harness for the live CLI battery (codex/ + claude/).

These tests drive the real CLIs through a gateway and assert exact,
model-independent outcomes (byte-exact files, exact markers). They are
NOT collected by the top-level suite:

- ``testpaths = ["tests"]`` in pyproject.toml excludes this directory.
- Every module in here additionally calls :func:`require_live`, which
  skips unless ``LLMS_BATTERY_LIVE=1`` is set — pointing pytest at this
  directory without the flag is always a no-op.

Each test gets a distinct ``ak-batteryN`` affinity key so the gateway's
bucket hash spreads legs across pools instead of pinning one.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest

TIMEOUT = int(os.getenv("BATTERY_TIMEOUT", "600"))
# Gateway 429s under parallel load are transient: retry slow (5s) and
# often (5 tries) before letting the leg fail.
RETRY_429_ATTEMPTS = int(os.getenv("BATTERY_RETRY_429_ATTEMPTS", "5"))
RETRY_429_DELAY_S = float(os.getenv("BATTERY_RETRY_429_DELAY_S", "5"))
# Leftover rollout sessions older than this are fair game to prune; live
# legs append transcripts continuously, so an actively running home never
# looks this idle.
STALE_SESSIONS_AGE_S = float(os.getenv("BATTERY_STALE_SESSIONS_AGE_S", "900"))
FREE_LIMIT_MARKER = "Free usage limit reached"
# Any free model reports the account-wide daily cap; use the default one
# so the probe works regardless of which model the leg drives.
FREE_PROBE_MODEL = os.getenv("BATTERY_FREE_PROBE_MODEL", "glm-5.3-flash-free")
QUOTA_SKIP_REASON = (
    "live battery: upstream free usage limit exhausted — resets at UTC "
    "midnight; rerun after the reset (or with an operator ZEN_API_KEY)"
)


def require_live() -> tuple[str, str]:
    """Skip unless live runs are explicitly enabled; return (base, key).

    ``BATTERY_BASE`` and the key file are the only inputs — no URLs or
    secrets live in this repo (the ``ak-`` affinity strings in tests are
    unauth routing prefixes, not credentials).
    """
    if os.getenv("LLMS_BATTERY_LIVE") != "1":
        pytest.skip("live battery: set LLMS_BATTERY_LIVE=1 to run")
    base = os.getenv("BATTERY_BASE", "").rstrip("/")
    if not base:
        pytest.skip("live battery: set BATTERY_BASE to the gateway URL")
    key_file = Path(os.getenv("BATTERY_KEY_FILE", "/tmp/upstream_sk"))
    try:
        sk = key_file.read_text().strip()
    except OSError:
        pytest.skip(f"live battery: key file {key_file} unreadable")
    if not sk.startswith("sk-"):
        pytest.skip(f"live battery: key file {key_file} has no sk- secret")
    return base, sk


def prune_stale_sessions(
    root: Path = Path("/tmp"),
    ignore: Path | None = None,
    max_age_s: float = STALE_SESSIONS_AGE_S,
    now: float | None = None,
) -> int:
    """Drop rollout transcripts left behind by finished battery legs.

    ``/tmp`` is a 3.9G tmpfs here and each leg's codex/claude home grows
    to ~100–300M of rollout sessions. Pruning only the current leg's home
    lets a full pass leave ~2–4G behind, so the next pass dies ENOSPC
    mid-run (live 2026-10-25: tmpfs at 100%, battery logs truncated).
    Homes untouched for ``max_age_s`` cannot belong to a running leg —
    both CLIs append transcripts continuously — so parallel runs (-n 3)
    are safe. ``ignore`` protects the leg that is about to start.

    Returns the number of session trees removed.
    """
    moment = time.time() if now is None else now
    removed = 0
    for sessions in sorted(root.glob("llms-battery-*/home/sessions")):
        if ignore is not None and sessions.parent == ignore:
            continue
        try:
            age = moment - sessions.stat().st_mtime
        except OSError:
            continue
        if age < max_age_s:
            continue
        shutil.rmtree(sessions, ignore_errors=True)
        removed += 1
    return removed


def battery_dirs(tag: str) -> tuple[Path, Path]:
    base = Path(f"/tmp/llms-battery-{tag}")
    home, ws = base / "home", base / "workspace"
    home.mkdir(parents=True, exist_ok=True)
    ws.mkdir(parents=True, exist_ok=True)
    for f in ws.glob("*"):
        if f.is_file():
            f.unlink()
    # /tmp is a small tmpfs here and codex/claude CLIs accumulate
    # rollout transcripts under home/sessions (~100M per leg per run —
    # live 2026-10-14: a 10x probe filled the 3.9G tmpfs to 100% and
    # failed legs with ENOSPC). No leg reads another leg's sessions —
    # every leg re-seeds its own history — so prune them at leg start,
    # including homes left behind by earlier passes.
    # The gateway log (/tmp/llms.log) remains the primary trace.
    shutil.rmtree(home / "sessions", ignore_errors=True)
    prune_stale_sessions(root=base.parent, ignore=home)
    return home, ws


def codex_env(home: Path, base: str, affinity: str, sk: str, model: str) -> dict:
    (home / "config.toml").write_text(
        f'model = "{model}"\nmodel_provider = "zen-proxy"\n\n'
        '[model_providers.zen-proxy]\nname = "zen-proxy"\n'
        f'base_url = "{base}/{affinity}/v1"\nwire_api = "responses"\n'
        'env_key = "OPENAI_API_KEY"\n'
    )
    return dict(os.environ, CODEX_HOME=str(home), OPENAI_API_KEY=sk)


def claude_env(base: str, affinity: str, sk: str) -> dict:
    env = dict(os.environ)
    env.pop("ANTHROPIC_MODEL", None)
    env.update(
        ANTHROPIC_BASE_URL=f"{base}/{affinity}",
        ANTHROPIC_API_KEY=sk,
        ANTHROPIC_AUTH_TOKEN="",
    )
    return env


def is_retryable_429(result: subprocess.CompletedProcess) -> bool:
    """True when the run died on a gateway-side 429 storm, not a real failure.

    Two 429s are NOT retryable:

    - the anonymous-shape error ("free tier can only be used from within
      OpenCode") — a deterministic request-shape rejection, not a rate
      limit;
    - the daily free-tier exhaustion ("Free usage limit reached") —
      upstream answers `retry-after` in hours (live 2026-10-25: 47195s ≈
      13h, resetting at UTC midnight), so retrying burns the budget for
      the same verdict.
    """
    blob = (result.stdout or "") + (result.stderr or "")
    if "can only be used from within OpenCode" in blob:
        return False
    if FREE_LIMIT_MARKER in blob:
        return False
    return "429" in blob and "Too Many Requests" in blob


def skip_if_quota_spent(result: subprocess.CompletedProcess) -> None:
    """Skip the leg when upstream's daily free quota is exhausted.

    A spent quota says nothing about the gateway, so failing the leg
    would report a code regression that does not exist; a skip with the
    reset hint keeps the signal honest.
    """
    blob = (result.stdout or "") + (result.stderr or "")
    if FREE_LIMIT_MARKER in blob:
        pytest.skip(QUOTA_SKIP_REASON)


def free_quota_spent(timeout_s: float = 30.0) -> bool:
    """One tiny free-model turn to read the 429 body the CLIs swallow.

    Codex/Claude print only ``last status: 429 Too Many Requests``, so in
    their output a spent daily quota (skip-worthy) is indistinguishable
    from a storm (retry-worthy). Called only after a 429 failure, so the
    extra request costs nothing in the healthy path.
    """
    base = os.getenv("BATTERY_BASE", "").rstrip("/")
    if not base:
        return False
    key_file = Path(os.getenv("BATTERY_KEY_FILE", "/tmp/upstream_sk"))
    try:
        sk = key_file.read_text().strip()
    except OSError:
        return False
    try:
        resp = httpx.post(
            f"{base}/v1/chat/completions",
            json={
                "model": FREE_PROBE_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            headers={"Authorization": f"Bearer {sk}"},
            timeout=timeout_s,
        )
    except Exception:
        return False
    return resp.status_code == 429 and FREE_LIMIT_MARKER in resp.text


def run_cli(
    cmd: list[str], cwd: Path, env: dict, input_text: str | None = None
) -> subprocess.CompletedProcess:
    """Run a CLI leg, retrying gateway 429 storms (5s x5) before failing."""
    last: subprocess.CompletedProcess | None = None
    for attempt in range(1 + RETRY_429_ATTEMPTS):
        last = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
            stdin=subprocess.DEVNULL if input_text is None else None,
            input=input_text,
        )
        if attempt < RETRY_429_ATTEMPTS and is_retryable_429(last):
            time.sleep(RETRY_429_DELAY_S)
            continue
        break
    assert last is not None
    skip_if_quota_spent(last)
    blob = (last.stdout or "") + (last.stderr or "")
    if "429" in blob and free_quota_spent():
        pytest.skip(QUOTA_SKIP_REASON)
    return last
