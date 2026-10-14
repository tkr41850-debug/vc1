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
import subprocess
import time
from pathlib import Path

import pytest

TIMEOUT = int(os.getenv("BATTERY_TIMEOUT", "600"))
# Gateway 429s under parallel load are transient: retry slow (5s) and
# often (5 tries) before letting the leg fail.
RETRY_429_ATTEMPTS = int(os.getenv("BATTERY_RETRY_429_ATTEMPTS", "5"))
RETRY_429_DELAY_S = float(os.getenv("BATTERY_RETRY_429_DELAY_S", "5"))


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
    # every leg re-seeds its own history — so prune them at leg start.
    # The gateway log (/tmp/llms.log) remains the primary trace.
    import shutil

    shutil.rmtree(home / "sessions", ignore_errors=True)
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

    The anonymous-shape error ("free tier can only be used from within
    OpenCode") is NOT retryable: it is a deterministic request-shape
    rejection, not a rate limit — retrying it burns 5 attempts for the
    same verdict.
    """
    blob = (result.stdout or "") + (result.stderr or "")
    if "can only be used from within OpenCode" in blob:
        return False
    return "429" in blob and "Too Many Requests" in blob


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
    return last
