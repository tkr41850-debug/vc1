"""Hermetic tests for the live-battery harness (battery/__init__.py).

The battery directory is NOT collected by the top-level suite (testpaths
excludes it; require_live skips without LLMS_BATTERY_LIVE=1), so the
helpers that gate retries, skip spent quotas, and bound /tmp usage get
their coverage here.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from battery import (
    free_quota_spent,
    is_retryable_429,
    prune_stale_sessions,
    skip_if_quota_spent,
)


def _result(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["codex", "exec"], returncode=1, stdout=stdout, stderr=stderr
    )


def test_shape_error_never_retries():
    # The anonymous-shape rejection ("free tier can only be used from
    # within OpenCode") is deterministic — retrying it 5x burns minutes
    # for the same verdict (live 2026-10-12: direct-anonymous legs).
    blob = (
        '{"type":"error","error":{"type":"FreeTierError","message":'
        '"OpenCode\'s free tier can only be used from within OpenCode"}}'
    )
    assert is_retryable_429(_result(stderr=blob)) is False
    assert is_retryable_429(_result(stdout=blob)) is False
    # Even alongside 429 bytes: the shape verdict dominates.
    assert is_retryable_429(_result(stderr="429 Too Many Requests\n" + blob)) is False


def test_genuine_429_storm_still_retries():
    assert (
        is_retryable_429(
            _result(
                stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests"
            )
        )
        is True
    )


def test_clean_failure_never_retries():
    assert is_retryable_429(_result(stdout="", stderr="")) is False
    assert is_retryable_429(_result(stdout="SUBAGENT-OK")) is False


def test_spent_free_quota_never_retries():
    # Upstream's daily cap answers retry-after in hours (live 2026-10-25:
    # 47195s ≈ 13h), so retrying five times only burns the leg's budget.
    blob = (
        '{"type":"error","error":{"type":"FreeUsageLimitError","message":'
        '"Free usage limit reached. Free models work in OpenCode"}}'
    )
    assert is_retryable_429(_result(stderr="429 Too Many Requests\n" + blob)) is False


def test_spent_free_quota_skips_the_leg():
    blob = "429 Too Many Requests: Free usage limit reached."
    with pytest.raises(pytest.skip.Exception, match="free usage limit exhausted"):
        skip_if_quota_spent(_result(stderr=blob))


def test_completed_leg_is_not_skipped():
    assert skip_if_quota_spent(_result(stdout="DONE-1")) is None


def _probe_response(status: int, text: str):
    request = httpx.Request("POST", "http://gw.test/v1/chat/completions")
    return httpx.Response(status, text=text, request=request)


def test_probe_reads_quota_from_gateway_429(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("sk-test\n")
    monkeypatch.setenv("BATTERY_BASE", "http://gw.test/")
    monkeypatch.setenv("BATTERY_KEY_FILE", str(key))
    seen = {}

    def fake_post(url, **kwargs):
        seen["url"] = url
        return _probe_response(429, '{"error":{"message":"Free usage limit reached"}}')

    monkeypatch.setattr(httpx, "post", fake_post)
    assert free_quota_spent() is True
    # Trailing slash must not double up in the probe URL.
    assert seen["url"] == "http://gw.test/v1/chat/completions"


def test_probe_treats_burst_429_and_success_as_not_spent(monkeypatch, tmp_path):
    key = tmp_path / "key"
    key.write_text("sk-test\n")
    monkeypatch.setenv("BATTERY_BASE", "http://gw.test")
    monkeypatch.setenv("BATTERY_KEY_FILE", str(key))
    monkeypatch.setattr(
        httpx, "post", lambda url, **kw: _probe_response(429, '{"error":"slow down"}')
    )
    assert free_quota_spent() is False
    monkeypatch.setattr(httpx, "post", lambda url, **kw: _probe_response(200, "{}"))
    assert free_quota_spent() is False


def _leg(root: Path, tag: str, age_s: float) -> Path:
    sessions = root / f"llms-battery-{tag}" / "home" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / "rollout.jsonl").write_text("{}\n")
    stamp = time.time() - age_s
    os.utime(sessions, (stamp, stamp))
    return sessions


def test_prune_stale_sessions_drops_old_legs_only(tmp_path):
    stale = _leg(tmp_path, "cx-old", age_s=3600)
    fresh = _leg(tmp_path, "cx-running", age_s=0)

    removed = prune_stale_sessions(root=tmp_path, max_age_s=900)

    assert removed == 1
    assert not stale.exists()
    # A leg that appended within the window is still writing: keep it.
    assert (fresh / "rollout.jsonl").exists()


def test_prune_stale_sessions_spares_the_starting_leg(tmp_path):
    own = _leg(tmp_path, "cx-current", age_s=3600)

    removed = prune_stale_sessions(root=tmp_path, ignore=own.parent, max_age_s=900)

    assert removed == 0
    assert own.exists()
