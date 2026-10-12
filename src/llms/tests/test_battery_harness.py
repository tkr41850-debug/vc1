"""Hermetic tests for the live-battery harness retry gate (battery/__init__.py).

The battery directory is NOT collected by the top-level suite (testpaths
excludes it; require_live skips without LLMS_BATTERY_LIVE=1), so the
retry-gate helper gets its hermetic coverage here: a shape rejection
must fail fast (no retries), a genuine 429 storm must still retry.
"""

from __future__ import annotations

import subprocess

from battery import is_retryable_429


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
            _result(stderr="ERROR: exceeded retry limit, last status: 429 Too Many Requests")
        )
        is True
    )


def test_clean_failure_never_retries():
    assert is_retryable_429(_result(stdout="", stderr="")) is False
    assert is_retryable_429(_result(stdout="SUBAGENT-OK")) is False
