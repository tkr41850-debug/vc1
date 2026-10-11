"""Multi-turn codex battery (responses leg). See codex/multi_turn.md."""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from battery import battery_dirs, codex_env, require_live, run_cli

CODEX_BIN = os.getenv("CODEX_BIN", "/home/uqmm/.nvm/versions/node/v24.15.0/bin/codex")
SPARK = "muse-spark-1.3-contributor-free"


def test_cx_multi_resume_last():
    base, sk = require_live()
    home, ws = battery_dirs("cx-resume-last")
    env = codex_env(home, base, "ak-battery-cx11", sk, SPARK)
    first = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Write exactly MARKER-RESUME-LAST-7Q2T to marker.txt and "
                "reply exactly MARKER-RESUME-LAST-7Q2T."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    second = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "resume",
            "--last",
            "Read marker.txt and reply exactly RESUMED-<contents> and nothing else.",
        ],
        ws,
        env,
    )
    assert second.returncode == 0, second.stderr[-1000:]
    assert "RESUMED-MARKER-RESUME-LAST-7Q2T" in second.stdout
    assert (ws / "marker.txt").read_bytes().rstrip(b"\n") == b"MARKER-RESUME-LAST-7Q2T"


def test_cx_multi_resume_session_id():
    base, sk = require_live()
    home, ws = battery_dirs("cx-resume-sid")
    env = codex_env(home, base, "ak-battery-cx12", sk, SPARK)
    first = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            "--json",
            (
                "Write exactly MARKER-SID-9X4P to marker.txt and reply "
                "exactly MARKER-SID-9X4P."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    session_id = ""
    for line in first.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = json.dumps(event)
        if "thread_id" in payload or "session" in payload.lower():
            text = json.dumps(event)
            import re

            m = re.search(r'"thread_id"\s*:\s*"([^"]+)"', text)
            if m:
                session_id = m.group(1)
                break
    assert session_id, "no session/thread id in --json events"
    second = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "resume",
            session_id,
            "Read marker.txt and reply exactly RESUMED-<contents> and nothing else.",
        ],
        ws,
        env,
    )
    assert second.returncode == 0, second.stderr[-1000:]
    assert "RESUMED-MARKER-SID-9X4P" in second.stdout
    assert (ws / "marker.txt").read_bytes().rstrip(b"\n") == b"MARKER-SID-9X4P"


def test_cx_multi_tool_chain():
    base, sk = require_live()
    home, ws = battery_dirs("cx-tool-chain")
    env = codex_env(home, base, "ak-battery-cx13", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "1. Create numbers.txt with integers 1-10 one per line. "
                "2. Shell-compute the sum into sum.txt. "
                "3. Read sum.txt. "
                "4. Create doubled.txt with exactly twice that sum. "
                "5. Reply exactly session-sum=<doubled.txt value> and "
                "nothing else."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "session-sum=110" in c.stdout
    assert (ws / "numbers.txt").read_text().split() == [str(i) for i in range(1, 11)]
    assert (ws / "sum.txt").read_text().strip() == "55"
    assert (ws / "doubled.txt").read_text().strip() == "110"


def test_cx_multi_thread_continuation():
    base, sk = require_live()
    home, ws = battery_dirs("cx-thread-cont")
    env = codex_env(home, base, "ak-battery-cx14", sk, SPARK)
    first = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Write exactly THREAD-CONT-3K8R to thread.txt and reply "
                "exactly THREAD-CONT-3K8R."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    second = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "resume",
            "--last",
            "Read thread.txt and reply exactly CONTINUED-<contents> and nothing else.",
        ],
        ws,
        env,
    )
    assert second.returncode == 0, second.stderr[-1000:]
    assert "CONTINUED-THREAD-CONT-3K8R" in second.stdout
    assert (ws / "thread.txt").read_bytes().rstrip(b"\n") == b"THREAD-CONT-3K8R"


@pytest.mark.xfail(
    reason="bare previous_response_id chaining drops prior context",
    strict=False,
)
def test_cx_multi_previous_response_id():
    """Direct chaining leg (no codex binary): turn 2 carries
    previous_response_id from turn 1. Under anonymous shaping (no key on
    the upstream request), each turn mints its own upstream session — so
    this asserts continuity differently: leg 2's reply must still echo the
    marker, proving the gateway carried the prior context into the turn.

    NOTE (live 2026-10-11): leg 2 echoed ECHOED-<marker> literally on
    5/5 attempts — prior-turn context is NOT surviving the anonymized
    round-trip (session split: each leg gets its own upstream ses_ since
    nothing up there maps the downstream response id back). xfail pin
    until the llms chaining path is fixed."""
    import urllib.request

    base, sk = require_live()

    def post(affinity: str, body: dict) -> dict:
        req = urllib.request.Request(
            f"{base}/{affinity}/v1/responses",
            data=__import__("json").dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {sk}",
                "Content-Type": "application/json",
                # Neutral UA: citr's edge WAF blocks bare-Python UAs (1010),
                # and codex/opencode UAs would key the wrong shaping path.
                "User-Agent": "llms-battery/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return __import__("json").loads(r.read().decode())

    leg1 = post(
        "ak-battery-cx15",
        {
            "model": SPARK,
            "instructions": "Reply exactly CHAIN-5M2N",
            "input": "emit marker",
        },
    )
    r1 = leg1.get("id", "")
    assert r1, f"no response id in {str(leg1)[:300]}"
    leg2 = post(
        "ak-battery-cx15",
        {
            "model": SPARK,
            "previous_response_id": r1,
            "input": "Echo the prior marker exactly as ECHOED-<marker> and "
            "nothing else",
        },
    )
    assert "ECHOED-CHAIN-5M2N" in __import__("json").dumps(leg2)


def test_cx_multi_review_modes():
    import shutil

    base, sk = require_live()
    home, ws = battery_dirs("cx-review")
    env = codex_env(home, base, "ak-battery-cx16", sk, SPARK)
    # battery_dirs only clears files; drop stale .git so reruns re-commit.
    shutil.rmtree(ws / ".git", ignore_errors=True)
    (ws / "review.txt").write_text("REVIEW-LINE\n")
    subprocess.run(["git", "init", "-q", "-b", "battery"], cwd=str(ws), check=True)
    subprocess.run(
        ["git", "add", "review.txt"],
        cwd=str(ws),
        check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=b@t", "-c", "user.name=b", "commit", "-qm", "base"],
        cwd=str(ws),
        check=True,
        capture_output=True,
    )
    (ws / "review.txt").write_text("REVIEW-LINE-CHANGED\n")
    # `codex review` is top-level (not `exec`); --uncommitted conflicts
    # with a positional prompt, so both legs are plain no-prompt reviews
    # whose output must mention the dirty file.
    plain = run_cli(
        [
            CODEX_BIN,
            "review",
            "--uncommitted",
        ],
        ws,
        env,
    )
    assert plain.returncode == 0, plain.stderr[-1000:]
    assert "review.txt" in plain.stdout
    again = run_cli(
        [
            CODEX_BIN,
            "review",
            "--uncommitted",
        ],
        ws,
        env,
    )
    assert again.returncode == 0, again.stderr[-1000:]
    assert "review.txt" in again.stdout
