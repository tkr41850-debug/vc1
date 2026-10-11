"""Multi-turn claude battery (messages leg). See claude/multi_turn.md."""

from __future__ import annotations

import json
import os

from battery import battery_dirs, claude_env, require_live, run_cli

CLAUDE_BIN = os.getenv("CLAUDE_BIN", "/home/uqmm/.local/bin/claude")


def test_cl_continue_recall():
    base, sk = require_live()
    _, ws = battery_dirs("cl-continue")
    env = claude_env(base, "ak-battery-cl11", sk)
    first = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            (
                "Reply with exactly MARKER-7F3A and nothing else. Do not "
                "use any tools. Do not add quotes, punctuation, or "
                "explanation."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    second = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--continue",
            "--model",
            "haiku",
            "--output-format",
            "text",
            (
                "Repeat the exact marker from this session and nothing "
                "else. Do not use any tools."
            ),
        ],
        ws,
        env,
    )
    assert second.returncode == 0, second.stderr[-1000:]
    assert second.stdout.strip() == "MARKER-7F3A"


def test_cl_resume_session():
    base, sk = require_live()
    _, ws = battery_dirs("cl-resume")
    env = claude_env(base, "ak-battery-cl12", sk)
    first = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "json",
            (
                "Reply with exactly MARKER-RESUME-51C2 and nothing else. "
                "Do not use any tools."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    payload = json.loads(first.stdout)
    s1 = payload.get("session_id", "")
    assert s1, f"no session_id in {first.stdout[:300]}"
    second = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--resume",
            s1,
            "--model",
            "haiku",
            "--output-format",
            "json",
            (
                "Repeat the exact marker from this session and nothing "
                "else. Do not use any tools."
            ),
        ],
        ws,
        env,
    )
    assert second.returncode == 0, second.stderr[-1000:]
    payload2 = json.loads(second.stdout)
    assert payload2.get("session_id", "") == s1
    assert json.dumps(payload2).count("MARKER-RESUME-51C2") >= 1


def test_cl_file_write_read():
    base, sk = require_live()
    _, ws = battery_dirs("cl-write-read")
    env = claude_env(base, "ak-battery-cl13", sk)
    target = ws / "marker1.txt"
    if target.exists():
        target.unlink()
    first = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--allowedTools",
            "Write",
            f"--add-dir={ws}",
            (
                'Use the Write tool to create file "marker1.txt" in the '
                "current working directory with the exact bytes "
                "HELLO-BATTERY-42 and no trailing newline. Reply with "
                "exactly WROTE-OK and nothing else."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    assert target.read_bytes() == b"HELLO-BATTERY-42"
    second = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--continue",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--allowedTools",
            "Read",
            f"--add-dir={ws}",
            (
                'Read file "marker1.txt" in the current working '
                "directory with the Read tool and reply with its exact "
                "contents and nothing else."
            ),
        ],
        ws,
        env,
    )
    assert second.returncode == 0, second.stderr[-1000:]
    assert second.stdout.strip() == "HELLO-BATTERY-42"


def test_cl_file_modify():
    base, sk = require_live()
    _, ws = battery_dirs("cl-modify")
    env = claude_env(base, "ak-battery-cl14", sk)
    target = ws / "lines.txt"
    if target.exists():
        target.unlink()
    first = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--allowedTools",
            "Write",
            f"--add-dir={ws}",
            (
                'Use the Write tool to create file "lines.txt" in the '
                "current working directory with the exact bytes "
                "LINE-ONE-1111 and no trailing newline. Reply with "
                "exactly WROTE-OK and nothing else."
            ),
        ],
        ws,
        env,
    )
    assert first.returncode == 0, first.stderr[-1000:]
    second = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--continue",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--allowedTools",
            "Read,Edit",
            f"--add-dir={ws}",
            (
                'Use the Read tool to read "lines.txt", then use the '
                "Edit tool to append exactly one line so the file "
                "becomes LINE-ONE-1111 newline LINE-TWO-2222 with no "
                "trailing newline. Reply with exactly APPENDED-OK and "
                "nothing else."
            ),
        ],
        ws,
        env,
    )
    assert second.returncode == 0, second.stderr[-1000:]
    assert target.read_bytes() == b"LINE-ONE-1111\nLINE-TWO-2222"
    third = run_cli(
        [
            CLAUDE_BIN,
            "-p",
            "--continue",
            "--model",
            "haiku",
            "--output-format",
            "text",
            "--allowedTools",
            "Read",
            f"--add-dir={ws}",
            (
                'Read file "lines.txt" with the Read tool and reply '
                "with its exact contents and nothing else."
            ),
        ],
        ws,
        env,
    )
    assert third.returncode == 0, third.stderr[-1000:]
    assert third.stdout.strip() == "LINE-ONE-1111\nLINE-TWO-2222"
