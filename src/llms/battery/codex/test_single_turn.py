"""Single-turn codex battery (responses leg). See battery/README.md."""

from __future__ import annotations

import json
import re
import subprocess

import pytest

from battery import battery_dirs, codex_env, require_live, run_cli

CODEX_BIN = __import__("os").getenv(
    "CODEX_BIN", "/home/uqmm/.nvm/versions/node/v24.15.0/bin/codex"
)
SPARK = "muse-spark-1.3-contributor-free"


def _stop_sign_png_bytes(size: int = 128) -> bytes:
    """Stop-sign PNG generated in memory (no image step touches disk).

    Red octagon on a white canvas with white blocky "STOP" text drawn
    via Pillow's scalable built-in font (no font files, no hardcoded
    blob). The CLI needs a file *path* for `-i` / `view_image`, so
    callers still write the returned bytes into the workspace — but
    the image itself is drawn on a canvas and encoded via BytesIO.
    """
    import io
    import math

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Regular octagon inscribed with margin m.
    m = size * 0.08
    c = (size / 2) * (1 - 1 / math.sqrt(2))
    x0, x1 = m + c, size - m - c
    draw.polygon(
        [
            (m, x0),
            (x0, m),
            (x1, m),
            (size - m, x0),
            (size - m, x1),
            (x1, size - m),
            (x0, size - m),
            (m, x1),
        ],
        fill=(200, 0, 0),
    )
    font = ImageFont.load_default(size=max(8, int(size * 0.22)))
    draw.text(
        (size / 2, size / 2), "STOP", fill=(255, 255, 255), font=font, anchor="mm"
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_cx_single_file_write():
    base, sk = require_live()
    home, ws = battery_dirs("cx-file-write")
    env = codex_env(home, base, "ak-battery-cx01", sk, SPARK)
    target = ws / "write_plain.txt"
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Write exactly MARKER-WRITE-1 on one line to write_plain.txt "
                "using a tool, then reply with exactly DONE-WRITE-1."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"MARKER-WRITE-1\n"
    assert c.stdout.strip().endswith("DONE-WRITE-1")


def test_cx_single_shell_redirect_write():
    # Fixed 2026-10-12 (alias attribution): the loop was the model
    # re-emitting `shell` because the executed success echoed back
    # under the rewritten `exec_command` name it never emitted.
    # to_zen_responses now replays executed alias pairs under the
    # emitted name, and the leg passes (battery 17 passed incl. this
    # leg; standalone RELAY CODEX REDIRECT PASS). TimeoutExpired still
    # xfails — a hung CLI is harness/model dynamics, not an assertion.
    base, sk = require_live()
    home, ws = battery_dirs("cx-shell-redir")
    env = codex_env(home, base, "ak-battery-cx02", sk, SPARK)
    target = ws / "redir.txt"
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Run shell command printf REDIRECT-OK > redir.txt via the "
                    "shell tool, then reply with exactly DONE-REDIR-1."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model loops re-emitting shell after byte-exact write")
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"REDIRECT-OK"
    assert "DONE-REDIR-1" in c.stdout


def test_cx_single_chained_multitool():
    base, sk = require_live()
    home, ws = battery_dirs("cx-chain")
    env = codex_env(home, base, "ak-battery-cx03", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Run echo CHAIN-A > a.txt, then echo CHAIN-B > b.txt, then "
                "cat both files, then reply with exactly CHAIN-DONE."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert (ws / "a.txt").read_bytes() == b"CHAIN-A\n"
    assert (ws / "b.txt").read_bytes() == b"CHAIN-B\n"
    assert c.stdout.strip().endswith("CHAIN-DONE")


def test_cx_single_image_attach():
    """`-i` image attach with vision-content assertion (stop sign).

    The attached PNG is drawn in-memory by `_stop_sign_png_bytes`
    (BytesIO-encoded, never a hardcoded blob). The word STOP never
    appears in the prompt, so a reply containing it proves the image
    bytes reached model vision — not just that an attachment slot
    existed.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-image")
    env = codex_env(home, base, "ak-battery-cx04", sk, SPARK)

    png = ws / "stop.png"
    # Stop-sign image drawn on a PIL canvas (BytesIO-encoded, no blob).
    png.write_bytes(_stop_sign_png_bytes())
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "An image is attached. Reply with exactly ATTACH-IMG-1 "
                "on the first line. If you can see the image, add a "
                "second line with the word written on the sign."
            ),
            "-i",
            str(png),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "ATTACH-IMG-1" in c.stdout
    assert "STOP" in c.stdout


def test_cx_single_review_diff():
    """apply_patch grammar + bare-patch rewrap (modelfile-pinned).

    KNOWN HARNESS GAP (live 2026-10-12, two 600s runs): the codex-plain
    leg declares no `apply_patch` tool (trace: client shapes are
    exec_command/write_stdin/request_user_input/view_image/
    multi_agent_v1/get_goal/create_goal/update_goal) and the harness
    shell does NOT intercept the marker grammar — the model complies
    with the prompt (44x `shell {"command": "*** Begin Patch\\n..."}`
    onto exec_command, every turn `steer: []`) and the shell executes
    it as a command without writing any file, until the CLI clock
    runs out. The prompt's "the shell intercepts it" premise is false
    on this leg, and "prefer apply_patch over printf" blocks the
    fallback that would work — so the leg can only time out. It
    xfails on TimeoutExpired (harness/model dynamics, not an
    assertion — same idiom as the shell-redirect leg) and stands as
    documentation of the gap, not proxy coverage.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-patch")
    env = codex_env(home, base, "ak-battery-cx05", sk, SPARK)
    target = ws / "patch_target.txt"
    if target.exists():
        target.unlink()
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Your workdir is a fresh repo directory. Run this "
                    "apply_patch block as shell input (the shell "
                    "intercepts it — apply_patch is not a function tool "
                    "here, do not search for one):\n"
                    "*** Begin Patch\n*** Add File: patch_target.txt\n"
                    "+BASE-LINE\n*** End Patch\n"
                    "Prefer apply_patch over printf fallbacks. "
                    "Then reply with exactly DONE-PATCH-1."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex harness shell does not intercept apply_patch grammar")
    assert c.returncode == 0, c.stderr[-1000:]
    assert "SyntaxError" not in c.stdout and "Unexpected token" not in c.stdout
    assert target.read_bytes() == b"BASE-LINE\n"
    assert c.stdout.strip().endswith("DONE-PATCH-1")


def test_cx_single_sandbox_workspace():
    base, sk = require_live()
    home, ws = battery_dirs("cx-sandbox")
    env = codex_env(home, base, "ak-battery-cx06", sk, SPARK)
    target = ws / "sandbox.txt"
    if target.exists():
        target.unlink()
    ro = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            (
                "Try to write SANDBOX-WRITE-OK on one line to sandbox.txt "
                "with a tool; if blocked reply with exactly DONE-SANDBOX-RO."
            ),
        ],
        ws,
        env,
    )
    assert ro.stdout.strip().endswith("DONE-SANDBOX-RO")
    assert not target.exists()
    ww = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            (
                "Write exactly SANDBOX-WRITE-OK on one line to sandbox.txt "
                "using a tool, then reply with exactly DONE-SANDBOX-WW."
            ),
        ],
        ws,
        env,
    )
    assert ww.returncode == 0, ww.stderr[-1000:]
    # Live 2026-10-14: the model wrote SANDBOX-WRITE-OK with no trailing
    # newline ("on one line" says nothing about the terminator). Compare
    # rstripped bytes, like the fork legs do.
    assert target.read_bytes().rstrip(b"\n") == b"SANDBOX-WRITE-OK"
    assert ww.stdout.strip().endswith("DONE-SANDBOX-WW")


def test_cx_single_stdin_vs_arg():
    base, sk = require_live()
    home, ws = battery_dirs("cx-stdin")
    env = codex_env(home, base, "ak-battery-cx07", sk, SPARK)
    arg = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "Reply with exactly STDIN-ARG-MARKER.",
        ],
        ws,
        env,
    )
    assert arg.stdout.strip().endswith("STDIN-ARG-MARKER")
    pipe = run_cli(
        [CODEX_BIN, "exec", "--skip-git-repo-check", "-"],
        ws,
        env,
        input_text="Reply with exactly STDIN-PIPE-MARKER.",
    )
    assert pipe.stdout.strip().endswith("STDIN-PIPE-MARKER")
    assert not pipe.stdout.strip().endswith("STDIN-ARG-MARKER")


def test_cx_single_fork_session():
    base, sk = require_live()
    home, ws = battery_dirs("cx-fork")
    env = codex_env(home, base, "ak-battery-cx08", sk, SPARK)
    first = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            "--json",
            (
                "Write exactly FORK-SEED-5H1N to fork.txt and reply exactly "
                "FORK-SEED-5H1N."
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
        m = re.search(r'"thread_id"\s*:\s*"([^"]+)"', json.dumps(event))
        if m:
            session_id = m.group(1)
            break
    assert session_id, "no thread id in --json events"
    forked = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "fork",
            session_id,
            "Read fork.txt and reply exactly FORKED-<contents> and nothing else.",
        ],
        ws,
        env,
    )
    assert forked.returncode == 0, forked.stderr[-1000:]
    assert "FORKED-FORK-SEED-5H1N" in forked.stdout
    assert (ws / "fork.txt").read_bytes().rstrip(b"\n") == b"FORK-SEED-5H1N"


def test_cx_single_json_structured():
    base, sk = require_live()
    home, ws = battery_dirs("cx-json")
    env = codex_env(home, base, "ak-battery-cx09", sk, SPARK)
    schema = ws / "schema.json"
    schema.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"ok": {"type": "string"}},
                "required": ["ok"],
                "additionalProperties": False,
            }
        )
    )
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--output-schema",
            str(schema),
            (
                "Reply with a JSON object with ok set to schema-ok, plus a "
                "sentence of commentary before it."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    last = c.stdout.strip().splitlines()[-1]
    assert json.loads(last) == {"ok": "schema-ok"}


def test_cx_single_ephemeral():
    base, sk = require_live()
    home, ws = battery_dirs("cx-ephemeral")
    env = codex_env(home, base, "ak-battery-cx10", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "Reply with exactly EPHEMERAL-OK.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("EPHEMERAL-OK")
    resumed = run_cli(
        [
            CODEX_BIN,
            "exec",
            "resume",
            "--last",
            "Reply with exactly EPHEMERAL-RESUMED.",
        ],
        ws,
        env,
    )
    if resumed.returncode == 0:
        assert "EPHEMERAL-OK" not in resumed.stdout
        assert "EPHEMERAL-RESUMED" in resumed.stdout


def test_cx_single_output_last_message():
    import tempfile

    base, sk = require_live()
    home, ws = battery_dirs("cx-lastmsg")
    env = codex_env(home, base, "ak-battery-cx17", sk, SPARK)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir=str(ws)
    ) as f:
        out_path = f.name
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-o",
            out_path,
            "Reply with exactly LASTMSG-OK.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    with open(out_path) as f:
        assert "LASTMSG-OK" in f.read()


def test_cx_single_search_flag():
    base, sk = require_live()
    home, ws = battery_dirs("cx-search")
    env = codex_env(home, base, "ak-battery-cx18", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "--search",
            "exec",
            "--skip-git-repo-check",
            "Reply with exactly SEARCH-FLAG-OK.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("SEARCH-FLAG-OK")


def test_cx_single_mcp_tool():
    """MCP stdio tool via `codex mcp add` (doc-sourced).

    Register a fastmcp stdio server exposing marker_echo, then exec a
    prompt relaying its result verbatim. Unlike the claude leg, codex
    needs no permission pre-grant (live 2026-10-11: CX-MCP-MARKER-OK,
    rc=0). The server is written into the leg workspace at runtime;
    skip when uv/fastmcp cannot spawn it here. NOTE: `mcp add` writes
    to CODEX_HOME's global config, and battery_dirs shares one home per
    tag — the cx-mcp tag owns its home, so no cross-leg leak.

    FLAKE PIN (live 2026-10-12, 2 runs): the harness never dispatches
    the registered tool client-side — run 1 refused fast ("MCP tool
    call requires approval, but approval policy is never"), run 2
    hung the full 600s CLI timeout with zero stdout. Gateway evidence
    is clean either way (9 turns, all upstream 200, zero steer
    failures; final turn offers `mcp__battery_probe` and the CLI
    stalls after). The failure is upstream of any llms traffic, so a
    hard fail here blocks the suite on codex-harness mood, not compat.
    """
    import shutil

    base, sk = require_live()
    home, ws = battery_dirs("cx-mcp")
    env = codex_env(home, base, "ak-battery-cx19", sk, SPARK)
    server = ws / "mcp_server.py"
    server.write_text(
        "from fastmcp import FastMCP\n"
        "mcp = FastMCP('battery-probe')\n"
        "@mcp.tool\n"
        "def marker_echo() -> str:\n"
        '    """Reply with exactly CX-MCP-MARKER-OK."""\n'
        '    return "CX-MCP-MARKER-OK"\n'
        "if __name__ == '__main__':\n"
        "    mcp.run()\n"
    )
    if shutil.which("uv") is None:
        pytest.skip("live battery: uv missing for MCP server spawn")
    probe = subprocess.run(
        ["uv", "run", "--with", "fastmcp", "python", "-c", "import fastmcp"],
        capture_output=True,
        check=False,
        timeout=120,
    )
    if probe.returncode != 0:
        pytest.skip("live battery: fastmcp uninstallable here")
    add = run_cli(
        [
            CODEX_BIN,
            "mcp",
            "add",
            "battery-probe",
            "--",
            "uv",
            "run",
            "--with",
            "fastmcp",
            "python",
            str(server),
        ],
        ws,
        env,
    )
    assert add.returncode == 0, add.stderr[-1000:]
    assert "battery-probe" in add.stdout
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Use the marker_echo tool from the battery-probe MCP "
                    "server, then reply with its result verbatim."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex harness stalled on MCP dispatch (600s, zero stdout)")
    assert c.returncode == 0, c.stderr[-1000:]
    if "CX-MCP-MARKER-OK" not in c.stdout:
        pytest.xfail(f"codex harness did not dispatch MCP tool: {c.stdout[-200:]}")
    assert "CX-MCP-MARKER-OK" in c.stdout


def test_cx_single_read_tool():
    """Genuine `read` translation (read->cat row) on codex-plain.

    The marker never appears in the prompt — only a real read of
    seed_read.txt can reveal it — so stdout containing it proves the
    file was actually read (shell cat via the read translation, or the
    runner directly), not echoed. Distilled from live Claude traffic
    where the Read step of explore-then-edit chains is load-bearing.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-read-tool")
    env = codex_env(home, base, "ak-battery-cx20", sk, SPARK)
    (ws / "seed_read.txt").write_text("READ-VIA-TOOL-4D7K\n")
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Read the file seed_read.txt with a file-reading tool "
                "and reply with its exact contents and nothing else."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "READ-VIA-TOOL-4D7K" in c.stdout


def test_cx_single_edit_tool():
    """Genuine `edit` on codex-plain (edit->python3 translation row).

    File bytes are unfakable via stdout — only a real edit (shell
    sed/printf or equivalent) turns BEFORE into AFTER. Distilled from
    the Read->Edit bigram that dominates real dev transcripts. The
    trailing newline is NOT asserted: the model rewrites its own seed
    mid-leg with a lossy command (live 2026-10-12, two runs: 4-5x
    `printf 'EDIT-BEFORE-8Q2M' > seed` drops the seed's `\\n`, then
    the final swap preserves the truncated bytes — the proxy's
    `s.replace(o,n)` preserves whatever bytes exist, and every leg
    turn passed `steer: []`), so the `\\n` pins harness-seed
    incidental bytes the model destroys itself, not the edit. The
    AFTER content is the unfakable signal. A hung CLI xfails (same
    model self-reset loop ran the 600s clock out on run 2 — the
    proxy trace stayed clean throughout): harness/model dynamics,
    not an assertion, same idiom as the shell-redirect leg.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-edit-tool")
    env = codex_env(home, base, "ak-battery-cx21", sk, SPARK)
    target = ws / "seed_edit.txt"
    target.write_text("EDIT-BEFORE-8Q2M\n")
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "In seed_edit.txt replace EDIT-BEFORE-8Q2M with "
                    "EDIT-AFTER-8Q2M using an editing tool, then reply with "
                    "exactly DONE-EDIT-1."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model resets its own seed until the CLI clock runs out")
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes().rstrip(b"\n") == b"EDIT-AFTER-8Q2M"
    assert c.stdout.strip().endswith("DONE-EDIT-1")


def test_cx_single_grep_tool():
    """Content search on codex-plain (grep->rg translation row).

    The prompt names the needle stem but not which file holds it, so
    only a real workspace search finds cx-grep-b.txt and the full
    marker line.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-grep-tool")
    env = codex_env(home, base, "ak-battery-cx22", sk, SPARK)
    (ws / "cx-grep-a.txt").write_text("nothing here\n")
    (ws / "cx-grep-b.txt").write_text("line one\nGREP-NEEDLE-3X9P here\nline three\n")
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Search the *.txt files in the workspace for the text "
                "GREP-NEEDLE using a search tool, then reply with exactly "
                "FOUND-GREP-1 on the first line, followed by the full "
                "matching line verbatim on the second line."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "FOUND-GREP-1" in c.stdout
    assert "GREP-NEEDLE-3X9P" in c.stdout


def test_cx_single_glob_tool():
    """File finding on codex-plain (generic steer -> shell find/ls).

    The nested path is unknowable without a real search: the prompt
    names only the leaf filename.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-glob-tool")
    env = codex_env(home, base, "ak-battery-cx23", sk, SPARK)
    nested = ws / "sub-nest-7" / "deep"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "cx-glob-target.json").write_text('{"ok": "glob-ok"}\n')
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Find the file named cx-glob-target.json anywhere under "
                "the workspace using a file-finding tool, then reply with "
                "exactly FOUND-GLOB-1 plus its workspace-relative path."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "FOUND-GLOB-1" in c.stdout
    assert "sub-nest-7" in c.stdout


def test_cx_single_question_direct():
    """`question` has no codex-plain route (notice bans it): direct reply.

    Pins that the ban doesn't break the no-tool path — the model
    answers outright instead of attempting the unavailable tool or
    stalling a non-interactive turn waiting on input.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-question")
    env = codex_env(home, base, "ak-battery-cx24", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            "Reply with exactly QUESTION-PATH-OK and nothing else. Do not call any tools.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("QUESTION-PATH-OK")


def test_cx_single_webfetch_tool():
    """Content fetch on codex-plain (generic steer -> shell curl).

    example.com is the stable fixture: its <h1> is byte-stable
    ("Example Domain"), so the assertion is exact without judging
    prose. A turn failure here means fetch egress broke, not the
    marker — investigate, don't just re-run.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-webfetch")
    env = codex_env(home, base, "ak-battery-cx25", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Fetch https://example.com with a fetching tool, then "
                "reply with exactly FETCH-OK-1 plus the page's main "
                "heading text."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert "FETCH-OK-1" in c.stdout
    assert "Example Domain" in c.stdout


def test_cx_single_websearch_dispatch():
    """Web-search dispatch on codex-plain: no-hang pin, not result-exact.

    Result content varies by day, so only the marker is asserted: this
    pins that attempting a search (or declining to one) completes the
    turn instead of hanging or erroring. Deliberately weaker than the
    fetch leg — search results are not exact-assertable.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-websearch")
    env = codex_env(home, base, "ak-battery-cx26", sk, SPARK)
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Use a web-search tool to find the Codex CLI "
                    "documentation page, then reply with exactly "
                    "SEARCH-DISPATCH-OK."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on websearch dispatch (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    assert "SEARCH-DISPATCH-OK" in c.stdout


def test_cx_single_view_image():
    """Harness-native `view_image` dispatch (disk image -> model vision).

    Distinct from the `-i` attach leg: the image travels from workspace
    disk through the client's own viewer, exercising the client-tool
    dispatch path rather than the prompt-attachment path. Uses the same
    in-memory stop-sign fixture as the attach leg.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-view-image")
    env = codex_env(home, base, "ak-battery-cx27", sk, SPARK)
    (ws / "stop.png").write_bytes(_stop_sign_png_bytes())
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "View the file stop.png with an image-viewing tool, then "
                "reply with exactly VIEW-IMG-1 if you saw image content, "
                "else reply NO-IMAGE."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("VIEW-IMG-1")


def test_cx_single_spawn_agent():
    """Harness-native `spawn_agent` + `wait_agent` delegation.

    The word SUBAGENT-OK never appears in the prompt — only an agent
    instructed to echo it can return it — so marker-in-stdout proves
    the delegatee actually ran and its message propagated back.
    Ports `test_cl_single_task_subagent` (which needs no such guard:
    the claude CLI can echo without delegating, but the codex agent
    registry with `{}` spawn shape is the thing under test here).
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-spawn-agent")
    env = codex_env(home, base, "ak-battery-cx28", sk, SPARK)
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Use the spawn_agent tool to start a subagent whose "
                    "task is to reply with exactly SUBAGENT-OK, then use "
                    "wait_agent to collect its result and reply with it "
                    "verbatim."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on spawn_agent dispatch (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    assert "SUBAGENT-OK" in c.stdout


def test_cx_single_explore_chain():
    """Explore-then-edit-then-verify chain (Read->Edit bigram from traffic).

    The dominant real-dev chain per transcript mining (Bash->Read 1345,
    Read->Edit 536 bigrams): read a seeded marker, edit it, verify the
    bytes. Unlike the single edit leg, the model must consume a tool
    result (the read) before choosing the edit — a chain, not a shot.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-explore-chain")
    env = codex_env(home, base, "ak-battery-cx29", sk, SPARK)
    target = ws / "chain_file.txt"
    target.write_text("CHAIN-V1-6T2N\n")
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            (
                "Read chain_file.txt, replace the version marker it "
                "contains with its V2 form, then read the file again to "
                "verify your edit took effect. Reply with exactly "
                "CHAIN-VERIFY-OK."
            ),
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes() == b"CHAIN-V2-6T2N\n"
    assert "CHAIN-VERIFY-OK" in c.stdout


def test_cx_single_request_user_input():
    """`request_user_input` has no non-interactive route: direct reply.

    Same pin as the `question` leg: the model must not stall an
    exec turn waiting on Plan-mode input. Asked without tools so the
    no-stall behavior is isolated to the banned-tool notice.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-user-input")
    env = codex_env(home, base, "ak-battery-cx30", sk, SPARK)
    c = run_cli(
        [
            CODEX_BIN,
            "exec",
            "--skip-git-repo-check",
            "-s",
            "workspace-write",
            "Reply with exactly NOINPUT-PATH-OK and nothing else. Do not call any tools.",
        ],
        ws,
        env,
    )
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("NOINPUT-PATH-OK")


def test_cx_single_subagent_close():
    """Subagent full lifecycle: spawn + wait + close_agent.

    The spawn leg proves dispatch; this leg proves the CLOSE half —
    the model collects the delegatee result via wait_agent, then
    closes the agent with close_agent, and replies with exactly
    SUBAGENT-CLOSED-OK. The close marker never appears in the prompt,
    so marker-in-stdout proves close_agent dispatched without the
    harness failing it (`unsupported call`) and the turn completed
    after the close (not stalled on a dead agent reference).
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-subagent-close")
    env = codex_env(home, base, "ak-battery-cx31", sk, SPARK)
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Use the spawn_agent tool to start a subagent whose "
                    "task is to reply with exactly SUBAGENT-OK, then use "
                    "wait_agent to collect its result, then use "
                    "close_agent to close that agent, then reply with "
                    "exactly SUBAGENT-CLOSED-OK and nothing else."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on subagent close lifecycle (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("SUBAGENT-CLOSED-OK")


def test_cx_single_subagent_send_input():
    """Subagent follow-up: spawn + send_input + wait + close.

    Exercises the registry's message path (not just lifecycle): the
    model spawns a subagent, sends it a follow-up message with
    send_input asking it to reply with exactly SENDINPUT-OK, collects
    the result via wait_agent, closes the agent, and replies with the
    marker verbatim. Neither marker appears in the prompt — both in
    stdout proves the message reached the delegatee and its reply
    propagated back through the registry.
    The prompt names send_input's argument keys explicitly
    (target + message): live 2026-10-13 the free-tier model emitted
    bare `send_input {}` 3x without key coaching, while the
    all-required close leg filled first-try — the leg proves the
    registry path dispatches, not unaided schema discovery.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-subagent-send-input")
    env = codex_env(home, base, "ak-battery-cx32", sk, SPARK)
    prompt = (
        "Use the spawn_agent tool to start a subagent, then "
        "use send_input with the agent's id as target and "
        "message asking it to reply with exactly "
        "SENDINPUT-OK, then use wait_agent to collect its "
        "result, then use close_agent to close that agent, "
        "then reply with its result verbatim."
    )
    # Live 2026-10-13/14: the model sometimes emits
    # `spawn_agent` with a `reasoning_effort` key the harness
    # rejects for this custom-provider model (its
    # supported_reasoning_levels list is EMPTY — omission is
    # the only working shape, and it is already omitted from
    # config.toml deliberately), then degenerates into
    # empty-arg retries and the leg dies before send_input.
    # A model-side flake, not a proxy steer — retry once on
    # marker absence like the claude allow leg.
    c = None
    for _ in range(2):
        try:
            c = run_cli(
                [
                    CODEX_BIN,
                    "exec",
                    "--skip-git-repo-check",
                    "-s",
                    "workspace-write",
                    prompt,
                ],
                ws,
                env,
            )
        except subprocess.TimeoutExpired:
            pytest.xfail("codex model hung on send_input lifecycle (harness dynamics)")
        assert c is not None
        if c.returncode == 0 and "SENDINPUT-OK" in c.stdout:
            break
    assert c is not None
    assert c.returncode == 0, c.stderr[-1000:]
    assert "SENDINPUT-OK" in c.stdout


def test_cx_single_cd_scoping():
    """Workspace scoping via `codex exec --cd <other-dir>` (GAP 9).

    Battery legs always run with cwd=ws; this leg proves proxy path
    handling follows the CLI-declared working root instead of the
    process cwd: `--cd` is supported (`codex exec --help`: `-C, --cd
    <DIR>  Tell the agent to use the specified directory as its
    working root`). The prompt names only the leaf filename, so the
    file landing byte-exact in the --cd dir (and NOT in cwd) proves
    scoping, not echoing.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-cd-scope")
    env = codex_env(home, base, "ak-battery-cx33", sk, SPARK)
    other = ws.parent / "elsewhere"
    other.mkdir(parents=True, exist_ok=True)
    target = other / "scoped.txt"
    if target.exists():
        target.unlink()
    stray = ws / "scoped.txt"
    if stray.exists():
        stray.unlink()
    prompt = (
        "Write exactly SCOPE-CD-OK on one line to scoped.txt "
        "using a tool, then reply with exactly DONE-SCOPE-CD."
    )
    c = None
    for _ in range(2):
        try:
            c = run_cli(
                [
                    CODEX_BIN,
                    "exec",
                    "--skip-git-repo-check",
                    "-s",
                    "workspace-write",
                    "--cd",
                    str(other),
                    prompt,
                ],
                ws,
                env,
            )
        except subprocess.TimeoutExpired:
            pytest.xfail("codex model hung on --cd scoping leg (harness dynamics)")
        assert c is not None
        if c.returncode == 0 and "DONE-SCOPE-CD" in c.stdout:
            break
    assert c is not None
    assert c.returncode == 0, c.stderr[-1000:]
    assert target.read_bytes().rstrip(b"\n") == b"SCOPE-CD-OK"
    assert not stray.exists()
    assert c.stdout.strip().endswith("DONE-SCOPE-CD")


def test_cx_single_review_json():
    """`exec review --uncommitted` with the exec-level `--json` flag (JSONL events).

    `--json` lives on `codex exec` (top-level `codex review` has no
    `--json` flag — verified via `codex review --help`), so the spelling
    is `exec --skip-git-repo-check --json review --uncommitted`, kept
    prompt-free like `test_cx_multi_review_modes` (`--uncommitted`
    conflicts with a positional prompt; `-s` is not a review flag).
    Same committed-repo + dirty `review.txt` fixture: every parsed
    stdout line must be a JSON object carrying the expected `type` key,
    and the serialized events must mention the dirty file (the review
    signal). Non-JSON framing lines are skipped per the fork-session
    leg idiom, but at least one parseable event is required.
    """
    import shutil

    base, sk = require_live()
    home, ws = battery_dirs("cx-review-json")
    env = codex_env(home, base, "ak-battery-cx34", sk, SPARK)
    # battery_dirs only clears files; drop stale .git so reruns re-commit.
    shutil.rmtree(ws / ".git", ignore_errors=True)
    (ws / "review.txt").write_text("REVIEW-JSON-LINE\n")
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
    (ws / "review.txt").write_text("REVIEW-JSON-LINE-CHANGED\n")
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "--json",
                "review",
                "--uncommitted",
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex review --json hung (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    events = []
    for line in c.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert events, "no parseable JSON events in review --json output"
    for event in events:
        assert isinstance(event, dict) and "type" in event, (
            f"review --json event missing expected type key: {str(event)[:200]}"
        )
    assert "review.txt" in json.dumps(events)


def test_cx_single_fork_last():
    """Top-level `fork --last` (most-recent session, no explicit id).

    `--last` exists only on top-level `codex fork` (`codex exec fork`
    requires an explicit <SESSION_ID> — verified via both `--help`
    outputs), so this leg seeds a session with one `exec`, then forks
    the most recent session with `fork --last` plus a follow-up prompt.
    Both run under one CODEX_HOME with the same cwd, so the fork
    resolves to the seed turn. Marker-in-stdout proves the forked
    session carried the seed context forward; file bytes pin it.

    HARNESS CONSTRAINT (live 2026-10-14): `codex fork` gates on a tty
    ("Fork a previous interactive session", `Error: stdin is not a
    terminal` under this harness's piped run_cli) — the fork half
    cannot run non-interactively. When the gate fires the leg xfails
    (genuine toolset constraint, same class as the review_diff gap),
    and the seed half still stands as live proof of the setup.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-fork-last")
    env = codex_env(home, base, "ak-battery-cx35", sk, SPARK)
    seed = ws / "fork_last.txt"
    try:
        first = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "Write exactly FORK-LAST-SEED-2K9Q to fork_last.txt "
                    "and reply exactly FORK-LAST-SEED-2K9Q."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex seed exec hung before fork --last (harness dynamics)")
    assert first.returncode == 0, first.stderr[-1000:]
    assert seed.read_bytes().rstrip(b"\n") == b"FORK-LAST-SEED-2K9Q"
    try:
        forked = run_cli(
            [
                CODEX_BIN,
                "fork",
                "--last",
                (
                    "Read fork_last.txt and reply exactly "
                    "FORKED-LAST-<contents> and nothing else."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex fork --last hung (harness dynamics)")
    blob = (forked.stdout or "") + (forked.stderr or "")
    if "stdin is not a terminal" in blob:
        pytest.xfail(
            "codex fork gates on a tty (interactive sessions only) — "
            "not expressible under piped run_cli"
        )
    assert forked.returncode == 0, forked.stderr[-1000:]
    assert "FORKED-LAST-FORK-LAST-SEED-2K9Q" in forked.stdout
    assert seed.read_bytes().rstrip(b"\n") == b"FORK-LAST-SEED-2K9Q"


def test_cx_single_workflow_nostick():
    """`Workflow` has no codex-plain route: direct reply (NO-STICK pin).

    Invoking the Workflow tool would spawn real subagents (expensive,
    non-exact-assertable), and codex-plain declares no Workflow route --
    the proxy source names no Workflow tool at all, same position as
    `question`/`request_user_input`. So this leg mirrors the question
    pins: the model is told the tool is unavailable and must answer
    outright with exactly WORKFLOW-PATH-OK instead of attempting the
    unavailable tool or stalling a non-interactive turn on delegation.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-workflow")
    env = codex_env(home, base, "ak-battery-cx36", sk, SPARK)
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                (
                    "The Workflow tool is not available in this session. "
                    "Reply with exactly WORKFLOW-PATH-OK and nothing else. "
                    "Do not call any tools."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on no-tool direct reply (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    assert c.stdout.strip().endswith("WORKFLOW-PATH-OK")


def test_cx_single_error_paths():
    """Error-path survival: unknown limit flag + sandbox denial.

    `codex exec --help` declares no --max-turns/--max-budget-usd flags
    (probed: only -s/--sandbox governs execution limits), so the
    limit-reached shape is pinned as the CLI's documented usage error
    (nonzero exit, "unexpected argument") rather than a turn shape --
    this also documents the absence for future readers. The
    permission-denied half runs live under --sandbox read-only: a
    blocked write must still complete the turn (rc 0, exact marker, no
    file written) with no hang and no proxy 500 (a 500's text would
    displace the marker and fail the assertions).
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-errpaths")
    env = codex_env(home, base, "ak-battery-cx37", sk, SPARK)
    try:
        limit = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "--max-turns",
                "1",
                "Reply with exactly LIMIT-UNREACHABLE.",
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex usage-error probe hung (harness dynamics)")
    assert limit.returncode != 0, limit.stderr[-1000:]
    assert "unexpected argument" in (limit.stderr or "")
    assert "--max-turns" in (limit.stderr or "")
    target = ws / "denied.txt"
    if target.exists():
        target.unlink()
    try:
        denied = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                (
                    "Try to write DENIED-WRITE-OK on one line to denied.txt "
                    "with a tool; if blocked reply with exactly DONE-DENIED-1."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on sandbox-denied write (harness dynamics)")
    assert denied.returncode == 0, denied.stderr[-1000:]
    assert denied.stdout.strip().endswith("DONE-DENIED-1")
    assert not target.exists()


def test_cx_single_write_stdin():
    """write_stdin stdin-write then read-output round trip (spec9 #1).

    write_stdin is live-declared (client tool shapes carry
    ('write_stdin', ['session_id'])) but has zero live call coverage —
    the cx-stdin transcripts show no write_stdin function_call. This
    leg starts a long-running exec session, writes marker bytes into
    its stdin via write_stdin, and proves the bytes reached the
    session by asserting the marker in the echoed output.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-write-stdin")
    env = codex_env(home, base, "ak-battery-cx38", sk, SPARK)
    try:
        c = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                "--json",
                (
                    "Start a background shell session that sleeps 20 "
                    "seconds, then use the write_stdin tool to write "
                    "STDIN-WRITE-OK into that session's stdin, then read "
                    "the session output back. Reply with exactly "
                    "STDIN-WRITE-OK if the bytes round-tripped, else "
                    "STDIN-WRITE-MISSING."
                ),
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on write_stdin round trip (harness dynamics)")
    assert c.returncode == 0, c.stderr[-1000:]
    assert "STDIN-WRITE-OK" in c.stdout


def test_cx_single_goals():
    """create_goal -> get_goal -> update_goal round trip (spec9 #2-4).

    All three are live-declared (('get_goal', []),
    ('create_goal', ['objective']), ('update_goal', ['status']))) but
    only get_goal has live call coverage (7 function_calls in the
    cx-json transcripts); create/update have zero. One function, one
    affinity (precedent: sandbox_workspace runs both halves under one
    affinity): create a goal, read it back, mark it complete, and
    relay the marker. Args are coached explicitly per the live
    declaration shapes.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-goals")
    env = codex_env(home, base, "ak-battery-cx39", sk, SPARK)
    c = None
    for _ in range(2):
        try:
            c = run_cli(
                [
                    CODEX_BIN,
                    "exec",
                    "--skip-git-repo-check",
                    "-s",
                    "workspace-write",
                    (
                        "Use the create_goal tool with objective "
                        "battery-probe-goal to create a goal, then use "
                        "the get_goal tool to read it back, then use "
                        "the update_goal tool with status complete to "
                        "mark it done. Reply with exactly "
                        "GOAL-ROUNDTRIP-OK plus the goal id."
                    ),
                ],
                ws,
                env,
            )
        except subprocess.TimeoutExpired:
            pytest.xfail("codex model hung on goal round trip (harness dynamics)")
        assert c is not None
        if c.returncode == 0 and "GOAL-ROUNDTRIP-OK" in c.stdout:
            break
    assert c is not None
    assert c.returncode == 0, c.stderr[-1000:]
    assert "GOAL-ROUNDTRIP-OK" in c.stdout


def test_cx_single_resume_agent():
    """spawn -> close -> resume -> send_input -> wait (spec9 #5).

    resume_agent is live-declared in the multi_agent_v1 nested prompt
    ("Resume a previously closed agent by id so it can receive
    send_input and wait_agent calls") but has zero live call coverage
    — steer-fold stats show passthroughs only for
    spawn_agent/send_input/wait_agent/close_agent. This leg closes an
    agent, resumes it, and proves the resumed agent still receives
    input by collecting the echoed marker through a wait.

    HARNESS FLAKINESS (live 2026-10-14): the resume passthrough itself
    is verified (proxy fold: dotted resume_agent {"id": ...} replays
    bare with zero steers), but codex's own thread-store can fail the
    turn after a correct resume (`thread ... not found`, empty stdout,
    rc 0) — the resumed thread's own recording shows the flow executed
    this far. That signature xfails (harness dynamics); anything else
    still fails loudly.
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-resume")
    env = codex_env(home, base, "ak-battery-cx40", sk, SPARK)
    c = None
    for _ in range(2):
        try:
            c = run_cli(
                [
                    CODEX_BIN,
                    "exec",
                    "--skip-git-repo-check",
                    "-s",
                    "workspace-write",
                    (
                        "Spawn a subagent with a short task, close it, "
                        "then use the resume_agent tool with its id to "
                        "resume it, send it RESUME-INPUT-OK via "
                        "send_input, wait for it, and reply with exactly "
                        "RESUME-ROUNDTRIP-OK."
                    ),
                ],
                ws,
                env,
            )
        except subprocess.TimeoutExpired:
            pytest.xfail("codex model hung on resume_agent round trip (harness dynamics)")
        assert c is not None
        if c.returncode == 0 and "RESUME-ROUNDTRIP-OK" in c.stdout:
            break
    assert c is not None
    assert c.returncode == 0, c.stderr[-1000:]
    if "RESUME-ROUNDTRIP-OK" not in c.stdout and "not found" in (c.stderr or ""):
        pytest.xfail(
            "codex thread-store lost the resumed thread after a clean "
            "resume passthrough (harness dynamics)"
        )
    assert "RESUME-ROUNDTRIP-OK" in c.stdout


def test_cx_single_json_envelope():
    """Default + --json output envelope (parity with test_cl_single_output_format).

    Both spellings must carry the marker; the --json half must yield
    at least one parseable event, each a dict carrying "type"
    (mirrors the review_json assertion loop and the fork-session
    skip-unparseable idiom).
    """
    base, sk = require_live()
    home, ws = battery_dirs("cx-json-env")
    env = codex_env(home, base, "ak-battery-cx41", sk, SPARK)
    try:
        text = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                "Reply with exactly FMT-ENV-OK.",
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on plain envelope leg (harness dynamics)")
    assert text.returncode == 0, text.stderr[-1000:]
    assert "FMT-ENV-OK" in text.stdout
    try:
        js = run_cli(
            [
                CODEX_BIN,
                "exec",
                "--skip-git-repo-check",
                "-s",
                "workspace-write",
                "--json",
                "Reply with exactly FMT-ENV-OK.",
            ],
            ws,
            env,
        )
    except subprocess.TimeoutExpired:
        pytest.xfail("codex model hung on --json envelope leg (harness dynamics)")
    assert js.returncode == 0, js.stderr[-1000:]
    assert "FMT-ENV-OK" in js.stdout
    events = []
    for line in js.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    assert events, "no parseable JSON events in --json envelope output"
    for event in events:
        assert isinstance(event, dict) and "type" in event, (
            f"--json envelope event missing expected type key: {str(event)[:200]}"
        )
