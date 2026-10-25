# Live CLI battery — spec index

Model-independent live tests for the Codex (`codex/`) and Claude Code
(`claude/`) CLIs driven through the llms gateway. Every test asserts an
exact, model-agnostic outcome (byte-exact file bytes, exact echoed
markers, exact arithmetic) — never judged prose.

## Layout

- `battery/__init__.py` — shared harness (`require_live`, per-leg
  `codex_env` / `claude_env`, `run_cli`, isolated `battery_dirs`).
- `battery/codex/` — responses-leg features (spark model).
  `README.md` there lists the single-turn specs, `multi_turn.md` the
  multi-turn specs.
- `battery/claude/` — messages-leg features (CLI-native haiku).
  Same README split.

## Running

Not collected by the top-level suite (`testpaths = ["tests"]`), and
every test skips unless live runs are explicitly enabled:

Run: `just battery codex/` (or `claude/`, or `battery/` for both).
Without `LLMS_BATTERY_LIVE=1`, every test skips — pointing pytest at
this directory by accident is always a no-op.

Each test uses a distinct `ak-batteryN` affinity key so gateway buckets
spread legs across pools. Expect codex legs to be flakier than claude
legs — that asymmetry is the point (codex compat debt), not noise.

## Failure vs skip

`run_cli` retries genuine `429 Too Many Requests` storms
(`BATTERY_RETRY_429_ATTEMPTS`, default 5, spaced
`BATTERY_RETRY_429_DELAY_S`, default 5s). Two 429s are never retried:
the anonymous shape rejection, and the daily free-tier cap
(`FreeUsageLimitError`, `retry-after` in hours, resetting at UTC
midnight). A leg that hits the spent quota is **skipped** with a hint —
a spent quota says nothing about the gateway, so failing would report a
regression that does not exist.

Legs live under `/tmp/llms-battery-<tag>` (a 3.9G tmpfs here) and each
CLI accumulates ~100–300M of rollout sessions per leg. `battery_dirs`
prunes the starting leg's sessions and any other leg's sessions idle for
`BATTERY_STALE_SESSIONS_AGE_S` (default 900s), so repeated passes cannot
fill `/tmp` and fail with ENOSPC.

## Single-turn specs

Per-CLI tables live in `battery/codex/README.md` and
`battery/claude/README.md` (12 codex + 12 claude single-turn specs).

## Multi-turn specs

See `battery/codex/multi_turn.md` (5 specs) and
`battery/claude/multi_turn.md` (4 specs).

## Status

Spec list + prompts designed by workflow `w8x022edv`
(enumerate 26 → design 26 → adversarial verify: 24 kept, 2 dropped as
vacuous — `cx-single-json-output` and `cl-followup-tool-output`, fixes
noted in the per-CLI READMEs), then extended with 10 doc-sourced specs
from `codex --help` / `claude --help` ground truth (fork, output-schema
violation-tempted, ephemeral, -o, search, review --uncommitted/--json,
json-schema, partial-messages, session-id/name, fork-session), resolving
the dropped cx schema test. Files landed; not yet run live.
`just battery codex/` / `just battery claude/` when ready.
