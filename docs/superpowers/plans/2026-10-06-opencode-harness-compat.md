# Opencode↔Harness Tool Compatibility Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate between the opencode-12 tool set offered upstream and each harness family's native tools in both directions, so the model is always offered names with real wire schemas and every emission dispatches to something the harness can execute.

**Architecture:** New pure module `llms/proxy/compat.py` holds two explicit tables (genuine→client for dispatch rewrite, client→genuine for history-echo normalization) plus a family detector (UA first, leg+tool-shape fallback). Both steer paths already share one classifier (`pipeline._classify_calls`, used by the streaming fold in `forward.py`), so wiring the dispatch half into the classifier covers streaming and non-streaming at once. Untranslatable names fail open (pass through unchanged) with a `logger.error` breadcrumb for later table fixes.

**Tech Stack:** Python, FastAPI/Starlette (proxy), pytest (hermetic), `scripts/` live probes against Zen free tier.

**Spec:** User direction in conversation 2026-10-06 (full 1:1 compat, UA-based families with leg fallback, fail-open + error logs, extensive live tests, schema audit afterwards) plus follow-up (codex `apply_patch`→`edit`/`write` upstream on follow-up turns; Claude `Read`→`read` upstream). Background: `docs/superpowers/plans/2026-10-05-codex-tool-compat-rca-DRAFT.md`.

## Global Constraints

- Outbound anonymous responses-leg tools stay genuine-12 ONLY, byte-identical, in order — never add a 13th tool (live 400 on `tools[12].description` length). Compat never adds tools outbound; it renames history/calls and rewrites arguments.
- Genuine opencode (`User-Agent: opencode/*`) keeps exact fidelity: no compat renames, no notice (existing `is_genuine_opencode` gate stays).
- Client credentials (`sk-` secrets, harness dummy keys) never reach upstream; only operator `ZEN_API_KEY` or anonymous `Bearer public`.
- Steer loop stays bounded (`STEER_MAX_ITERS=3`, fail-closed terminal). Compat rewrites must REDUCE steers, never add a steer cycle: a compat rewrite that fails validation steers once with the existing correction, never loops.
- Fail open on untranslatable: pass through unchanged + `logger.error`, never synthesize a guessed call, never 400 a session.
- Probes live in `scripts/` (externals-dependent); hermetic tests in `tests/`.
- Failing hermetic test first for every new translation (systematic-debugging Phase 4); commit only after live probe verification + green suite, message ends with `Co-Authored-By: Claude Code <noreply@anthropic.com>`.
- `ak-` path affinity is unauth routing, `sk-` header is auth (unchanged).

## Review Focus

- A client tool sharing a key is not the capability (`read`/`write` renamed onto `view_image` failed live twice): every table entry must prove capability-equivalence, not key-overlap.
- `printf`/`cat` shell fallbacks mangle binary content, newlines, and quotes: shell-fallback entries must quote via `shlex` (existing `_read_via_shell_redirect` pattern) and declare the binary-content exclusion.
- Patch-text synthesis is lossy if the marker grammar is guessed: no `write`→`apply_patch` entry until the Add-File/Update/Delete grammar is verified against the harness (Task 3 blocks Task 5).
- A bare `function_call apply_patch` on the luna leg fails lookup (only `custom_tool_call exec` dispatches): every luna-bound rewrite must ride `__exec_rewrite__`, never a bare rename.
- History echo of a translated call must carry the translated name/args (existing rule): the reverse table must apply at echo time or the model re-emits dead names.
- Family misdetect must degrade to today's behavior, never to a wrong family's rewrite: unknown family = `*` entries only (shell→cmd-shape), family-specific entries require positive detection.

---

## File Structure

- **Create `llms/proxy/compat.py`** — pure helpers, no pipeline/forward imports (same discipline as `client_tools.py`): `HarnessFamily` constants, `detect_family()`, `TO_CLIENT` dispatch table + `translate_to_client()`, `TO_GENUINE` history table + `translate_to_genuine()`, per-pair arg translators, `log_untranslatable()`.
- **Modify `llms/proxy/pipeline.py`** — `run()` detects family (headers + ingress + tools) and threads it to `to_zen_responses` (history echo), `_classify_calls` (dispatch rewrite), and `forward()` (streaming fold).
- **Modify `llms/proxy/forward.py`** — `forward()` / `fold_and_steer_streaming()` accept `family`; fold's `_classify` call passes it; untranslatable fold outcomes `logger.error`.
- **Modify `llms/proxy/translate.py`** — `to_zen_responses()` accepts `family` and applies the client→genuine history normalization at the existing `_translate` echo site.
- **Modify `llms/proxy/client_tools.py`** — extend `translate_genuine_call`/`_translate_genuine_args` ONLY via compat table lookup (no behavior change for pairs the table doesn't cover; the `view_image` trap tests keep passing).
- **Tests:** fill `tests/test_compat_table.py` (created, docstring-only placeholder); extend `tests/test_pipeline.py`, `tests/test_proxy.py` for new rewrites + fail-open logging.
- **Probes:** create `scripts/probes/compat_matrix_probe.py` (per-leg × per-pair live matrix, byte-exact assertions).
- **Docs:** finalize `docs/superpowers/plans/2026-10-05-codex-tool-compat-rca-DRAFT.md` (rename without `-DRAFT`, add compat mechanisms + schema audit appendix).

## Family + tool-shape inventory (verified live 2026-10-05/06)

| Leg | Ingress | Client declares | Native file tools? |
|---|---|---|---|
| codex-plain (spark) | responses, top-level `tools` | `exec_command{cmd}`, `write_stdin`, `request_user_input`, `view_image{path}`, `multi_agent_v1`, goals, `web_search` | none — files via `exec_command` only; no `apply_patch` declared |
| luna-deferred | responses, `additional_tools` ns `functions` | custom `exec` (nested: `exec_command`, `apply_patch` freeform, goals, `view_image`, `write_stdin`) + `wait`, `request_user_input` | `apply_patch` nested, only via `custom_tool_call exec` channel |
| claude | messages, capitalized tools | `Read`/`Edit`/`Write`/… (shapes need capture — Task 1) | yes, native |
| dsh | chat (assumed; verify Task 1) | shapes need capture | unknown |

Downstream UA is NOT currently captured (relay stores body only; `run()` reads UA only for the genuine check) — Task 1 verifies what's actually set.

---

### Task 1: Family detection — verify UA, inventory shapes, detector + tests

**Files:**
- Modify: `llms/proxy/pipeline.py` (log downstream UA at ingress, one debug line)
- Create: `llms/proxy/compat.py` (`detect_family` only)
- Test: `tests/test_compat_table.py` (detector cases)

**Interfaces:**
- Consumes: nothing new.
- Produces: `detect_family(user_agent: str | None, ingress: str, tools: tuple[ToolDef, ...]) -> str` returning `"codex-plain" | "luna" | "claude" | "dsh" | "unknown"`, used by Tasks 4–5.

- [ ] **Step 1: Log downstream UA at ingress**

```python
logger.debug("[%s] ingress UA=%r ingress=%s", trace_id, request.headers.get("user-agent", ""), ingress)
```

Place next to the existing `log_ingress` call in `run()` (`pipeline.py`, near the `log_ingress(trace_id, ...)` site).

- [ ] **Step 2: Run one probe per family, record actual UA strings**

Run: `cd /home/uqmm/vc1/src/llms && PROBE_PORT=8793 .venv/bin/python scripts/codex_probe.py` (codex leg; check `scripts/claude_probe.py` and `scripts/dsh_headless_probe.py` exist and run the same way for their legs)
Expected: proxy log shows the downstream UA per family. Record the verbatim strings in the RCA draft's inventory table. If a leg sends no UA or a generic one, note that — the detector falls back to leg+shape.

- [ ] **Step 3: Write failing detector tests**

```python
def test_detect_family_prefers_ua_falls_back_to_shape():
    from llms.proxy.compat import detect_family
    # UA wins when present (verbatim strings from Step 2).
    assert detect_family("codex-cli/0.154.0", "responses", ()) == "codex-plain"
    # No UA: responses + additional_tools functions namespace -> luna.
    assert detect_family(None, "responses", (_luna_namespace(),)) == "luna"
    # No UA: responses + top-level function tools -> codex-plain.
    assert detect_family(None, "responses", (EXEC_COMMAND,)) == "codex-plain"
    # No UA: messages leg with capitalized tools -> claude.
    assert detect_family(None, "messages", (READ,)) == "claude"
    # Nothing recognizable -> unknown (today's behavior, *-only entries).
    assert detect_family(None, "chat", ()) == "unknown"
```

(Use the real UA strings observed in Step 2, not the placeholders above. Import `_luna_namespace` from `tests/test_client_tools.py` or redefine a minimal namespace ToolDef locally.)

- [ ] **Step 4: Run to verify they fail**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python -m pytest tests/test_compat_table.py -q`
Expected: FAIL (no `compat` module yet).

- [ ] **Step 5: Implement `detect_family` only**

```python
"""Opencode<->harness tool compatibility tables (pure: no pipeline/forward imports)."""
from __future__ import annotations

CODEX_PLAIN = "codex-plain"
LUNA = "luna"
CLAUDE = "claude"
DSH = "dsh"
UNKNOWN = "unknown"

def detect_family(user_agent, ingress, tools) -> str:
    ua = (user_agent or "").lower()
    if ua.startswith("codex/") or "codex-cli" in ua:
        return CODEX_PLAIN
    if ua.startswith("claude/") or "claude-cli" in ua or "claude-code" in ua:
        return CLAUDE
    if "dsh" in ua:
        return DSH
    if ingress == "messages":
        return CLAUDE
    if ingress == "chat":
        return DSH
    from llms.proxy.client_tools import _namespace_exec_desc
    for t in tools or ():
        if _namespace_exec_desc(t):
            return LUNA
    if any(getattr(t, "name", "") for t in (tools or ())):
        return CODEX_PLAIN
    return UNKNOWN
```

(Tune UA prefixes to the verbatim strings from Step 2.)

- [ ] **Step 6: Run to verify pass**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python -m pytest tests/test_compat_table.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/llms/llms/proxy/compat.py src/llms/tests/test_compat_table.py src/llms/llms/proxy/pipeline.py
git commit -m "feat: harness family detection (UA first, leg+shape fallback)

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### Task 2: Dispatch half — `translate_to_client` + table (`*` entries first)

**Files:**
- Modify: `llms/proxy/compat.py` (table + `translate_to_client`)
- Test: `tests/test_compat_table.py`

**Interfaces:**
- Consumes: `detect_family` (Task 1).
- Produces: `translate_to_client(genuine_name: str, arguments: str, family: str, owned: dict, defs: dict) -> tuple[str, str] | None` — `(client_name, client_arguments)` or `None` (untranslatable → caller fails open + logs). Used by Task 4's classifier wiring.

- [ ] **Step 1: Write failing tests for the `*` (all-family) entries**

```python
def test_dispatch_shell_to_cmd_runner_all_families():
    # shell{"command"} -> exec_command{"cmd"} on every family that owns
    # a cmd-shaped runner (live spark leg, zero-steer turn 2026-10-06).
    from llms.proxy.compat import translate_to_client
    owned = {"exec_command": "exec_command"}
    defs = {"exec_command": EXEC_COMMAND_DEF}  # required == ["cmd"]
    assert translate_to_client("shell", '{"command": "cat /tmp/hi"}', "codex-plain", owned, defs) == (
        "exec_command", '{"cmd": "cat /tmp/hi"}')
    assert translate_to_client("shell", '{"command": "cat /tmp/hi"}', "unknown", owned, defs) == (
        "exec_command", '{"cmd": "cat /tmp/hi"}')
    # Missing payload key or no cmd-runner: None (fail open, never guess).
    assert translate_to_client("shell", '{"workdir": "/tmp"}', "codex-plain", owned, defs) is None
    assert translate_to_client("shell", '{"command": "x"}', "codex-plain", {}, {}) is None
    # Same-name ownership wins (existing rule): client declared `shell`
    # itself -> None so the owned argument-correction path applies.
    assert translate_to_client("shell", '{"command": "x"}', "codex-plain",
                               {"shell": "Shell"}, {"shell": SHELL_DEF}) is None
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python -m pytest tests/test_compat_table.py -q`
Expected: FAIL (`translate_to_client` missing).

- [ ] **Step 3: Implement table + function (minimal: `*` shell entry only)**

```python
import json as _json

def _shell_to_cmd(payload: dict, required: list) -> str | None:
    if "command" not in payload:
        return None
    if "cmd" in set(required or []):
        return _json.dumps({"cmd": payload["command"]})
    return None

# (genuine, family["*" = any], client-lowered, arg-translator)
TO_CLIENT: tuple = (
    ("shell", "*", "exec_command", _shell_to_cmd),
)

def translate_to_client(genuine_name, arguments, family, owned, client_defs):
    from llms.proxy.ir import ToolDef
    lowered = genuine_name.lower() if isinstance(genuine_name, str) else ""
    if lowered in owned:  # same-name ownership wins
        return None
    try:
        payload = _json.loads(arguments) if isinstance(arguments, str) else None
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for genuine, fam, client_lower, convert in TO_CLIENT:
        if genuine != lowered or (fam != "*" and fam != family):
            continue
        declared = owned.get(client_lower)
        tool = client_defs.get(client_lower)
        if declared is None or not isinstance(tool, ToolDef):
            continue
        required = (tool.parameters or {}).get("required") or []
        converted = convert(payload, required)
        if converted is not None:
            return declared, converted
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python -m pytest tests/test_compat_table.py tests/test_client_tools.py -q`
Expected: PASS (existing `translate_genuine_call` untouched — still delegates in Task 4).

- [ ] **Step 5: Commit**

```bash
git add src/llms/llms/proxy/compat.py src/llms/tests/test_compat_table.py
git commit -m "feat: compat dispatch table with all-family shell entry

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### Task 3: Patch-grammar verification (blocks write/edit dispatch entries)

**Files:**
- Probe: `scripts/probes/compat_patch_grammar.py` (new)
- Docs: RCA draft inventory table (append grammar findings)

**Interfaces:**
- Consumes: nothing (standalone probe).
- Produces: verified Add-File / Update / Delete patch-text grammar; go/no-go for Task 5's `write`→`apply_patch` entry.

- [ ] **Step 1: Write the grammar probe**

Standalone script (memory: probes in `scripts/`): drive the luna leg with an explicit `custom_tool_call exec` carrying `await tools.apply_patch("<candidate patch>")` for three candidates — Add-File (`*** Begin Patch ***\n*** Add File: <ws>/grammar-add.txt\n<body>\n*** End Patch ***`), Update (same with `*** Update File:` + `@@` context hunk), Delete (`*** Delete File:`) — then read back / assert byte-exact content. One candidate per run; print PASS/FAIL per candidate with the harness's verbatim error on failure. (Exact script body follows the `scripts/codex_probe.py` `running_proxy` pattern; keep each candidate's patch text in a named constant so failures are reproducible.)

- [ ] **Step 2: Run Add-File candidate**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python scripts/probes/compat_patch_grammar.py add`
Expected: PASS with byte-exact file content (Add-File grammar already exercised live 2026-10-05/06 via `patch-content-*` probes).

- [ ] **Step 3: Run Update + Delete candidates**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python scripts/probes/compat_patch_grammar.py update` then `... delete`
Expected: PASS/FAIL recorded verbatim. If Update fails, Task 5 ships `write`→Add-File only and `edit` fails open with error logs until the grammar is known — record the verdict in the RCA draft.

- [ ] **Step 4: Record grammar in RCA draft**

Append the verified marker grammar (verbatim, with the passing patch texts) to the DRAFT RCA's inventory section. No commit (docs + probe ship with Task 7's verification commit).

### Task 4: Wire dispatch into the shared classifier + streaming fold

**Files:**
- Modify: `llms/proxy/client_tools.py` (`translate_genuine_call` delegates to compat table, keeps `view_image` guards)
- Modify: `llms/proxy/pipeline.py` (`_classify_calls` takes `family="unknown"`, passes to compat; `run()` detects family and threads it to `_steer_genuine_calls` → `_genuine_calls_in` → `_classify_calls`)
- Modify: `llms/proxy/forward.py` (`forward()`/`fold_and_steer_streaming()` take `family`, pass to `_classify`)
- Test: `tests/test_pipeline.py`, `tests/test_proxy.py`

**Interfaces:**
- Consumes: `translate_to_client` (Task 2), `detect_family` (Task 1).
- Produces: end-to-end `shell`→client rewrite on both steer paths with family threading; untranslatable outcomes logged via `log_untranslatable`.

- [ ] **Step 1: Write failing classifier test with family**

```python
def test_classify_shell_rewrites_with_family_threaded():
    # Same shape as test_shell_retransmit_same_call_id_steers_not_replays
    # but the rewrite path carries family="luna" and still lands the
    # exec-channel marker.
    ...
```

(Concrete body: build `_luna_namespace()` tools, one `shell {"command": "cat f"}` call dict, call `_classify_calls(..., family="luna")`, assert passthrough carries `__exec_rewrite__` with `await tools.exec_command({"cmd": "cat f"})`. Mirror for `family="unknown"` on plain `EXEC_COMMAND` tools asserting plain rename without the marker.)

- [ ] **Step 2: Run to verify fail**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python -m pytest tests/test_pipeline.py -q -k family`
Expected: FAIL (`family` kwarg missing).

- [ ] **Step 3: Implement delegation + threading (minimal)**

`client_tools.translate_genuine_call(..., family="unknown")`: after the same-name ownership check, try `compat.translate_to_client` first; fall back to the existing client-scan loop (keeps current behavior for pairs the table doesn't cover). `_classify_calls(..., family="unknown")`: pass `family` to `translate_genuine_call`. `run()`: `family = detect_family(request.headers.get("user-agent"), ingress, req.tools)`; thread through `_steer_genuine_calls` → `_genuine_calls_in` → `_classify_calls`, and into `forward(..., family=family)`. `forward()`/`fold_and_steer_streaming()`: accept `family="unknown"`, pass to the `_classify` call. Untranslatable genuine names on the fold path: `logger.error("[%s] no compat entry: %s (family=%s) — passing through", ...)` via `compat.log_untranslatable`.

- [ ] **Step 4: Run affected suites**

Run: `cd /home/uqmm/vc1/src/llms && .venv/bin/python -m pytest tests/test_pipeline.py tests/test_proxy.py tests/test_client_tools.py tests/test_compat_table.py -q`
Expected: PASS, including the pinned `view_image` trap tests (unchanged — no entry renames onto a viewer).

- [ ] **Step 5: Commit**

```bash
git add src/llms/llms/proxy/ src/llms/tests/test_pipeline.py src/llms/tests/test_proxy.py
git commit -m "feat: thread harness family through both steer paths

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### Task 5: Family-specific dispatch entries (read/write/edit) — only what Task 3 + captures prove 1:1

**Files:**
- Modify: `llms/proxy/compat.py` (append proven entries to `TO_CLIENT`)
- Test: `tests/test_compat_table.py`, `tests/test_proxy.py` (streaming replay for luna write→apply_patch)

**Interfaces:**
- Consumes: `translate_to_client` (Task 2), patch grammar verdict (Task 3), Claude/dsh tool schemas (Task 1 captures).
- Produces: new passing entries; unproven pairs stay absent (fail open + error log).

Shipped only with live or capture proof per entry:

| Entry | Proof required |
|---|---|
| `read{path}` → claude `Read` | Task 1 Claude schema capture shows path-compatible `Read` |
| `read{path}` → codex-plain `exec_command{cmd: cat …}` | existing `_read_via_shell_redirect` (promote into table, keep `shlex.quote`) |
| `read{path}` → luna `exec_command`-via-exec-channel | Task 3-style live probe OR hermetic replay on req-113 tools |
| `write{path,content}` → claude `Write` | Task 1 Claude schema capture |
| `write{path,content}` → luna `apply_patch`-via-exec-channel (Add-File patch synthesis) | Task 3 Add-File PASS |
| `edit{…}` → claude `Edit` / luna Update-hunk | Task 3 Update PASS, else ships absent (fail open) |
| codex-plain `write` (no native writer, no apply_patch declared) | ships absent: fail open + `logger.error` (printf fallback is NOT 1:1 — quoting/binary — unless a live probe proves byte-exact round-trip for the corpus in Task 7) |

- [ ] **Step 1: Write one failing test per entry above** (same style as Task 2 Step 1: positive translation + `None` on missing keys / wrong family / viewer trap).
- [ ] **Step 2: Run to verify fail** (`pytest tests/test_compat_table.py -q`, FAIL).
- [ ] **Step 3: Implement only the proven entries**, each as a `(genuine, family, client, converter)` row + pure converter (patch synthesis for Add-File: `*** Begin Patch ***\n*** Add File: {path}\n{content}\n*** End Patch ***` — exact grammar from Task 3's verdict, not this sketch).
- [ ] **Step 4: Run full hermetic suite** (`pytest tests/ -q`), expect PASS.
- [ ] **Step 5: Commit** (same trailer).

### Task 6: History-echo normalization (client→genuine: apply_patch→edit/write, Read→read)

**Files:**
- Modify: `llms/proxy/compat.py` (`TO_GENUINE` + `translate_to_genuine`)
- Modify: `llms/proxy/translate.py` (`to_zen_responses(..., family=...)` applies it at the `_translate` echo site; `run()` passes family)
- Test: `tests/test_translate.py`, `tests/test_compat_table.py`

**Interfaces:**
- Consumes: `detect_family` (Task 1); patch-grammar verdict (Task 3) for the apply_patch→write/edit parse direction.
- Produces: echoed history carries opencode-native names so the model never obeys dead client-native names from history.

- [ ] **Step 1: Write failing echo tests**

```python
def test_history_echo_normalizes_client_native_to_genuine():
    # Luna follow-up: ToolCallBlock apply_patch "*** Begin Patch *** Add File: f ..." echoes as
    # function_call write {"path": ..., "content": ...} (Add-File only; Task 3 grammar).
    # Claude follow-up: ToolCallBlock Read echoes as function_call read.
    # Unknown/parse-failed: echoes verbatim (fail open, no guess).
```

- [ ] **Step 2: Run to verify fail** (`pytest tests/test_translate.py tests/test_compat_table.py -q`, FAIL).
- [ ] **Step 3: Implement `translate_to_genuine(client_name, arguments, family)`** + call it at the echo site alongside the existing genuine→client `_translate` (existing direction keeps precedence: try genuine→client first for genuine names, client→genuine for owned client names; either `None` → verbatim echo).
- [ ] **Step 4: Run suites** (expect PASS, existing history-rewrite tests unaffected).
- [ ] **Step 5: Commit** (same trailer).

### Task 7: Live matrix probe + byte-exact verification per leg

**Files:**
- Create: `scripts/probes/compat_matrix_probe.py`
- Docs: RCA draft (record per-cell verdicts)

Matrix (skip cells the table doesn't cover — those assert fail-open passthrough + error-log presence instead of execution):

- codex-plain: `shell`→cat `/tmp/hi` (expect `Helllo world!!!`); `read`→cat; untranslatable `write` → passthrough + error log, no file.
- luna: `shell`→exec-channel cat (expect `luna-probe-content`); `write`→apply_patch Add-File (expect byte-exact file); `edit`→Update-hunk only if Task 3 passed.
- claude/dsh legs: only cells with captured schemas + table entries; everything else asserts fail-open.

- [ ] **Step 1: Write probe** (one `run_cell(leg, genuine_call)` returning `(model_emission, harness_outcome, proxy_log_markers)`; assert byte-exact file content / stdout per cell; collect `no compat entry` error-log lines per untranslatable cell).
- [ ] **Step 2: Run matrix, record verdicts** (each cell PASS/FAIL with verbatim evidence; FAIL = fix table, not the probe — one variable at a time).
- [ ] **Step 3: Record verdicts in RCA draft** (no commit yet — ships with Task 8).

### Task 8: Schema translation audit + finalize RCA + green suite + commit

**Files:**
- Docs: rename `docs/superpowers/plans/2026-10-05-codex-tool-compat-rca-DRAFT.md` → `docs/superpowers/plans/2026-10-06-opencode-harness-compat.md` (finalized: compat mechanisms, per-entry proof pointers, audit appendix)
- All modified code + tests

- [ ] **Step 1: Schema audit pass** — for every shipped table entry, diff the genuine-12 schema (`zen_tools.py`) against the client's declared schema (live captures req-93/95/113 + Task 1 Claude/dsh captures): list every dropped/renamed key per entry (e.g. `shell{workdir,timeout,…}` → `exec_command{cmd}` drops extras by contract) and confirm each drop is covered by a hermetic test asserting the dropped key never reaches the harness.
- [ ] **Step 2: Log review** — grep the probe-window proxy log for `no compat entry` lines; every distinct name gets either a new table entry (back to Task 5 with proof) or a recorded wont-translate reason in the audit appendix.
- [ ] **Step 3: Full suite + ruff** — `cd /home/uqmm/vc1/src/llms && .venv/bin/python -m pytest tests/ -q` (expect all green) and `ruff check` + `ruff format --check` on touched files (repo is check-clean not format-clean at HEAD — keep own lines formatted, don't reformat pre-existing lines).
- [ ] **Step 4: Commit the worktree** (only after Steps 1–3 + Task 7 matrix green), message ends with `Co-Authored-By: Claude Code <noreply@anthropic.com>`. Decide `docs/` disposition (untracked — commit with the work or leave out; record the choice in the commit message).

## Self-Review

- **Spec coverage:** user asked (a) full 1:1 compat with usual tools untouched ✓ (table only renames where client owns the target; `_with_genuine_tools` unchanged), (b) UA-based families with leg fallback ✓ (Task 1), (c) fail-open + error logs ✓ (Tasks 2/4/5 + `log_untranslatable`), (d) extensive live/probe tests ✓ (Tasks 3/7), (e) schema audit afterwards ✓ (Task 8), (f) apply_patch→edit/write + Read→read on follow-ups ✓ (Task 6). Probe-first + RCA-finalize-before-commit preserved in task gates.
- **Placeholder scan:** Task 5's patch-synthesis sketch is explicitly marked as superseded by Task 3's verdict (not a placeholder — a pointer). UA strings in Task 1 Step 3 are marked placeholders to replace with Step 2 observations. Task 4 Step 1 references an existing test by name for shape. No TBD/TODO.
- **Type consistency:** `family: str` threads as plain string constants (`CODEX_PLAIN`, …) across `detect_family` → `run()` → `forward()`/`to_zen_responses` → `_classify_calls`/`translate_to_*`; `translate_to_client` returns `(declared_casing_name, json_args)` matching `translate_genuine_call`'s existing tuple contract; `__exec_rewrite__` marker flow unchanged.
- **Review Focus:** each of the six lines maps to a task gate (capability-proof per entry Task 5; shell-quoting constraint Task 5 table note; grammar-governed synthesis Task 3→5; exec-channel marker Task 4 tests; echo normalization Task 6; unknown-family degradation Task 1 `unknown` + Task 8 log review).
