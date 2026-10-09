# Codex tool-compat RCA — FINAL (live-verified 2026-10-06)

> **Status: FINAL.** Stop condition met asymmetrically: live
> `exec_command` (cat) proven on BOTH legs, luna-leg `apply_patch`
> Add-File proven byte-exact (details below), hermetic suite 520
> passed, ruff clean on touched files, worktree committed. The
> spark-leg `apply_patch` file-write is NOT provable: the codex-plain
> leg offers no exec channel (mechanisms 10, 12) — a harness
> capability gap, not a proxy bug.

## Symptom

Codex harness turns died client-side (`tokens used 0`, `unsupported call:
shell`) with steer sequences like `['execute'] → ['read'] →
['exec_command'] → budget exhausted`. The model never produced a working
file read or patch on either leg.

## Confirmed mechanisms

1. **`{}` emission: the model fills `cmd` iff the schema marks it required.**
   Direct-probe A/B: `required: []` → model emits `{}`; `required:
   ["cmd"]` → model emits `{"cmd": "cat /tmp/hi"}`. An empty-args call
   against a `cmd`-required tool is therefore a *model* behavior to
   correct with a redirect, not a wire-shape bug.
2. **Namespaced emission both directions.** The harness applies its own
   defaulting (`with_default_namespace` fills `default` when absent), so
   the model emits `default.exec_command` while the client declares bare
   `exec_command` (and vice versa on other legs). Ownership, routing, and
   casing must key both dotted and bare forms (`owned_tool_names` /
   `dispatchable_names` bare-keying).
3. **Repeat-detector miss.** Redirect text stores the bare name
   (`exec_command`) in `seen_redirects`, but the check compared the full
   emitted name (`default.exec_command`) — namespaced repeats never
   matched, so the fail-fast never fired and the budget burned. Fixed
   with bare-vs-bare comparison via `split_call_name`.
4. **Verbose redirect never converted.** The ~1.7KB full-schema correction
   produced two verbatim `default.exec_command {}` repeats. Replaced with
   exact-emitted-name + missing keys + one-line retry sketch
   (`_minimal_arg_example`); full schema deliberately not repeated (the
   notice already carries it).
5. **History taught the dead name.** Echoed history replayed the dead
   genuine call (`shell` + `unsupported call: shell`) verbatim while the
   redirect said `exec_command` — the model obeyed history, not the
   redirect. Fixed: history blocks carry the rewritten name/args
   (`translate.py` applies `translate_genuine_call` to echoed calls).
6. **`read`/`write` never translate onto shared-`path`-key tools.**
   Renaming `read` onto `view_image` (spark + luna nested) replayed bytes
   the harness failed client-side. `read` steers with a directed
   `exec_command`/`cat` correction (`_read_via_shell_redirect`); only
   `shell {"command"}` → `{"cmd"}` translates.
7. **Nested tools need the exec channel.** Luna's deferred `functions`
   namespace only executes nested tools through the `exec` JS
   orchestrator as `custom_tool_call` named `exec` — a bare
   `function_call` named `exec_command` fails lookup. Classifier rewrites
   valid nested calls into that channel (`__exec_rewrite__` marker).
   Nested required keys are the INNER arg fields (`cmd`), never the TS
   `args:` wrapper (requiring the wrapper rejected every valid call).
8. **Session-per-turn minted fresh sessions.** Body-only relays strip
   thread headers, so every codex turn minted a new session and the
   upstream prompt cache never warmed. Fixed: `conversation_ref` falls
   back to body `client_metadata.thread_id/session_id`.
9. **Fold gaps on done-only / renamed turns.** Turns with no delta frames
   folded to empty args (steered as owned-but-invalid); translated calls
   replayed under the renamed target matched no frame. Fixed: added-frame
   argument seeding + `done_args` fallback (never merged with deltas),
   and replay matching by EMITTED name (`_emitted_name_for`).
10. **Notice placement + shape.** Full client-schema dump (~1.7KB)
    crowded out the signal and taught the namespaced form. Notice now
    renders one-line call sketches (`_call_sketch` / `_short_decl`),
    names the anti-prefix rule explicitly, and appends to instructions
    AND the first non-system input message.

## Fixes applied (committed)

`client_tools.py` (ownership/route bare-keying, redirect tightening,
`steer_to_equivalent`, `translate_genuine_call`, notice sketches),
`forward.py` (fold seeding, emitted-name replay, bare repeat detection,
steer log carries truncated args), `pipeline.py` (classifier rewrite
arms, history translation, exec-channel replay), `sessions.py`
(client_metadata thread ref), `stream_translate.py` (`done_args`),
`translate.py` (history rewrite, notice dual placement),
`compat.py` (family detector, TO_CLIENT dispatch incl. execute row,
fail-open breadcrumb). Tests: 520 passed, ruff clean on touched
files, including
`test_streaming_namespaced_emitter_gets_argument_correction` pinning the
live spark shape.

## Live verification 2026-10-06, round 2 (19:20–20:00 UTC, manual proxy :8793)

9. **The model emits genuine `execute {"code"}` carrying the nested JS
   verbatim — the SAME source the exec channel runs.** Manual probe 8
   log (with the new steer-argument logging): `execute
   {"code":"await tools.apply_patch('*** Begin Patch ***\\n*** Add
   File: grammar-add.txt\\nhello-grammar\\n*** End Patch ***')"}`.
   The pointed execute→exec rewrap redirect never fired (generic went
   out) because no classifier arm consumed `execute` — the
   `translate_genuine_call` gate admitted only read/shell/write/edit.
   Fix committed: `execute`→`exec` compat-table row (inner `code`
   replays verbatim as channel input) + gate admission +
   classifier short-circuit attaching `__exec_rewrite__` for
   declared==exec (feeding the JS back through
   `exec_channel_source` would ask for a nested tool named `exec`).
   Hermetic pins: `test_execute_rewrites_onto_nested_exec_channel`
   (luna rewrites, blank code steers, spark steers);
   `test_streaming_steer_log_records_call_arguments` (steer INFO
   carries truncated args/input).
10. **Wrong patch markers taught by the notice (proxy-side doc bug,
    now fixed).** The 19:31 probe's `\\n`-escaped retry failed patch
    verification (`The first line of the patch must be '*** Begin
    Patch'`) — but the failure was the TRAILING `***`, not the
    encoding: the notice HOW-TO and probe candidates taught `***
    Begin Patch ***` / `*** End Patch ***` with bare content, while
    the harness requires `*** Begin Patch` / `*** End Patch` (no
    trailing `***`) with `+`-prefixed added lines. Corrected form
    proven live on the luna leg (round 3): `await
    tools.apply_patch('*** Begin Patch\n*** Add File:
    grammar-add.txt\n+hello-grammar\n*** End Patch')` → harness
    `Script completed`, file byte-exact `hello-grammar\n` (14 bytes,
    od-verified). The earlier "harness-side newline double-encoding"
    framing was wrong — no double-encoding exists; real newlines in
    the JS string work. Fixed: notice HOW-TO + probe candidates show
    the proven form (commit 518e327); the write→apply_patch table
    entry still stays absent until Update/Delete grammars verify
    (Task 3 gate holds —
    `test_write_to_apply_patch_needs_live_grammar_proof` still pins
    None).
11. **Empty exec outputs are a harness display artifact, not empty
    execution.** `exec_command cat` over a real file returns `Output:`
    (blank) in `custom_tool_call_output` while the harness terminal
    shows the file content (`alpha`/`beta` rendered, `succeeded in
    327ms`). The output pipe reaches the harness UI but not the
    tool-result record. Multi-line reads therefore cannot be verified
    through tool results — only through terminal display or
    single-token echoes (earlier `Helllo world!!!` byte-exact cat on
    both legs remains the output-content proof).
12. **Spark-leg `apply_patch` is structurally impossible (harness
    capability gap, not routable).** The codex-plain leg declares only
    the `exec_command` function runner — no `exec` channel, so the
    classifier correctly has no `execute`→`exec` row there (`no
    compat entry: execute (family=codex-plain) — generic steer`, and
    the model confirms: "Tool `execute` is not available in this
    session"). The notice's "do NOT call `exec_command` directly; run
    via `shell`" line is ignored by the model (harmlessly): the
    "shell tool" rollout shows 11 genuine `exec_command {"cmd":
    "echo trying-shell"}` emissions, each executed fine by the
    harness. File-write on this leg would need a harness-declared
    channel that does not exist — both-legs `apply_patch` is
    unachievable without a harness change, which is out of scope.

## Live verification 2026-10-06, round 3 (20:10–20:45 UTC, manual proxy :8793)

- Luna `apply_patch` Add-File: PASS. First file-creation attempt
  with the corrected grammar returned harness `Script completed`
  and `/tmp/manual10-ws/grammar-add.txt` byte-exact
  `hello-grammar\n` (14 bytes, `od -c` verified). This is the
  `apply_patch` proof the stop condition required — on the leg
  that has the channel.
- Spark `exec_command` cat via exec channel: DONE, zero steers on
  the emission path (the 11-call repeat in the "shell tool" probe
  is model-side looping — history echo is healthy, 14/45 upstream
  POSTs carried `function_call_output`; the model re-ran a
  succeeded command instead of replying DONE).
- Spark `apply_patch` via shell-worded channel: DONE with no file
  (`/tmp/spark-ws` holds only `probe.txt`) — expected per
  mechanism 12: two `execute` steers with the CORRECT grammar, then
  stop. The grammar is right; the channel does not exist.
- Open alternative (not yet probed): the spark leg's `exec_command`
  is a working shell runner (11/11 emissions executed live,
  including `pwd; ls -la; cat` compound commands). A file-write on
  this leg may be achievable WITHOUT the exec channel — via a
  shell heredoc/redirect (`cat > file <<'EOF'`) through the
  already-proven `exec_command` path — rather than via
  `apply_patch` at all. That reframes the spark-leg gap from
  "impossible" to "needs a shell-redirection probe, not an
  apply_patch probe." No spark-leg workspace file has ever been
  created through any path (only `probe.txt` from setup), and the
  lone `patch-final.txt` reference in captures belongs to a luna
  session (`req-119.json`, model `gpt-5.6-luna`, exec channel) that
  reported `No such file`.
- Upstream instability 20:42 UTC: relayed `upstream status=429`
  plus 504/503 flaps and direct-probe 502s — quota/cooldown window,
  retried on the next tick.

## Live verification 2026-10-07, round 5 (02:30–02:40 UTC, quota clear)

End-to-end proof of the SHIPPED translation rows (b632ecc) under
natural prompts — no JS coaching, no channel naming:

- Spark natural file-write: PASS, first try, zero proxy steers.
  Model emitted genuine `function_call exec_command
  {"cmd": "printf 'hello-natural\\\\n' > spark-natural.txt"}`,
  harness `Process exited with code 0`, file byte-exact
  `hello-natural\n` (`od -c` verified). The `write`→`exec_command`
  row fires through the full classifier path on a natural prompt.
- Luna natural file-write, attempt 1: FAIL-nofile (model-side
  stall, not a proxy miss). Zero proxy steers — the model never
  emitted ANY tool call (single ingress, replied DONE with no
  action). Same shape as the round-3 attempt-1/2 stall: with no
  JS coaching the model sometimes declines the exec channel it
  cannot see as a wire tool.
- Luna natural file-write, attempt 2 (retry): PASS after
  model-side thrash. The model emitted 15× bare-patch `exec`
  inputs (`'*** Begin Patch\\n...'` WITHOUT the `await
  tools.apply_patch(...)` wrapper — `Script failed` + `SyntaxError`
  each time), one stray `exec_command cat` probe, then
  self-corrected to the shell path: `await
  tools.exec_command({"cmd": "pwd; ls -la; printf
  'hello-natural\\\\n' > luna-natural2.txt; ...})` → `Script
  completed`, file byte-exact `hello-natural\n` (`od -c`
  verified).
- Mechanism 13 (new): the luna notice teaches the apply_patch
  FORM (`await tools.apply_patch('*** ...')`) but the model
  drops the wrapper under a natural prompt and emits the patch
  text as the whole `exec` input. The `execute`→`exec` table row
  covers genuine `execute {"code"}` emissions, but a bare-patch
  `custom_tool_call exec` input has no classifier arm — nothing
  re-wraps it, so it fails in the harness JS parser 15 turns
  running. Fix direction: a classifier arm that detects a bare
  `*** Begin Patch` exec input and re-wraps it as `await
  tools.apply_patch(<input>)` (hermetic + live-proven before
  shipping — same Task-3-gate discipline). IMPLEMENTED 2026-10-07
  (58656ec): pure helper `rewrap_bare_patch_exec_input` +
  classifier arm in the Custom-route `exec` branch, riding the
  `__exec_rewrite__` marker both replay paths apply; hermetic pins
  at pure-helper, classifier (hand-built dict), and fold→classifier
  wire-path levels (ac7a1be — the wire-path pin exists because the
  dict-level test would pass even if the fold dropped the Custom
  `input`). LIVE PROOF STILL OPEN: the m13 natural-write probe ran
  against PRE-arm code (file created via model shell fallback,
  rewrap unfired), so the arm awaits its own live round (natural
  luna file-write via :8799, quota-gated) before it closes out.
- Practical upshot: luna file-write works end-to-end today via
  the model's own shell fallback (proven twice now), and via the
  exec channel when the model keeps the wrapper (rounds 3–4);
  the bare-patch stall has a shipped proxy-side arm with full
  hermetic cover, awaiting live proof — not a grammar or routing
  bug.

## Live verification 2026-10-07, round 4 (01:00–01:25 UTC, quota cleared)

Quota cleared ~01:01 UTC (HI round-trip 200,
`{"output":[{"content":[{"text":"HI"}]}]}` after ~4h of
`FreeUsageLimitError`/429). All probes ran through fresh probe
proxies on free ports (:8796–8798; :8793 is held by a stale proxy
from an earlier session and 401s fresh secrets — left untouched).
Coder note: `running_proxy`'s teardown races a still-running
`codex exec` (`TimeoutExpired` on terminate); the probe result
still lands — check the workspace file, not the exit path. Also,
`codex exec` without `stdin=DEVNULL` hangs on "Reading additional
input from stdin" with zero ingress.

- Luna `apply_patch` UPDATE grammar: PASS. First emission used
  `\\n`-escaped JS (`'*** Begin Patch\\n*** Update File: ...'`),
  harness `FileChange update` with unified diff
  `@@ -1 +1 @@\n-old-line\n+new-line\n`, `Script completed`,
  `grammar-update.txt` reads `new-line\n` (`od -c` verified).
  The `\\n`-escaped form works — the round-3 "real newlines only"
  framing was overstated: BOTH encodings execute (the harness runs
  the decoded JS either way). The model's own unescaped retry
  (`SyntaxError: Invalid or unexpected token`) failed, as expected
  for a bare newline inside a single-quoted JS string.
- Luna `apply_patch` DELETE grammar: PASS. Model first tried the
  unescaped form (`Script failed` + `SyntaxError`, same as above),
  then self-corrected to the `\\n`-escaped form on its own:
  `FileChange delete`, `Script completed`,
  `grammar-delete.txt` removed. Self-correction without any new
  steer is further proof the notice grammar is right. (The probe
  asserts removal, not bytes — the UPDATE probe above is the
  byte-exact proof; DELETE's assertion is file-absent.)
- Task 3 gate consequence: Add + Update + Delete grammars are ALL
  proven live. `write`→`apply_patch` (Add-File subset) and
  `edit`→`apply_patch` (Update subset) table entries are now
  unblocked pending implementation — the hermetic gate tests
  (`test_write_to_apply_patch_needs_live_grammar_proof`) may be
  retired when the rows land.
- Spark-leg shell-write via `exec_command` redirect: PASS.
  Natural file-creation prompt (no JS coaching), model emitted
  genuine `function_call exec_command
  {"cmd": "printf 'hello-shell\\\\n' > shell-write.txt"}` —
  zero steers in the proxy log — harness executed it
  (`Process exited with code 0`), `shell-write.txt` byte-exact
  `hello-shell\n` (`od -c` verified, 12 bytes). This is the FIRST
  spark-leg workspace file ever created through any path, and it
  reframes mechanism 12: spark-leg file-write is ACHIEVED via the
  already-proven `exec_command` shell path, not via `apply_patch`
  (which remains structurally impossible — no exec channel on the
  codex-plain leg). The gated `write`→`exec_command` synthesis row
  (`test_write_to_shell_redirect_needs_live_proof`) is now
  unblocked pending implementation.
- Spark cat re-probe: PASS. `exec_command {"cmd": "cat
  probe.txt"}`, `function_call_output` carried `Helllo world!!!`
  inline, model echoed `<content>Helllo world!!!</content>` —
  zero steers (`grep -c 'no compat entry'` = 0). First probe
  attempt hung on stdin (no ingress at all) — fixed with
  `stdin=DEVNULL`.
- Luna cat re-probe: PARTIAL (known artifact, not a regression).
  `read {"path": "...probe.txt"}` steered (correctly — `read` has
  no compat entry), model fell back to `custom_tool_call exec`
  `await tools.exec_command(...)` which the harness EXECUTED
  (`Script completed`, `function_call_output` shows
  `Output:\nHelllo world!!!` inline in the FIRST rollout's
  `function_call_output` record). But the model never relayed the
  content into its reply text, so the echo-assertion failed. This
  matches mechanism 11 (empty exec outputs are a harness display
  artifact — `custom_tool_call_output` shows blank `Output:` while
  the `function_call_output` record carries the bytes) plus a model
  narration choice: with a blank tool-result record it summarized
  instead of quoting. Byte transport is proven; the probe's echo
  assertion is too strict for the luna exec-channel path. Not a
  proxy bug — no code change.

## Live verification 2026-10-06 (quota cleared ~05:00 UTC)

- Strict A/B (`/tmp/direct_probe11.py`, direct to :8789): both
  `strict=false` and `strict=true` returned live
  `exec_command {"cmd": "cat /tmp/hi"}` turns — strict flag does not
  gate emission on this leg.
- Spark codex cat (manual proxy :8795, `muse-spark-1.3-contributor-free`):
  `Helllo world!!!` byte-exact, zero steers. Verbatim UA:
  `codex_exec/0.154.0 (Debian 13.0.0; x86_64) xterm-256color
  (codex_exec; 0.154.0)` — UA-prefix family matching holds.
- Luna codex cat (`gpt-5.6-luna` → spark upstream via MODEL_ALIASES,
  `-s workspace-write`): `Helllo world!!!` byte-exact, zero steers
  (note: bare `codex exec` without `-s workspace-write` hung with no
  ingress — harness approval prompt, not proxy behavior).
- Hermetic regression fixed same session: the empty-Custom steer guard
  (`pipeline.py` `_classify_calls`) initially broke
  `test_streaming_nested_exec_command_replays_as_exec_custom_call`
  e2e while passing in isolation — root cause was the fold dropping
  Custom `input` + wire type, so the tail check steered the proxy's
  own valid `exec` rewrite (indistinguishable from the live
  empty-exec shape). Fix: `fold_stream_calls` carries the announced
  wire `input` (even when empty) + `type: custom_tool_call` into the
  folded entry (`forward._remember_announced_arguments` extended);
  the classifier's blank-input steer then fires only on truly empty
  payloads. Pinned by `test_classifier_steers_empty_custom_exec_call`
  (folds live `/tmp/exec_stream.txt` bytes) and
  `test_fold_keeps_custom_input_for_valid_exec_rewrite`.
- Luna `apply_patch` Add-File grammar: FAIL (no file created) across
  three attempts (06:02, 06:38, 08:03 UTC). Two distinct failure
  modes observed:
  - Attempts 1–2: the model never emitted any tool call (single
    ingress, zero steers; reply: "there's no `exec` custom tool
    available in this session"). The notice's only `custom_tool_call`
    example used a shell command while the apply_patch entry showed a
    TS declaration — the model concluded the `exec` channel did not
    exist. Single-variable fix applied: the HOW-TO line now leads
    with a file-creation example
    (`await tools.apply_patch('*** Begin Patch ***...')`) ahead of
    the shell example (`client_tools.build_tool_notice`; pinned by
    `test_notice_nested_patch_example_names_custom_form`).
  - Attempt 5 (natural file-write, no JS coaching, 08:18 UTC):
    the model emitted genuine `write` (wire schema — proves it reads
    the wire tools and wants the file-write capability), got steered
    (correctly: `write` is untranslatable per the Task 3 gate), then
    emitted `execute`, and its visible text narrates the channel
    problem out loud: "Need to use exec wrapper — attempting file
    creation" → "NEED custom_tool_call exec — trying syntax" → gave
    up with DONE and no file. The model UNDERSTANDS the exec channel
    from the notice but cannot construct the emission: it knows WHAT
    (`custom_tool_call exec`) but not the exact frame syntax.
    Mechanism hypothesis, single variable: the HOW-TO example shows
    the item with `"type"` FIRST
    (`{"type": "custom_tool_call", "name": "exec", "input": ...}`),
    but the harness/router may key on a different field order or the
    model may need the `id`/`call_id` + `status` fields the added
    frame carries live. Needs frame-level comparison of a KNOWN-GOOD
    custom emission (none captured yet — the cat probes all ran over
    function_call) vs what the model attempts here. Capture first:
    log the steered-away custom attempts' frames before changing the
    example.
  Classifier-side the path is proven hermetically: a bare
  `custom_tool_call apply_patch` with marker text passes through
  with the `__exec_rewrite__` into the exec channel, and the fold
  carries Custom `input` so the tail check judges the real payload.

## Live verification 2026-10-07, round 6 (06:29 UTC, coached channel proof)

- Coached luna file-write via natural-write probe shape (PROBE_MODEL
  `gpt-5.6-luna`, proof proxy :8796 — NOT :8799): workspace file
  `m13-coached.txt` byte-exact `hello-natural\n` (`od -c` verified,
  14B). Rollout ordinal 38: `await
  tools.apply_patch('*** Begin Patch\n*** Add File:
  m13-coached.txt\nhello-natural\n*** End Patch')` (\n-escaped) →
  `FileChange add` + `Script completed` 5.9s → DONE. This proves the
  exec-channel DELIVERY mechanism end-to-end through the FIXED code
  (13a2f32: wire-Custom collector + replay by_id).
- Model fluency variance on the same run (same prompt family):
  ordinals 12/28 emitted the wrapper with REAL newlines inside the
  single-quoted JS string → `SyntaxError: Invalid or unexpected
  token`; ordinal 20 emitted a bare patch (no wrapper) → `SyntaxError:
  Unexpected token '**'`. Only the \n-escaped form executes — rounds
  3–4 framing holds. The bare-patch ordinal-20 turn is exactly the
  mechanism-13 shape the shipped arm targets, but it ran BEFORE the
  collector fix could see it (pre-fix `_genuine_calls_in` dropped
  wire Custom items → `([],[])` → arm never fired).
- Quota state: HI 200 at 06:14 UTC (free tier clear). Rate-limit
  tiers per operator (memory `zen-ratelimit-tiers`): 60s tier → wait
  60s; ~24h tier → `warp-cli disconnect` + `warp-cli connect`.
- PROXY HYGIENE (2026-10-07 ~07:00 UTC check): `:8799` (PID 1263097,
  booted Oct 6 21:07) runs PRE-arm code — it predates every compat
  commit after 58aa7cc (shipped rows b632ecc, arm 58656ec, collector
  13a2f32 all Oct 7). Do NOT run arm proofs against `:8799`; it
  would re-prove nothing (same trap as round 5's pre-arm probe).
  Live work must use a fresh `running_proxy` on a free port
  (8792/8794/8797/8798 verified free) with current HEAD code, as
  `scripts/probes/compat_matrix_probe.py` does. `:8796` (booted Oct
  7 06:28) does carry the full arm — left running, owned by the
  coached-proof run.
- Claude leg: FIRST live messages-leg proofs via proof proxy :8796
  (`sk-probe-...` bearer): `/v1/messages` → 200 `HI`, and
  `/ak-claude/v1/messages` → 200 `HI` (ak- path affinity stripped,
  same handler). Family detection confirmed in proxy log:
  `ingress UA='claude-cli/2.1.291 (external, sdk-cli)'
  ingress=messages`. Task 1 Claude schema capture still open
  (declared-tool input_schema + first file-write round-trip).

## Schema audit appendix (Task 8 Step 1 — hermetic, 2026-10-07)

Per-entry dropped/renamed keys vs `zen_tools.py` GENUINE_TOOLS, pinned
by `test_schema_audit_dropped_keys_pinned`:

| Row | Genuine keys in | Consumed | Dropped / refused |
|---|---|---|---|
| `shell`→`*`/`exec_command` | command, workdir, timeout, background | `command`→`cmd` (rename) | workdir/timeout/background dropped by contract (client cmd-runner takes `cmd` alone; runner cwd applies) |
| `execute`→`*`/`exec` | code | `code` verbatim as channel input | none (single-key schema) |
| `write`→luna/`exec` | path, content | both (Add-File patch synthesis) | unknown extras ignored |
| `edit`→luna/`exec` | path, oldString, newString, replaceAll | path/old/new (Update hunk) | `replaceAll: true` REFUSED (None — no Update-hunk form for global replace; NOT silently dropped) |
| `write`→codex-plain/`exec_command` | path, content | both (heredoc body verbatim) | none; delimiter collision (`EOF` line) or non-string fields → None (never synthesize a truncating command) |

Unshipped genuine names stay fail-open (`log_untranslatable`):
`read`, `glob`, `skill` (log review below). No claude/dsh-family
rows shipped — detection only; the `*` shell/execute rows serve
those legs unproven (Task 7 matrix).

## Log review (Task 8 Step 2 — all probe windows, 2026-10-07)

`no compat entry` lines across every proxy log in the job tmp dir:

- `read (family=luna)` ×11 — EXPECTED: Task 5 table ships no
  `read` row (mechanism 6: no 1:1 target; steers to `cat` via
  `_read_via_shell_redirect`). Wont-translate recorded.
- `skill (family=luna)` ×2, `glob (family=luna)` ×1 — EXPECTED:
  no client harness declares these; fail-open passthrough is
  correct. Wont-translate recorded.
- Zero distinct names without a verdict. No new table entry
  required from this review.

## Live verification 2026-10-08, round 7 (06:49–07:20 UTC, matrix probe on current code)

Claude-leg root cause + fix (stop-hook gap closed): 7x upstream-200
text-only turns ("I'll run the echo command via Bash") while spark
passed byte-exact. Double inversion of the spark-proven pattern —
outbound is genuine-12-only so client `Bash` has no wire schema, and
the `shell` directive never fired on the claude leg (`_ALIASES`
matched only `exec_command`, absent from the 23-tool capture; the
`cmd`-only gate rejected verbatim-`command` `Bash`). Fix (`e2114de`):
`(bash, shell, command)` alias row + `_is_shell_runner` admits the
verbatim-`command` shape; notice now directs shell commands at
upstream `shell` and demotes `Bash` to a pointer. Live: model
self-corrects ("Need to use bare `shell` tool name per instructions,
retrying") → `PASS claude: Bash tool-ok round-trip`, 2x consecutive
(`claude-fix` :8798 + `claude-rerun` :8791 probes). Cleanup `1edd455`
drops the orphaned `_is_cmd_shaped`.

- Mechanism-13 arm LIVE proof (stop-hook gap closed): luna
  file-write `PASS matrix-luna.txt: byte-exact` AND the arm fired
  24x across 24 traces (`mechanism-13 rewrap` log line, `bb6df62`
  observability; traces in `/tmp/m13-probe-proxy.log`). The model
  emits bare-patch exec inputs on the natural-write path; every one
  re-wraps into the apply_patch channel. Zero steer lines: clean
  passthrough each turn.
- Spark regression check: `PASS matrix-spark.txt: byte-exact` in 3
  upstream turns, zero steers (clean-workspace rerun; two earlier
  failures were stale-workspace + model-side variance — same code
  passes, `multi_agent_v1` tool set identical across pass/fail).
- Full hermetic suite green (535), ruff clean on touched files.
  Recorded in empty commit `206d63b` (verdicts only — logs stay in
  job tmp per probes-vs-tests split).

## Schema audit appendix, round 7 delta (Task 8 Step 1 — hermetic, 2026-10-08)

New rows since the 2026-10-07 audit (all pinned in
`test_schema_audit_dropped_keys_pinned` + `test_claude_rows_...`):

| Row | Genuine keys in | Consumed | Dropped / refused |
|---|---|---|---|
| `shell`→claude/`Bash` | command, workdir, timeout, background | `command` verbatim (1:1 — capture: Bash requires `["command"]`) | workdir/timeout/background dropped by contract (Bash carries its own timeout/description; client defaults apply) |
| `read`→claude/`Read` | path, offset, limit | `path`→`file_path` (rename; capture: Read requires `["file_path"]`) | offset/limit dropped (whole-file read is what the overlay asked for) |
| `write`→claude/`Write` | path, content | `path`→`file_path` rename, `content` verbatim (capture: Write requires both) | unknown extras ignored |
| `edit`→claude/`Edit` | path, oldString, newString, replaceAll | three renames (capture: Edit requires all three) | `replaceAll: true` REFUSED (None — client `replace_all` defaults False with different semantics; NOT silently dropped) |

Notice-side change (no wire effect — notice text only): the `shell`
directive + `Bash` pointer demotion on the claude leg (same
spark-proven shape; classifier keys on names, never on notice text).

## Log review, round 7 (Task 8 Step 2 — current-code probe windows, 2026-10-08)

`no compat entry` lines across `/tmp/*proxy*.log` + job tmp (all
windows, old and new):

- `read (family=luna)` ×11 — EXPECTED (unchanged): Task 5 ships no
  `read` row; steers to `cat` via `_read_via_shell_redirect`.
- `skill (family=luna)` ×3, `glob (family=luna)` ×1 — EXPECTED:
  no client harness declares these; fail-open passthrough correct.
- Zero lines from any current-code window (m13/spark-clean/
  luna-clean/claude-fix/claude-rerun/spark-retry logs): every
  genuine emission on every leg now hits a table row. No new table
  entry required from this review.

## Round 8 delta (2026-10-08 — streaming rename, buffering, casing norm)

Two commits since round 7: a2b540e (streaming `to_client` rename in
`translate_streaming` — genuine calls rename onto client declarations
mid-stream) and 119c8ca (buffer genuine chunks to the done frame +
normalize client-cased emissions to declared casing).

Why: live claude file-write probe FAILed — upstream `shell` replayed
verbatim downstream 15x, CLI dispatcher `No such tool available:
shell` every time (only dispatches `Bash`). My earlier synthesize-path
fix never runs live for this leg (all turns `stream=True`);
`translate_streaming` passed names verbatim. Then genuine `write`
(~105 chars, multi-delta) defeated the first-chunk-only rename the
same way (partial JSON -> None -> verbatim -> CLI rejection 3x).

### Live ledger, current code (119c8ca)

- Claude Bash: `FILE: 'hello-echo'` byte-exact, transcript
  `TOOL_USE: Bash`, `shell -> Bash` fired on the streaming turn.
- Claude Write: `FILE: 'hello-write'` byte-exact, transcript
  `TOOL_USE: Write`, `write -> Write` fired (buffered full payload),
  first try after the buffering fix.
- Spark: `PASS matrix-spark.txt: byte-exact 'hello-matrix\n'`
  (fresh run on 119c8ca).
- Luna: `PASS matrix-luna.txt: byte-exact 'hello-matrix\n'`
  (retry converged after 17 turns; m13 arm firing throughout).

BOTH-leg translation proven live on current code.

### Schema audit, round 8 (Task 8 Step 1 — no new rows)

No new table rows since round 7: the streaming rename reuses the
proven `translate_genuine_call` table (same drops/refusals, pinned by
`test_schema_audit_dropped_keys_pinned`). New behavior is transport
(buffering, casing norm), not schema:

- Casing norm (`bash` -> declared `Bash`): name-only, args untouched
  (payload already the client's own shape) — pinned by
  `test_streaming_normalizes_client_tool_casing`.
- Buffer-then-rename: identical translated output to the fold path,
  only later in the turn — pinned by
  `test_streaming_rename_buffers_multichunk_genuine_write`.

Hermetic: 539 passed, ruff check + format clean.

### Log review, round 8 (Task 8 Step 2 — new probe windows)

`no compat entry` across the six new windows (claude-write9/10/11,
matrix-spark3, matrix-luna3/4):

- `edit (family=luna)` ×1 (luna4, final turn): model emitted a
  degenerate `edit {"newString":"test",...}` probe-shaped call; no
  luna `edit` row exists by design (Task 5 ships write/edit via the
  exec channel only for usable payloads — a same-string no-op edit
  has no valid translation). Generic steer applied, turn completed,
  probe PASSED byte-exact on the same run. EXPECTED — no new entry.
- Zero lines from any claude/spark window: every genuine emission
  hits a table row. No new table entry required.

### Rounds 9–10: streaming fail-closed guards (repeat + stall) — 2026-10-08

Two live failure shapes from the `~/DONOTCOMMIT_logs.txt` window and
the stress runs, both fixed in `forward.py` (`fold_and_steer_streaming`)
with hermetic pins in `test_stream_translate.py`:

1. Same-name repeat (round 9, commits 3e816fc/05cd5ee): two defects —
   the repeat guard keyed redirects on the full emitted form
   (`default.view_image`) while comparing the split bare name
   (`view_image`), so it silently never fired (fix: bare-name keying);
   and the trip `break` only left the inner `for`, falling through to
   the re-request (fix: `_steer_repeat` flag + outer break). Pinned by
   `test_streaming_fold_repeat_prefixed_call_fails_closed_fast`
   (verified to fail on pre-fix code via path-scoped stash).
2. Cross-name stall (round 10, commit 43d4997): trace 883373baaf5f
   cycled `read` -> `default.view_image` -> `default.view_image` (4
   steers, budget death) — cycling names never trips the same-name
   guard. A `_steer_stall` counter now fails closed after 2
   consecutive fully-steered turns (`test_proxy.py` exhaustion test
   updated: `len(calls) == 2`, stall trip instead of budget burn).
   Pinned by `test_streaming_fold_cross_name_stall_fails_closed_fast`.

Live proof on the stall-guard commit: trace 069de27d81b6 (spark view
shape) steered `read` then `default.view_image` and failed closed with
the stall warning — 1 re-request, no budget burn, no dead turn leaked.

Stress verdict (shared proxy, fresh code): spark 4/5 + verify PASS,
claude 4/5 + verify PASS, luna 900s rerun PASS byte-exact (26/27
turns via mechanism-13 rewrap, zero guard trips; earlier 1/5 was the
300s harness cap vs slow upstream, not translation). Zero
`streamfailed`/`stream incomplete`/`unsupported call` signatures in
any stress output.

Hermetic: 541 passed, ruff clean.

### Round 11: write->shell-redirect emitted bare-string args (2026-10-09)

Residual ~1/4 heredoc FAIL across fix3/fix4/fix5 batches
(`tail ["exec_command printf %s > ... <<'EOF'..."]`, file missing,
RC=0): the tail detail shows RAW COMMAND TEXT as the arguments — a
shape no upstream JSON-args turn produces. The promoted loop-verdict
line (aba8cd1) settled it on the first instrumented death (trace
0a2ea3db627c): `loop calls=[{write {"content","path"}}]
passthrough=[exec_command] steer=[]` — the loop PASSED the turn via
the `write -> exec_command` compat-table translation, and the tail
steered the proxy's OWN replayed frames as `exec_command` missing
`cmd`. Root cause in `compat._write_to_shell_redirect`: it returned
the synthesized heredoc as a BARE STRING, the only translation in
the table not returning a JSON object. The replay wrote that string
into the added/delta/done frames; the tail refold judged
`exec_command` + raw text as owned-but-invalid (missing `cmd`) and
failed closed. Fix (8babec2): the converter returns
`{"cmd": "<heredoc>"}` — same contract as every other translation.
Hermetic: loop+tail agree on all three delta shapes (single, multi,
done-only) against the real spark fixture; the two
`test_compat_table` claude-fixture failures are pre-existing on HEAD
(missing `/home/uqmm/.claude/jobs/.../claude-schema.jsonl`).

Live on the fix (fix7 batch, :8790): 4/4 PASS, zero `budget
exhausted` / stall / repeat lines. Cell 4 exercised the fixed path
(trace 4426ece73ba8: `write {content,path}` -> passthrough
`exec_command`, no tail warning, file byte-exact `codex-ok`,
`od -c` verified `0000011 / 9 bytes`). Affected-area hermetic: 155
passed (2 pre-existing fixture failures excluded).

### Round 12: heredoc shape was a silent no-op; printf-argument fix (2026-10-09)

The `{"cmd": "<heredoc>"}` fix unblocked the tail (no more
fail-closed) but the six tmux turns that followed still left
`stress-b.txt` at 0 bytes with RC=0 — and the model looped
re-emitting `write`, rationally. Exact repro via `/bin/bash -lc`
showed why: `printf %s` with NO argument prints empty and NEVER
reads stdin, so `printf %s > path <<'EOF'...` exits 0 having
consumed nothing. The duplication loop on this leg was a
success-with-empty-output retry, not a proxy ack bug: every one of
the six "successful" turns did exactly nothing. Fix (045e5bc):
`_write_to_shell_redirect` now emits
`{"cmd": "printf '%s' '<content>' > <path>"}` — the content rides a
single-quoted printf ARGUMENT (embedded quotes escape as `'\''`,
byte-exact through bash; empty content writes an empty file, correct
semantics). Hermetic: converter + `/bin/bash -lc` execution proof
(plain/quotes/multiline/empty) + loop/tail agreement on
single/multi/delta/done-only shapes; full suite 536 passed, 8 failed
— all 8 `FileNotFoundError` on the absent
`/home/uqmm/.claude/jobs/.../claude-schema.jsonl` capture fixture,
pre-existing on HEAD (fails identically on the clean tree).

Live on the printf-argument fix (tmux `codex-stress` session, :8790
restarted on 045e5bc): 53 `steer fold iter` verdicts, ZERO
`budget exhausted` / `failing closed` / `stalled` lines; every
`write` turn `passthrough=[exec_command] steer=[]` and pane shows
`/bin/bash -lc "printf '%s' 'betan' > ..."` succeeding with
`stress-b.txt` non-empty (`od -c`: 6-byte `alphan`, 5-byte `betan`;
later turns `beta\n` with trailing newline as emitted). Two `no
compat entry: read (family=codex-plain)` lines — EXPECTED: no
codex-plain `read` row ships by design (same wont-translate class as
the round-7/8 luna log reviews); the lone steered `read` turn
completed via generic steer.

Session caveats (harness-side, not proxy defects): the stress
prompt's `stress-step-1` / `stress-done` validators never existed on
PATH (shell mangled the backticks/`\n` in the tmux prompt), so step 1
exited 127 until PATH shims were installed mid-session (`STEP1-OK /
EXIT:0` first observed after); the model drifted into off-task
source-tree greps late in the run; the session died on upstream
`429 Too Many Requests` (free-tier quota after ~339k tokens), never
emitting step-4 `STRESS-DONE` — convergence of the full 5-step
sequence was NOT observed, though every proxy-leg step (1–3) is
green in isolation.

Stability verdict: spark + claude verify cells PASS clean with
guards idle; luna 900s rerun PASS byte-exact (`matrix-luna.txt`,
CELL_RC=0; 26/27 turns via mechanism-13 rewrap, zero guard trips —
the earlier 1/5 was the 300s harness cap vs slow upstream, not
translation); 53 codex-leg turns with zero guard trips and no new
`no compat entry` name. No new proxy defect found — the stress loop
is closed on this leg.
