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
- Upstream instability 20:42 UTC: relayed `upstream status=429`
  plus 504/503 flaps and direct-probe 502s — quota/cooldown window,
  retried on the next tick.

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
