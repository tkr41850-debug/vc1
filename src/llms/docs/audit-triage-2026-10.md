# Audit triage 2026-10 — dispositions for the 34 killed-audit rows

Source: `wf_99ec9554-3b9/journal.jsonl` (34 finder claims + 14 verify
verdicts, all `real: true`). Each row was re-verified against the current
tree before disposition — pre-labels from the plan were not taken on trust.
Two pre-labels were corrected (drain tooltip math is still wrong, now
floored at `<1m` instead of `~93m;` ring `pool[0]` still ignores bucket).

Labels: `fixed-already` (in tree, commit cited) / `task-covered` (Tasks 1-2
design) / `needs-fix → Task 5/6 bundle` (native execution; fanout
unavailable in this session) / `wont-fix-by-design` (operator sk-contract).

## fixed-already (10)

- publish_throttled quiet-window flag never reset (`admin_hub.py:72`,
  med) — `c9644e6` reworked to fire-and-forget: callers never block; the
  trailing publish delay is the throttle window working as designed.
- throttled push latches pending; cancel wedges hub (`admin_hub.py:72`,
  med) — `c9644e6`: CancelledError arms clear pending/scheduled and
  re-raise; verified lines 88-92, 106-110.
- Pool-dry shed leaks in-flight (`pipeline.py:710`, med) — release present
  at line 763; `a1a0bb4`/`bff913d`.
- Dedup hit/timeout paths leak in-flight (`pipeline.py:850`, med) —
  `a1a0bb4` + `test_inflight_leaks.py`.
- in-flight count leaks on shed/dedup early returns (`pipeline.py:710`,
  high; duplicate of the two above) — same cover.
- cancelled JSON request leaks in-flight (`pipeline.py:854`, med) —
  `a1a0bb4` BaseException handler +
  `test_cancelled_json_request_releases_inflight`.
- applyProviders merge silently drops changed fields (`App.tsx:41`,
  high) — `bff913d`: merge now compares label/kind/deletable/drain/
  retry_reason/health/exit fields; models use `join("\n")`.
- Optimistic toggle applyOne races SSE (`ProvidersTab.tsx:402`, med) —
  `bff913d`: functional updater, no stale closure.
- Toggle disabled during drain (`ProvidersTab.tsx:555`, med) — `bff913d`:
  `transitioning()` covers preparing/pending only; drain toggle stays
  enabled as the re-enable-to-cancel path.
- DebugModal freezes on first snapshot (`ProvidersTab.tsx:81`, med) —
  `bff913d`: `useEffect` resyncs live from the provider prop (lines 96-102).

## task-covered (2)

- boot_epoch never stamped (`main.py:88`, med) — Task 1 (`710eb30`):
  lifespan boot loop stamps `boot_epoch`; verified in tree.
- Ring tries one candidate then fails open (`egress.py:190`, med) —
  Task 2 design (single pick + queue-behind; plan forbids iterating
  candidates). Residual nit, not a fix: all-limited branch still picks
  `pool[0]` while the comment claims rotation.

## needs-fix → Task 5 stream-terminal bundle (5)

- Truncated upstream stream synthesizes completed terminal
  (`stream_translate.py:307`, high).
- Chat stream emitter masks failed terminal as stop (`:538`, high).
- Messages stream emitter masks failed terminal as end_turn (`:959`, high).
- Truncated SSE defaults to completed (`:218`, high; same `finish()`
  root cause as the first row — one fix covers both).
- Upstream failure reasons collapse to clean stop
  (`translate_response.py:175`, med; `da4cf88` preserved only the
  in-vocabulary tuple — unknown reasons e.g. `error` still collapse).

## needs-fix → Task 6 hardening bundle (15)

Translate leg (3): `tool_choice {"name"}` verbatim on chat egress
(`translate.py:542`, high; test at `test_translate.py:971` enshrines it);
`"none"`→`auto` inversion (`:176` + `to_zen_messages`, high); stop
sequences dropped on responses egress (`:730`, med).

Auth/edge (4): healthz leaks raw key-store parser error
(`routes/__init__.py:12`, med); redaction keeps bearer prefix, ignores
`x-api-key` (`logging.py:31`, med); provider-id path traversal
(`routes/providers.py:133`, med); unthrottled sk- guessing
(`middleware.py:105`, low).

Pool-state races (4): stale reconnect bounce skips trailing publish
(`routes/providers.py:356`, med — stale `registry.reconnect()` also
mutates health/clears backoff with no gen guard); provider delete leaks
gen/runtime/egress (`:296`, high — no gen bump, no `drop_egress`, stale
settle task can drop the recreated pool); stale auto-cycle bounce
resurrects dropped pools (`pipeline.py:1137`, high — `finally`
`refresh_health` → `ensure_pool` with no enabled/gen check); stale SOCKS
ports dial dead exits (`providers.py:460`, high — zero-ready clears
`num_slots` but not cached ports; `pick_port` ignores `num_slots`).

UI tail (3): drain tooltip clock-mix, backend row (`providers.py:529`)
+ UI row (`ProvidersTab.tsx:38`) — one fix (backend ships remaining-ms);
add/delete have no local apply (`ProvidersTab.tsx:449`, med); models
draft resync only guards focus (`:87`, low).

Order-identity guard applies to every tool-shape fix: the 12 genuine
tools go out byte-identical, in order, ahead of extras on anonymous legs.

## wont-fix-by-design (2)

- Operator reconnect reachable with any sk- key (`:374`, high) and
  Operator ips endpoint exposed to any sk- key (`:536`, med) — operator
  routes (`/api/providers/*`) take sk- secret key only by contract;
  gating them on GitHub OAuth would break the operator contract
  (plan Review Focus row 5). No code.

## Orphan check

10 + 2 + 5 + 15 + 2 = 34. Every finder title has exactly one
disposition; every needs-fix row names its Task 5/6 bundle.
