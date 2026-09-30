# Provider lifecycle, push-only SSE, and ring routing — design spec

Date: 2026-09-30. Status: approved (sections §1–§7 signed off in chat, one by one).

## Intent

Make provider enable/disable feel instant and observable: the server acks
immediately, SSE pushes transitions (`preparing` / `draining`) then settled
states, and the UI never blocks on warp-cli work. Fix ring routing so traffic
spreads across warp providers by key/affinity/model/session. Harden SSE
heartbeats so no server-side failure can silence a stream past Cloudflare's
120s idle timeout. Move the server to push-only (no steady-state polling).

## Non-goals

Dark mode, restyle, keys/models tabs, pagination/sorting, pixel-rendering
tests. Ops-console restraint is the aesthetic. No teardown of warp datadirs
on disable, ever (Cloudflare re-registration is heavily rate-limited).

## §1 — Lifecycle: derived, not stored

Wire value `lifecycle` in `provider_snapshot()`, one of:

- `off` — disabled, quiesced: no pool, `in_flight == 0`
- `preparing` — enabled warp, not yet serving (turning on)
- `ready` — enabled and serving (noproxy, or ≥1 ready exit)
- `ratelimited` — enabled and `retry_in > 0` (overlay, except over draining)
- `draining` — disabled but not quiesced (turning off), carries drain detail
- `unhealthy` — enabled warp, 0 ready past the boot grace, with error detail

Derivation inputs (all in-memory on `ProviderRuntime` + `Provider.enabled`;
nothing new persisted except `enabled` itself): `p.enabled`, `rt.in_flight`
(new, §3), pool-alive read (`supervisor.get(id)` + daemon liveness, read-only,
no spawn), `rt.drain_until` (monotonic deadline, 0 = not draining),
`rt.boot_epoch` (monotonic, 0 = unknown), existing
`rt.health.{fetched_at,exits,error}` and `rt.retry_in()`.

Precedence (top wins, pure function, no I/O):

1. `draining` if `not enabled and (pool alive or in_flight > 0)`
2. `ratelimited` if `enabled and retry_in() > 0`
3. `preparing` if `enabled and kind == warp and (fetched_at == 0 or (0 ready
   and now - boot_epoch < boot_grace))`
4. `ready` if `enabled and (kind == noproxy or any ready)`
5. `unhealthy` if `enabled and kind == warp and 0 ready past grace`
6. `off` otherwise

Safety properties: the function never raises (unexpected shape falls back to
`preparing` when enabled / `off` when disabled); restart-safe (after restart
pools are empty: disabled → `off`, enabled → `preparing`; startup reaps orphan
`llms-warp-<id>-*` sockets/daemons for disabled providers, namespaced, never
touching the host warp-svc); SSE can never miss a transition (every frame
recomputes from truth; `drain_until`/`boot_epoch` are monotonic, no wall
clock). `boot_grace` defaults to the 45s bring-up window the pool already
uses; `drain_until = now + STREAM_TIMEOUT_S` (600s, single knob shared with
the upstream read budget).

## §2 — Ack + background tasks with generation-cancel

`PUT enabled` / `POST reconnect` validate, persist intent (YAML), stamp epoch
fields, publish one SSE frame, return in ms with the transition snapshot
(full snapshot shape, status 200). No `await` on warp-cli, pool boot, health
poll, or daemon kill in the request path.

`rt.gen: int` (in-memory) increments on every intent change
(disable/enable/reconnect/delete). Background tasks capture `gen` at spawn
and exit silently on mismatch after every `await`.

- Disable: request persists `enabled=false` (cordon immediate — `resolve()`
  skips disabled), stamps `drain_until`, publishes (Draining at once). Task
  waits for `in_flight == 0` or deadline (~2s checks), then
  `supervisor.drop(id)` + `drop_egress(id)` + `invalidate_ips(id)`, publishes
  (Offline). Deadline expiry publishes Offline with `drain.forced = true`.
  Datadirs never touched.
- Enable: persist `enabled=true`, stamp `boot_epoch`, publish (Preparing) →
  `ensure_pool` → forced `refresh_health` → publish (Ready/Unhealthy).
- Reconnect: same ack treatment; background does bounce → clear backoff →
  forced refresh → publish. Fixes today's tens-of-seconds HTTP block.

Background tasks never raise to HTTP: exceptions log + publish the best-known
snapshot. A crashed task self-heals (next intent bumps `gen`; every frame
recomputes from truth).

## §3 — In-flight counting + throttled push

`rt.in_flight: int` (event loop only, `max(0, …)` guard on decrement).
Increment in `pipeline.py` right after `resolve()` assigns a `provider_id`.
Decrement by wrapping `response.body_iterator` once at the pipeline layer
(covers all four stream generators in `forward.py`; `finally` covers abrupt
disconnect). JSON/synthesize legs decrement after `_record_usage`. Dedup-shed
429s and pre-resolve failures never incremented → never decrement.

Count-only changes publish through a per-topic ~2s throttle on `AdminHub`
(first call immediate, rest coalesce to one trailing publish). CRUD / health /
retry transitions bypass the throttle (immediate). The full snapshot carries
`in_flight`; the 60s flush loop stays as backstop (but stops publishing —
see §4).

## §4 — Push-only: what polling dies

Dies: (1) the 60s usage-flush republish (`main.py`) — flush keeps persisting,
stops publishing; (2) the providers-SSE `refresh()` pass (`admin_streams.py`)
as a timer — health refresh becomes event-driven (enable/boot task, reconnect
task, post-cycle refresh, 429-triggered refresh, explicit `GET …/health`).

Stays: heartbeat pings (keepalive, not polls); `WarpPool._status_refresher`
(supervisor liveness, out of scope); UI sustained-darkness re-probe
(failure-path only). Net: zero steady-state server work per idle admin
connection beyond the 15s ping yield.

## §5 — Heartbeat guarantees (Cloudflare 120s)

Uniform across all four SSE sites (`admin_streams._collection_stream`,
`providers.provider_stream`, `forward` streaming generators):

1. Ping emission unconditional and infallible (constant bytes, no formatting /
   I/O / locks). `try/except BaseException → yield ping → continue` around the
   wait; `is_disconnected()` errors mean clean exit, never stream death.
2. Snapshot/refresh work can never block a heartbeat: `snapshot()` +
   `json.dumps` inside try/except (failure degrades to ping + log);
   providers `refresh()` under a 10s hard timeout (under the 15s cadence).
3. Cadence budget: admin 15s, provider-recent 15s, forward 30s — worst case
   single miss still delivers at 2× cadence, well under 120s. Hermetic test
   asserts all heartbeat constants `< 60` (half the CF budget).
4. Client: `EventSource` auto-reconnect + sustained-darkness re-probe stay.

Verification: hermetic tests drive each generator with poisoned snapshot /
hanging refresh, assert pings arrive on cadence and the stream never ends.
Live CF-tunnel check is an optional manual probe (`scripts/`), not a gate.

## §6 — API cleanup + UI

Wire: `provider_snapshot()` gains `lifecycle`, `in_flight: int`, `drain:
{until_ms, forced} | null` (draining only). All other fields byte-identical.
`PUT`/`POST reconnect` return the full transition snapshot (replacing today's
`{id, enabled}` stub). No new endpoints, no new streams. `retry_in` /
`retry_reason` / `cycling` / `cycle_cooldown_remaining` stay (lifecycle
derives from them).

UI (`ProvidersTab.tsx`, `App.tsx`, `api.ts`; maybe a tiny `lifecycle.ts`):

- Dots: `ready` 🟢, `ratelimited` 🟡, `off` ⚪, `unhealthy` 🔴, `preparing` 🔵
  (new), `draining` 🟠 (new). Every dot, the Busy header + cells,
  transition-disabled buttons, and the debug-modal lifecycle/drain line carry
  `title=` hover tooltips (established file pattern) with live detail
  (`Draining — 3 in flight, quiescing in ~4m`, `Ratelimited — retry in 42s`).
- New Busy column: `in_flight` per provider, live over SSE. Makes failover
  visible (limited provider 🟡 + survivors' Busy ticking up, same stream).
- Instant actions: toggle/reconnect/delete apply the returned transition
  snapshot to local state immediately (no `reload()` round-trip; `reload` is
  already a no-op). Buttons disable while `preparing`/`draining` (affordance;
  `gen` counter makes doubles safe server-side regardless).
- Reconnect: ack spinner from returned snapshot, settles over SSE.
- Debug modal: lifecycle line + drain info when present.

## §7 — Ring routing (no-IP hash)

Root cause of warp-1 monopoly: buckets pick a *slot within the chosen
provider*; `resolve()` picks the *provider* by first-match-wins in file order,
never consulting the bucket.

Fix: `resolve(model, bucket)` hash-partitions: `candidates = [enabled warp
serving model, not draining/cycling]`; pick `candidates[bucket % len]` with
skip-at-pick-time (`draining` cordon, `cycling`, `retry_in > 0` unless all
limited → least-wait wins, fail-open preserved). noproxy stays the fallback.
Hash is `ak- + sk- + model + session`, no client IP (spoofable/unstable behind
proxies; the 4-tuple already gives full entropy). Same session → same bucket
→ same provider (prompt-cache warmth preserved). `create_provider` triggers a
background boot+poll (§2 task shape) so a new provider serves its ring share
within seconds.

## Review fold-in (open thread)

Relaunched agents (correctness, streams-SSE, pipeline) return structured JSON;
security-auth returned 8 findings (unverified). Confirmed findings get fixed
first, before lifecycle work. Any finding contradicting this design comes back
to the user before implementation.

## Test plan

Hermetic (`tests/`): lifecycle precedence matrix; ack-timing (toggle returns
< 1s with transition snapshot); generation-cancel (re-enable mid-drain kills
no daemons); drain force path (deadline → Offline forced); in-flight
increment/decrement incl. abrupt disconnect; throttle coalescing; ring spread
uniformity + skip-set + session pinning; heartbeat poison tests + constants
`< 60`; SSE snapshot-shape (transition frames carry new fields).
Probes (`scripts/`, externals-dependent): optional manual CF-tunnel idle
check; warp drain against real daemons. Probes verify, never gate — commits
need green tests + relevant probe eyeball per repo memory rules.
