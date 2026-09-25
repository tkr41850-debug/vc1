# llm-server API contracts

Local LLM gateway. Accepts OpenAI Responses, OpenAI chat, and Anthropic Messages
dialects; routes each model to its preferred Zen endpoint, translating as needed.

## Auth vs affinity (never mixed)

Two different key namespaces, two different transports:

- `sk-...` — the API secret key. Lives **only** on the request header:
  `Authorization: Bearer sk-...` (OpenAI-style) or `x-api-key: sk-...`
  (Anthropic-style). Must exist and be enabled in `data/keys.yaml`;
  otherwise `401 {error.message: "missing or invalid secret key"}`.
  Authenticated here, attributed in usage, **never forwarded upstream**.
- `ak-...` — the affinity tag. Lives **only** in the request path as an
  optional first segment (`/{affinity}/...`, matching `ak-[A-Za-z0-9_-]+`).
  Unauthenticated: it is hashed with the model into a bucket, and the bucket
  picks the egress pool. Never auth, never billed, never forwarded upstream.

## Ingress paths

Every request except `/healthz` and the OAuth login flow must carry an
`sk-` secret key on the header. The `ak-` affinity prefix is optional
(bucket routing only):

```
GET  /healthz
GET  /api/admin/login | /api/admin/callback | POST /api/admin/logout

POST /v1/responses | /responses                    (Authorization: Bearer sk-...)
POST /v1/chat/completions | /chat/completions      (Authorization: Bearer sk-...)
POST /v1/messages | /messages                      (Authorization: Bearer sk-...)
GET  /v1/models | /models                          (Authorization: Bearer sk-...)

# same paths with optional unauthenticated affinity prefix:
POST /{affinity}/v1/responses | ...
```

The affinity prefix is stripped before routing; usage is attributed to the
`sk-` key, so different affinities sharing one secret aggregate together.

## Model catalog

`GET /v1/models | /models` (secret-key header required, affinity prefix
optional) returns the enabled models from `data/models.yaml`, OpenAI list
shape. Each entry carries `id`, `object`,
`created`, `owned_by: llms`, plus `zen_endpoint` (model's native Zen path),
`context_window` / `max_output_tokens` (`null` = unverified),
`reasoning_effort` tiers (or `null`) with `thinking_toggle` for on/off
reasoning models, `tools` / `streaming` support (`null` = unverified),
`pricing` (all catalog entries are free), and `contributor_terms` (prompts
may train future models). `data/models.yaml` — managed via the admin UI —
supersedes the seed set: the served list is exactly the enabled ids in the
file (missing file falls back to the seed). Unknown ids get a minimal entry.
`just llms catalog` diffs the seed against the live Zen free set. Served locally,
no upstream call.

## Admin API (GitHub OAuth session required)

```
GET    /api/admin/keys            keys with live usage aggregates
POST   /api/admin/keys            {key: sk-..., label?, enabled?} -> 201 (ak- rejected: 400)
PUT    /api/admin/keys/{key}      {label?, enabled?}
DELETE /api/admin/keys/{key}
GET    /api/admin/models          [{id, label, enabled}]
POST   /api/admin/models          {id, label?, enabled?} -> 201
PUT    /api/admin/models/{id}     {label?, enabled?}
DELETE /api/admin/models/{id}
GET    /api/admin/usage           {keys: {<key>: {requests, input_tokens, output_tokens, models}}}
GET    /api/admin/providers       [{id, label, kind, models, enabled, exits, retry_in, health{exits[]}}]
POST   /api/admin/providers       {id, label?, kind?, models?, enabled?, exits?} -> 201 (409 duplicate, 400 bad kind/exits)
PUT    /api/admin/providers/{id}  {label?, models?, enabled?, exits?}
DELETE /api/admin/providers/{id}  (noproxy: 403; datadirs kept for same-id re-create)
GET    /api/admin/providers/{id}/health   force-refresh + warp-cli debug
POST   /api/admin/providers/{id}/reconnect  bounce exits, clear backoff, re-poll
GET    /api/admin/providers/{id}/recent     last 10 requests
GET    /api/admin/providers/{id}/stream     SSE: recent snapshot + live requests

Warp providers own local exits: `exits` sizes the supervised exit count
(`data/warps/<id>/warp<N>/` datadirs, per-exit `warp-svc`, SOCKS on
`WARP_BASE_SOCKS_PORT`-up). Remote-pool `base_url` is rejected with a
migration error — llms manages warp-cli datadirs in-process, no sidecar.
```

Unauthenticated: `401 {error.message: "admin login required"}`. OAuth:
`GET /api/admin/login` redirects to GitHub; `GET /api/admin/callback?code=`
exchanges, allowlists `ADMIN_GITHUB_USERS`, sets a session, redirects to `/`.
The admin UI (`/`, same port) 302s to login when unauthenticated.

## Usage aggregation

In-memory per-`sk-` counters (`requests`, `input_tokens`, `output_tokens`,
`cached_tokens`, `reasoning_tokens`, per-model breakdown), recorded from
upstream `usage` payloads for all three dialects (streams count the request,
tokens `None`). Cache detail sources: responses/chat `*_tokens_details`
(`cached_tokens`/`reasoning_tokens`), messages `cache_read_input_tokens`.
Token usage carries `cached_tokens` / `reasoning_tokens` breakdowns across
dialects and streams. Flushed to `data/usage.json` every 60s and on shutdown;
reloaded as baseline on startup (old snapshots without the new fields migrate
to 0).

## Affinity buckets

`bucket = sha256("{affinity or ''}\x00{secret or ''}\x00{model.lower()}") mod NUM_BUCKETS`
(`NUM_BUCKETS`, default 1024). Affinity is the optional unauthenticated
`ak-` path prefix (empty when absent), secret is the authenticated `sk-`
header — separate namespaces, both hashed, neither forwarded upstream.
Traffic still spreads by model when both are absent.
Buckets map to egress slots
(`bucket % NUM_SLOTS` initially); on a rate-limit signal the bucket advances
to the next slot with cooldown `max(SLOT_COOLDOWN_S, Retry-After)`.
Warp pool providers created after startup (admin UI or
`/api/admin/providers`) resize the table to their live ready-exit count on
create/update/delete and on health refreshes; direct-only deploys stay at
one slot.

## Request contract (all dialects)

- `model` optional; defaults per dialect (`ZEN_DEFAULT_MODEL`,
  `ZEN_DEFAULT_CHAT_MODEL`, `ZEN_DEFAULT_MESSAGES_MODEL`).
- Unknown top-level fields are dropped; the proxy rebuilds a whitelisted
  request from its intermediate format. Unknown roles/content-block types
  yield `400 {error.message}` naming the offender.
- Client credentials (`sk-` secrets, harness dummy keys) are **never**
  forwarded upstream. The operator `ZEN_API_KEY` authenticates upstream when
  set; otherwise requests ride the anonymous free tier.
- Reasoning effort is translated, not dropped:
  `responses.reasoning.effort` ↔ `chat.reasoning_effort` (or
  `thinking: {type: enabled}` → `medium`) ↔ `messages.thinking`
  (budget ↔ tier via low 1024 / medium 4096 / high 16384).

## Upstream headers (rebuilt per request, never passthrough)

`User-Agent: opencode/<channel>/<ver>/<client>`, `x-opencode-client`,
`x-opencode-project`, per-key stable `x-opencode-session` (mirrored as
`x-session-affinity` / `x-session-id`), `Content-Type: application/json`,
plus `Authorization: Bearer <ZEN_API_KEY>` when the operator key is set
(`Bearer public` for the anonymous free tier — required, omitting it 403s).
The responses body carries `prompt_cache_key` equal to the session id.
Anonymous responses requests additionally shape `instructions`/tools to
whole genuine turns (`llms/proxy/zen_prompts.py`, tracked by the
fingerprint checker): bare single-shots ride the title prefix with no
tools key; anything richer rides the agent prompt with the genuine tool
set plus client extras. Non-client tool calls are steered back with a
redirect error and re-requested (bounded).
No `x-opencode-request`: genuine v2 omits it and Zen 403s when present.
`host`, `content-length`, `accept-*`, and harness identity headers are dropped;
httpx regenerates transport headers. Free-tier Zen access depends on the
opencode identity set; omitting it yields `MissingSessionID`.

## Response contract

- Upstream status is preserved verbatim. `429`/quota errors
  (`FreeUsageLimitError`, `rate limit`, `too many`, `quota`) rebalance the
  bucket and fail fast to the client; `Retry-After` is forwarded when present.
- Same-dialect responses pass through; cross-dialect responses are translated
  (`stop`↔`completed`, `length`↔`incomplete`, tool calls both ways,
  token usage remapped). Translation failure yields `502`.
- Token usage carries `cached_tokens` / `reasoning_tokens` breakdowns across
  dialects and streams. Truncation carries `incomplete_details.reason`
  (`length`/`max_tokens` normalize to `max_output_tokens`).
- Streams: `text/event-stream`. Chat streams terminate with `data: [DONE]`;
  Responses streams terminate with `response.completed` (no `[DONE]`); the
  translator synthesizes whichever terminator the ingress client expects.
  Zen `inference-cost` frames are stripped.

## Model routing (static)

- `/responses`: `muse-spark*`, `gpt-*`, `grok-*`
- `/chat/completions`: `deepseek*`, `kimi*`, `glm-*`, `minimax*`, `mimo*`,
  `big-pickle`, `ling-*`, `nemotron*`, `qwen3-coder`, `trinity*`, `ring-*`,
  `north-*`, `hy3*`
- `/messages`: `claude-*`, `qwen*` (except `qwen3-coder`)
- Unknown models stay on the ingress endpoint (passthrough).
- `MODEL_ALIASES` (`pattern=target,...`, `*` suffix = prefix match) rewrites
  the requested id before routing; both ids are logged. Bucketing and rate
  limits apply to the resolved model.

## Egress contract

`EgressProvider`: `num_slots()`, `client_for(bucket, slot) -> httpx client`,
`aclose()`. `DirectEgress` is the single-client fallback (one slot).
`WarpSocksEgress` dials a warp provider's ready SOCKS ports in-process
(`slot % ready exits`); zero ready exits raises, and the pipeline fails open
to direct. `ProviderEgress.resolve(model)` picks the warp provider serving
the model (warp takes precedence over the `noproxy` seed), skipping pools
with known-zero ready exits unless the pool may still recover (any slot not
last-seen disconnected — a mid-boot snapshot must not pin traffic to direct
forever); `sync_bucket_slots(table)` resizes the bucket table to the largest
live ready-exit count. Responses carry `x-egress-provider` (+
`x-pool-active-warp` with the request slot's position in the ready spread).

## Configuration

Environment reference lives in [README.md](README.md#environment).
This document specifies wire behavior only.
