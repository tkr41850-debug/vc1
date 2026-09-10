# llm-server API contracts

Local LLM gateway. Accepts OpenAI Responses, OpenAI chat, and Anthropic Messages
dialects; routes each model to its preferred Zen endpoint, translating as needed.

## Ingress paths

Every request except `/healthz` and the OAuth login flow must carry a key
prefix from `data/keys.yaml`:

```
GET  /healthz
GET  /api/admin/login | /api/admin/callback | POST /api/admin/logout

POST /{key}/v1/responses | /{key}/responses
POST /{key}/v1/chat/completions | /{key}/chat/completions
POST /{key}/v1/messages | /{key}/messages
GET  /{key}/v1/models | /{key}/models
```

`{key}` matches `ak-[A-Za-z0-9_-]+` and must exist and be enabled in
`data/keys.yaml`; otherwise `401 {error.message: "unknown or disabled API
key"}`. The prefix is stripped before routing; the key is carried as request
affinity, never forwarded upstream.

## Model catalog

`GET /{key}/v1/models | /{key}/models` returns the enabled models from
`data/models.yaml`, OpenAI list shape. Each entry carries `id`, `object`,
`created`, `owned_by: llms`, plus `zen_endpoint` (model's native Zen path),
`context_window` / `max_output_tokens` (`null` = unverified),
`reasoning_effort` tiers (or `null`) with `thinking_toggle` for on/off
reasoning models, `tools` / `streaming` support (`null` = unverified),
`pricing` (all catalog entries are free), and `contributor_terms` (prompts
may train future models). Override the set with `ZEN_FREE_MODELS`
(comma-separated); unknown ids get a minimal entry. `just catalog` diffs the
seed against the live Zen free set. Served locally, no upstream call.

## Admin API (GitHub OAuth session required)

```
GET    /api/admin/keys            keys with live usage aggregates
POST   /api/admin/keys            {key, label?, enabled?} -> 201
PUT    /api/admin/keys/{key}      {label?, enabled?}
DELETE /api/admin/keys/{key}
GET    /api/admin/models          [{id, label, enabled}]
POST   /api/admin/models          {id, label?, enabled?} -> 201
PUT    /api/admin/models/{id}     {label?, enabled?}
DELETE /api/admin/models/{id}
GET    /api/admin/usage           {keys: {<key>: {requests, input_tokens, output_tokens, models}}}
```

Unauthenticated: `401 {error.message: "admin login required"}`. OAuth:
`GET /api/admin/login` redirects to GitHub; `GET /api/admin/callback?code=`
exchanges, allowlists `ADMIN_GITHUB_USERS`, sets a session, redirects to `/`.
The admin UI (`/`, same port) 302s to login when unauthenticated.

## Usage aggregation

In-memory per-key counters (`requests`, `input_tokens`, `output_tokens`,
`cached_tokens`, `reasoning_tokens`, per-model breakdown), recorded from
upstream `usage` payloads for all three dialects (streams count the request,
tokens `None`). Cache detail sources: responses/chat `*_tokens_details`
(`cached_tokens`/`reasoning_tokens`), messages `cache_read_input_tokens`.
Token usage carries `cached_tokens` / `reasoning_tokens` breakdowns across
dialects and streams. Flushed to `data/usage.json` every 60s and on shutdown;
reloaded as baseline on startup (old snapshots without the new fields migrate
to 0).

## Affinity buckets

`bucket = sha256("{key}\x00{model.lower()}") mod NUM_BUCKETS`
(`NUM_BUCKETS`, default 1024). Every gated request carries its key as affinity.
Buckets map to egress slots
(`bucket % NUM_SLOTS` initially); on a rate-limit signal the bucket advances
to the next slot with cooldown `max(SLOT_COOLDOWN_S, Retry-After)`.

## Request contract (all dialects)

- `model` optional; defaults per dialect (`ZEN_DEFAULT_MODEL`,
  `ZEN_DEFAULT_CHAT_MODEL`, `ZEN_DEFAULT_MESSAGES_MODEL`).
- Unknown top-level fields are dropped; the proxy rebuilds a whitelisted
  request from its intermediate format. Unknown roles/content-block types
  yield `400 {error.message}` naming the offender.
- `Authorization: Bearer <key>` from the client is used **only** when no
  `ZEN_API_KEY` is configured **and** `ZEN_ALLOW_CLIENT_KEYS=1`. Otherwise
  the operator credential (or anonymous free tier) wins; harness dummy keys
  are never forwarded.
- Reasoning effort is translated, not dropped:
  `responses.reasoning.effort` ↔ `chat.reasoning_effort` (or
  `thinking: {type: enabled}` → `medium`) ↔ `messages.thinking`
  (budget ↔ tier via low 1024 / medium 4096 / high 16384).

## Upstream headers (rebuilt per request, never passthrough)

`User-Agent: opencode/<ver>`, `x-opencode-client`, `x-opencode-project`,
per-key stable `x-opencode-session`, per-request `x-opencode-request`,
`Content-Type: application/json`, plus `Authorization` per the rule above.
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

## Egress contract (for vsp)

`EgressProvider`: `num_slots()`, `client_for(bucket, slot) -> httpx client`,
`aclose()`. Default `EGRESS_MODE=direct` (single client, one slot).
Warp-pool egress is `NotImplementedError` until vsp exposes per-bucket warp
selection; the required server side is: lease/select a warp exit for
`(bucket, slot)` and report health, honoring the same rate-limit signals.

## Configuration

Environment reference lives in [README.md](README.md#environment).
This document specifies wire behavior only.
