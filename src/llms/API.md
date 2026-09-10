# llm-server API contracts

Local LLM gateway. Accepts OpenAI Responses, OpenAI chat, and Anthropic Messages
dialects; routes each model to its preferred Zen endpoint, translating as needed.

## Ingress paths

Plain and affinity-prefixed forms serve identically:

```
POST /v1/responses | /responses
POST /v1/chat/completions | /chat/completions
POST /v1/messages | /messages
GET  /healthz

POST /{affinity}/v1/responses | /{affinity}/responses
POST /{affinity}/v1/chat/completions | /{affinity}/chat/completions
POST /{affinity}/v1/messages | /{affinity}/messages
```

`{affinity}` matches `ak-[A-Za-z0-9_-]+`. Any other first segment falls through
to normal routing (usually 404). The prefix is stripped before routing; the key
is carried as request affinity, never forwarded upstream.

## Affinity buckets

`bucket = sha256("{affinity or ''}\x00{model.lower()}") mod NUM_BUCKETS`
(`NUM_BUCKETS`, default 1024). Plain-path traffic hashes with empty affinity,
so it still spreads by model. Buckets map to egress slots
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

## Egress contract (for vsp)

`EgressProvider`: `num_slots()`, `client_for(bucket, slot) -> httpx client`,
`aclose()`. Default `EGRESS_MODE=direct` (single client, one slot).
Warp-pool egress is `NotImplementedError` until vsp exposes per-bucket warp
selection; the required server side is: lease/select a warp exit for
`(bucket, slot)` and report health, honoring the same rate-limit signals.

## Configuration

Environment reference lives in [README.md](README.md#environment).
This document specifies wire behavior only.
