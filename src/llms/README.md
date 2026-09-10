# llm-server

Local LLM gateway: OpenAI Responses, OpenAI chat, and Anthropic Messages
dialects in; each model routed to its preferred Zen endpoint with translation.
See [API.md](API.md) for the wire contracts.

## Quickstart

```sh
cp ../../.env.example ../../.env  # once; fill in GitHub OAuth + admin users
just sync       # install deps with uv
just up         # build web UI + start gateway in background
just down       # stop it
just test       # mocked unit suite
just lint       # ruff check + format
just docker-up  # or: containerized (same port, ./data mounted)
```

Health: `curl localhost:8789/healthz` → `{"status":"ok"}`.
Port via `ZEN_GATEWAY_PORT` (default `8789`).

## Admin UI

`just up`, then open `http://localhost:8789/` and sign in with GitHub
(register an OAuth app; callback `http://localhost:8789/api/admin/callback`;
put your login in `ADMIN_GITHUB_USERS`). Manage keys (with aggregate usage)
and models; stored in `data/keys.yaml` / `data/models.yaml` at the repo root.

## Claude Code (free, via Muse Spark)

Client side (`~/.bashrc`, then `source ~/.bashrc`; create the `sk-` key in
the admin UI first — needs `just up` running):

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8789"
export ANTHROPIC_API_KEY="sk-your-key"
export ANTHROPIC_MODEL="muse-spark-1.3-contributor-free"
```

`ANTHROPIC_MODEL` matters: plain `claude --model <id>` gets overridden by
Claude's own default, the env var sticks. The `sk-` secret authenticates you
to the gateway (usage is attributed to it); the gateway itself authenticates
upstream with its own `ZEN_API_KEY` or the anonymous free tier. Prefixing the
path with `ak-<affinity>` is optional and only picks the egress pool.

Server side (same file is fine on the same machine — `just up` inherits it;
re-run `just up` afterwards so the server picks it up):

```bash
export MODEL_ALIASES="gpt-*=muse-spark-1.3-contributor-free,claude-*=muse-spark-1.3-contributor-free"
```

This is belt-and-braces behind `ANTHROPIC_MODEL`: even if Claude falls back
to its own default (e.g. `gpt-5.4-xhigh-fast`), llms still serves Muse Spark
instead of failing with a billing 401. One-shot alternative without touching
`.bashrc`: `MODEL_ALIASES="..." just up`.

## Other consumers

```sh
just probe        # OpenAI SDK → Responses → Muse Spark
just probe-codex  # Codex CLI  → Responses → Muse Spark
just probe-dsh    # DeepSeek harness (chat) → auto-translated to model endpoint
just probe-claude # Claude Code headless (needs a messages-capable model)
```

All requests (including `/v1/models`) must carry an `sk-` secret key from
`data/keys.yaml` on the header (`Authorization: Bearer sk-...` or
`x-api-key`), e.g. `curl -H "Authorization: Bearer sk-team1"
localhost:8789/v1/models`. Missing, unknown, or disabled secrets get `401`.
Only `/healthz` and the OAuth login flow stay open. An optional `ak-`
affinity path prefix (`POST /ak-team1/v1/responses`) picks the egress pool
but never authenticates.

`GET /v1/models` lists the free catalog with limits, effort tiers, routing,
and tool/streaming support; `just catalog` checks the seed against live Zen.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `ZEN_API_KEY` | — | Operator Zen key; empty = anonymous free tier |
| `ZEN_BASE_URL` | `https://opencode.ai/zen/v1` | Upstream gateway |
| `ZEN_DEFAULT_MODEL` | `muse-spark-1.3-contributor-free` | Responses fallback |
| `ZEN_DEFAULT_CHAT_MODEL` | `muse-spark-1.3-contributor-free` | Chat fallback |
| `ZEN_DEFAULT_MESSAGES_MODEL` | `claude-haiku-4-5` | Messages fallback |
| `ZEN_GATEWAY_OPENCODE_VERSION/CLIENT/PROJECT` | `1.18.4`/`cli`/`global` | Upstream identity headers |
| `NUM_BUCKETS` | `1024` | Affinity hash space |
| `NUM_SLOTS` | `8` | Egress slots (warp pool size) |
| `SLOT_COOLDOWN_S` | `60` | Minimum bucket cooldown after a rate limit |
| `EGRESS_MODE` | `direct` | `direct` today; warp pool once vsp is ready |
| `MODEL_ALIASES` | — | Opt-in remap, e.g. `gpt-*=muse-spark-1.3-contributor-free,claude-*=muse-spark-1.3-contributor-free`. Stabilizes clients pinned to billed models, but the client sees the requested id while another model answers |
| `VSP_BASE_URL`, `VSP_TOKEN` | — | Warp pool endpoint/credential (future) |
| `ZEN_GATEWAY_PORT` | `8789` | Listen port |
| `ZEN_TIMEOUT_S` | `120` | Upstream timeout |
| `DATA_DIR` | `./data` | Keys/models YAML + usage.json |
| `GITHUB_CLIENT_ID/SECRET` | — | OAuth app for the admin UI |
| `GITHUB_REDIRECT_URI` | `http://localhost:8789/api/admin/callback` | OAuth callback |
| `ADMIN_GITHUB_USERS` | — | Comma-separated admin logins |
| `ADMIN_SESSION_SECRET` | — | Session cookie signing secret |
