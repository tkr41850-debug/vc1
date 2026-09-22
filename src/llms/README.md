# llm-server

Local LLM gateway: OpenAI Responses, OpenAI chat, and Anthropic Messages
dialects in; each model routed to its preferred Zen endpoint with translation.
See [API.md](API.md) for the wire contracts.

## Quickstart

```sh
cp ../../.env.example ../../.env  # once; fill in GitHub OAuth + admin users
just llms sync       # install deps with uv
just llms up         # build web UI + start gateway in background
just llms keygen     # bootstrap an sk- key without the admin UI (prints export lines)
just llms down       # stop it
just llms test       # mocked unit suite
just llms lint       # ruff check + format
just docker-up  # or: containerized (same port, ./data mounted)
```

First key without OAuth: `just llms keygen [label]` appends a random `sk-` secret
to `data/keys.yaml` and prints `export`/`curl` lines. (The admin UI can manage
keys afterwards, but needs a configured GitHub OAuth app — keygen breaks the
chicken-and-egg.)

Health: `curl localhost:8789/healthz` → `{"status":"ok"}`.
Port via `ZEN_GATEWAY_PORT` (default `8789`).

## Admin UI

`just llms up`, then open `http://localhost:8789/` and sign in with GitHub
(register an OAuth app; callback `http://localhost:8789/api/admin/callback`;
put your login in `ADMIN_GITHUB_USERS`). Manage keys (with aggregate usage),
models, and warp providers; stored in `data/keys.yaml` / `data/models.yaml` /
`data/providers.yaml` at the repo root.

Warp providers are supervised in-process: llms owns its warp-cli datadirs
under `data/warps/<provider_id>/warp<N>/`, spawns `warp-svc` per exit, and
drives registration/proxy-mode/connect itself (no sidecar). A provider's
`exits` count sizes its local exit pool; requests dial the ready exits'
SOCKS ports directly, and traffic fails open to direct when no exit is
ready. `just docker-up` installs `warp-cli` in the image — the container
needs `/dev/net/tun` + `NET_ADMIN` (wired in `docker-compose.yml`) for
warp-svc to bring the tunnel up; without them pools stay unhealthy and
everything still serves direct.

## Claude Code (free, via Muse Spark)

Client side (`~/.bashrc`, then `source ~/.bashrc`; create the `sk-` key in
the admin UI first — needs `just llms up` running):

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8789"
export ANTHROPIC_AUTH_TOKEN="sk-your-key"
export ANTHROPIC_MODEL="muse-spark-1.3-contributor-free"
```

Prefer `ANTHROPIC_AUTH_TOKEN` over `ANTHROPIC_API_KEY` here: it is sent as
`Authorization: Bearer` (Anthropic's documented gateway path) with no
approval prompt, while `ANTHROPIC_API_KEY` goes out as `x-api-key` and
triggers Claude Code's one-time key approval, which may display the key in
`sk-ant-` form. The gateway accepts both headers, and also accepts an
`sk-ant-<rest>` key when `sk-<rest>` is allowlisted.

`ANTHROPIC_MODEL` matters: plain `claude --model <id>` gets overridden by
Claude's own default, the env var sticks. The `sk-` secret authenticates you
to the gateway (usage is attributed to it); the gateway itself authenticates
upstream with its own `ZEN_API_KEY` or the anonymous free tier. Prefixing the
path with `ak-<affinity>` is optional and only picks the egress pool.

Server side (same file is fine on the same machine — `just llms up` inherits it;
re-run `just llms up` afterwards so the server picks it up):

```bash
export MODEL_ALIASES="gpt-*=muse-spark-1.3-contributor-free,claude-*=muse-spark-1.3-contributor-free"
```

This is belt-and-braces behind `ANTHROPIC_MODEL`: even if Claude falls back
to its own default (e.g. `gpt-5.4-xhigh-fast`), llms still serves Muse Spark
instead of failing with a billing 401. One-shot alternative without touching
`.bashrc`: `MODEL_ALIASES="..." just llms up`.

## Other consumers

```sh
just llms probe        # OpenAI SDK → Responses → Muse Spark
just llms probe-codex  # Codex CLI  → Responses → Muse Spark
just llms probe-dsh    # DeepSeek harness (chat) → auto-translated to model endpoint
just llms probe-claude # Claude Code headless (needs a messages-capable model)
```

All requests (including `/v1/models`) must carry an `sk-` secret key from
`data/keys.yaml` on the header (`Authorization: Bearer sk-...` or
`x-api-key`), e.g. `curl -H "Authorization: Bearer sk-team1"
localhost:8789/v1/models`. Missing, unknown, or disabled secrets get `401`.
Only `/healthz` and the OAuth login flow stay open. An optional `ak-`
affinity path prefix (`POST /ak-team1/v1/responses`) picks the egress pool
but never authenticates.

`GET /v1/models` lists the free catalog with limits, effort tiers, routing,
and tool/streaming support; `just llms catalog` checks the seed against live Zen.

## Local CLI (no login)

`just llms local ...` talks to the `just llms up` gateway directly: HTTP
calls reuse the first enabled key from `data/keys.yaml`, file reads skip
auth entirely. Same script runs standalone: `uv run python
scripts/local_call.py ...` from `src/llms`.

```sh
just llms local chat "hi"      # one chat turn
just llms local stream "hi"    # streamed chat turn
just llms local models         # list model ids
just llms local usage          # per-key token totals from usage.json
just llms local providers      # providers.yaml + ready warp exits
just llms local reconnect-pool warp-1  # bounce pool without admin login
just llms local logs [lines]           # tail -F /tmp/llms.log
just llms local add-provider warp-1 1   # append warp provider (exits=1),
                                        # then: just llms reconnect-pool warp-1
```

`exits` is the provider's pool size: how many local warp tunnels llms owns
for it (each with its own `data/warps/<id>/warp<N>/` dir). Traffic spreads
across ready exits; with none ready it fails open to direct.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `ZEN_API_KEY` | — | Operator Zen key; empty = anonymous free tier |
| `ZEN_BASE_URL` | `https://opencode.ai/zen/v1` | Upstream gateway |
| `ZEN_DEFAULT_MODEL` | `muse-spark-1.3-contributor-free` | Responses fallback |
| `ZEN_DEFAULT_CHAT_MODEL` | `muse-spark-1.3-contributor-free` | Chat fallback |
| `ZEN_DEFAULT_MESSAGES_MODEL` | `claude-haiku-4-5` | Messages fallback |
| `ZEN_GATEWAY_OPENCODE_VERSION/CLIENT/PROJECT` | `2.0.12`/`cli`/`global` | Upstream identity headers |
| `ZEN_GATEWAY_CHANNEL` | `latest` | `User-Agent: opencode/<channel>/<version>/<client>` |
| `NUM_BUCKETS` | `1024` | Affinity hash space |
| `NUM_SLOTS` | `8` | Bucket-table slots (grows to the live warp ready-exit count) |
| `SLOT_COOLDOWN_S` | `60` | Minimum bucket cooldown after a rate limit |
| `EGRESS_MODE` | `direct` | Reserved; warp providers route via SOCKS when healthy, else fail open to direct |
| `MODEL_ALIASES` | — | Opt-in remap, e.g. `gpt-*=muse-spark-1.3-contributor-free,claude-*=muse-spark-1.3-contributor-free`. Stabilizes clients pinned to billed models, but the client sees the requested id while another model answers |
| `WARP_EXITS` (`WARP_SLOTS` legacy) | `8` | Default warp exits per provider (per-provider `exits` overrides) |
| `WARP_HOLD_TIMEOUT` | `10` | Seconds to wait for a ready warp exit |
| `WARP_REG_INTERVAL_SEC` | `28800` | Stagger between slot registrations (Cloudflare rate limit) |
| `WARP_BOOT_RETRY_SEC` | `300` | Retry delay for unready exits |
| `WARP_BASE_SOCKS_PORT` | `40001` | First per-slot SOCKS port (unique ports allocated per slot) |
| `WARP_PROTOCOL` / `WARP_MASQUE` | `MASQUE` / — | `warp-cli mode` protocol; MASQUE endpoint override |
| `WARP_NET_MTU` | — | Optional MTU set on the egress iface at container boot |
| `ZEN_GATEWAY_PORT` | `8789` | Listen port |
| `ZEN_TIMEOUT_S` | `120` | Upstream timeout |
| `DATA_DIR` | repo-root `data/` | Keys/models YAML + usage.json |
| `GITHUB_CLIENT_ID/SECRET` | — | OAuth app for the admin UI |
| `GITHUB_REDIRECT_URI` | `http://localhost:8789/api/admin/callback` | OAuth callback |
| `ADMIN_GITHUB_USERS` | — | Comma-separated admin logins |
| `ADMIN_SESSION_SECRET` | — | Session cookie signing secret |
