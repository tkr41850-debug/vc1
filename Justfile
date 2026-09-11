repo := "/home/uqmm/vc1/.claude/worktrees/web-ui"
set working-directory := repo + "/src/llms"

port := env_var_or_default("ZEN_GATEWAY_PORT", "8789")
pidfile := "/tmp/llms.pid"
logfile := "/tmp/llms.log"
webdir := repo + "/web/llms"
staticdir := repo + "/src/llms/llms/proxy/static"

default:
    @just --list

web-install:
    npm ci --prefix {{webdir}}

web-build: web-install
    npm run build --prefix {{webdir}}
    ls {{staticdir}}/index.html

up: web-build
    #!/usr/bin/env bash
    set -u
    if [ ! -f {{repo}}/.env ]; then echo "missing {{repo}}/.env (copy .env.example)"; exit 1; fi
    set -a; source {{repo}}/.env; set +a
    echo "aliases: ${MODEL_ALIASES:-none}"
    if [ -f {{pidfile}} ] && kill -0 "$(cat {{pidfile}})" 2>/dev/null; then kill "$(cat {{pidfile}})"; sleep 1; fi
    pkill -f "uvicorn llms.proxy.main" 2>/dev/null || true; sleep 1
    nohup uv run uvicorn llms.proxy.main:app --host 127.0.0.1 --port {{port}} > {{logfile}} 2>&1 & echo $! > {{pidfile}}
    for i in $(seq 1 30); do curl -s --max-time 2 http://127.0.0.1:{{port}}/healthz | grep -q ok && break; sleep 1; done
    curl -s --max-time 5 http://127.0.0.1:{{port}}/healthz; echo

down:
    #!/usr/bin/env bash
    if [ -f {{pidfile}} ]; then kill "$(cat {{pidfile}})" 2>/dev/null || true; rm -f {{pidfile}}; fi
    pkill -f "uvicorn llms.proxy.main" 2>/dev/null || true
    echo stopped

docker-up:
    #!/usr/bin/env bash
    set -u
    cd {{repo}}
    if [ ! -f .env ]; then echo "missing {{repo}}/.env (copy .env.example)"; exit 1; fi
    if command -v docker-compose >/dev/null 2>&1; then COMPOSE="docker-compose"; else COMPOSE="docker compose"; fi
    $COMPOSE --env-file .env up --build -d
    for i in $(seq 1 60); do curl -s --max-time 2 http://127.0.0.1:{{port}}/healthz | grep -q ok && break; sleep 2; done
    curl -s --max-time 5 http://127.0.0.1:{{port}}/healthz; echo

docker-down:
    #!/usr/bin/env bash
    cd {{repo}}
    if command -v docker-compose >/dev/null 2>&1; then COMPOSE="docker-compose"; else COMPOSE="docker compose"; fi
    $COMPOSE down
    echo stopped

sync:
    uv sync --group dev

dev:
    uv run uvicorn llms.proxy.main:app --host 127.0.0.1 --port 8789

test:
    uv run pytest -q

probe:
    uv run python scripts/deepseek_harness_probe.py

probe-admin:
    uv run python scripts/admin_usage_probe.py

probe-warp:
    uv run python scripts/warp_probe.py

providers:
    #!/usr/bin/env bash
    set -u
    if [ ! -f {{repo}}/.env ]; then echo "missing {{repo}}/.env (copy .env.example)"; exit 1; fi
    set -a; source {{repo}}/.env; set +a
    curl -s --max-time 10 -b /tmp/llms-admin-cookie.txt -c /tmp/llms-admin-cookie.txt http://127.0.0.1:{{port}}/api/admin/providers | python3 -m json.tool

probe-dsh:
    uv run --with deepseek-harness-sdk python scripts/dsh_headless_probe.py

probe-claude:
    uv run python scripts/claude_probe.py

probe-codex:
    uv run python scripts/codex_probe.py

probe-tools:
    uv run python scripts/tool_roundtrip_probe.py

probe-search:
    uv run python scripts/websearch_probe.py

catalog:
    uv run python scripts/refresh_catalog.py

lint:
    uv run ruff check llms tests scripts
    uv run ruff format --check llms tests scripts
