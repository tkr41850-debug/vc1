set working-directory := "/home/uqmm/vc1/src/llms"

port := env_var_or_default("ZEN_GATEWAY_PORT", "8789")
pidfile := "/tmp/llms.pid"
logfile := "/tmp/llms.log"

default:
    @just --list

up:
    #!/usr/bin/env bash
    set -u
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

sync:
    uv sync --group dev

dev:
    uv run uvicorn llms.proxy.main:app --host 127.0.0.1 --port 8789

test:
    uv run pytest -q

probe:
    uv run python scripts/deepseek_harness_probe.py

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
