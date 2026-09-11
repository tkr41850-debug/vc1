set working-directory := "./src/llms"

port := env_var_or_default("ZEN_GATEWAY_PORT", "8789")
pidfile := "/tmp/llms.pid"
logfile := "/tmp/llms.log"

default:
    @just --list

# All repo-root paths resolve via justfile_directory() inside recipes
# (just 1.50 rejects function calls in const context, so no := vars).
_root := justfile_directory()

web-install:
    npm ci --prefix {{ _root }}/web/llms

web-build: web-install
    npm run build --prefix {{ _root }}/web/llms
    ls {{ _root }}/src/llms/llms/proxy/static/index.html

up: web-build
    #!/usr/bin/env bash
    set -u
    root="{{ _root }}"
    if [ ! -f "$root/.env" ]; then echo "missing $root/.env (copy .env.example)"; exit 1; fi
    set -a; source "$root/.env"; set +a
    export DATA_DIR="$root/data"
    echo "aliases: ${MODEL_ALIASES:-none}"
    echo "data: $DATA_DIR"
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
    set -eu
    cd "{{ _root }}"
    if [ ! -f .env ]; then echo "missing .env (copy .env.example)"; exit 1; fi
    if ! docker compose version >/dev/null 2>&1; then echo "docker compose plugin required"; exit 1; fi
    docker compose --env-file .env up --build -d
    ok=0
    for i in $(seq 1 60); do curl -s --max-time 2 http://127.0.0.1:{{port}}/healthz | grep -q ok && ok=1 && break; sleep 2; done
    if [ "$ok" != "1" ]; then echo "gateway did not become healthy"; exit 1; fi
    curl -s --max-time 5 http://127.0.0.1:{{port}}/healthz; echo

docker-down:
    #!/usr/bin/env bash
    set -eu
    cd "{{ _root }}"
    if ! docker compose version >/dev/null 2>&1; then echo "docker compose plugin required"; exit 1; fi
    docker compose down
    echo stopped

docker-rebuild:
    #!/usr/bin/env bash
    # Bust poisoned layer cache (e.g. truncated npm packages from an
    # interrupted fetch surfacing as cryptic tsc crashes).
    set -eu
    cd "{{ _root }}"
    if ! docker compose version >/dev/null 2>&1; then echo "docker compose plugin required"; exit 1; fi
    docker compose --env-file .env build --no-cache

sync:
    uv sync --group dev

dev:
    #!/usr/bin/env bash
    export DATA_DIR="{{ _root }}/data"
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
    root="{{ _root }}"
    if [ ! -f "$root/.env" ]; then echo "missing $root/.env (copy .env.example)"; exit 1; fi
    set -a; source "$root/.env"; set +a
    curl -s --max-time 10 -b /tmp/llms-admin-cookie.txt -c /tmp/llms-admin-cookie.txt http://127.0.0.1:{{port}}/api/admin/providers | python3 -m json.tool

reconnect-pool id:
    #!/usr/bin/env bash
    # Manual pool reconnect: bounce the pool, drop cached egress, re-poll
    # health. Needs an admin session (log into the UI first).
    set -u
    root="{{ _root }}"
    if [ ! -f "$root/.env" ]; then echo "missing $root/.env (copy .env.example)"; exit 1; fi
    set -a; source "$root/.env"; set +a
    curl -s --max-time 60 -X POST -b /tmp/llms-admin-cookie.txt -c /tmp/llms-admin-cookie.txt http://127.0.0.1:{{port}}/api/admin/providers/{{id}}/reconnect | python3 -m json.tool

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

keygen label="local":
    #!/usr/bin/env bash
    # Bootstrap a gateway secret key without needing the admin UI/OAuth:
    # generates sk-<random>, appends it to data/keys.yaml, prints export lines.
    set -u
    root="{{ _root }}"
    secret="sk-$(openssl rand -hex 16)"
    if [ ! -f "$root/data/keys.yaml" ]; then printf '[]\n' > "$root/data/keys.yaml"; fi
    uv run --no-project --with pyyaml python - "$root/data/keys.yaml" "$secret" "{{label}}" <<'EOF'
    import sys, yaml
    path, secret, label = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        keys = yaml.safe_load(open(path).read()) or []
    except Exception:
        keys = []
    keys = [k for k in keys if isinstance(k, dict) and k.get("key") != secret]
    keys.append({"key": secret, "label": label, "enabled": True})
    open(path, "w").write(yaml.safe_dump(keys, sort_keys=False))
    EOF
    if [ ! -f "$root/.env" ]; then cp "$root/.env.example" "$root/.env"; echo "(created .env from example)"; fi
    echo "export LLMS_API_KEY=\"$secret\""
    echo "curl -H \"Authorization: Bearer $secret\" http://127.0.0.1:{{port}}/v1/models"

lint:
    uv run ruff check llms tests scripts
    uv run ruff format --check llms tests scripts
