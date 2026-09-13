mod llms "src/llms/justfile"

port := env_var_or_default("ZEN_GATEWAY_PORT", "8789")

default:
    @just --list

docker-up:
    #!/usr/bin/env bash
    set -eu
    cd "{{ justfile_directory() }}"
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
    cd "{{ justfile_directory() }}"
    if ! docker compose version >/dev/null 2>&1; then echo "docker compose plugin required"; exit 1; fi
    docker compose down
    echo stopped

docker-rebuild:
    #!/usr/bin/env bash
    # Bust poisoned layer cache (e.g. truncated npm packages from an
    # interrupted fetch surfacing as cryptic tsc crashes).
    set -eu
    cd "{{ justfile_directory() }}"
    if ! docker compose version >/dev/null 2>&1; then echo "docker compose plugin required"; exit 1; fi
    docker compose --env-file .env build --no-cache
