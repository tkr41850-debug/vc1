set working-directory := "/home/uqmm/vc1/src/llms"

default:
    @just --list

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

lint:
    uv run ruff check llms tests scripts
    uv run ruff format --check llms tests scripts
