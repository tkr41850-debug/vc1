set working-directory := "/home/uqmm/vc1/src/llms"

default:
    @just --list

sync:
    uv sync --group dev

dev:
    uv run uvicorn proxy.main:app --host 127.0.0.1 --port 8789

test:
    uv run pytest -q

lint:
    uv run ruff check proxy tests
    uv run ruff format --check proxy tests
