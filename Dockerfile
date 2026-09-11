# Single image: node builds web/llms -> python serves gateway + SPA on :8789.
FROM node:24-slim AS web
WORKDIR /build/web/llms
COPY web/llms/package.json web/llms/package-lock.json ./
RUN npm ci && npx tsc --version
COPY web/llms/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app/src/llms
RUN pip install --no-cache-dir uv
COPY src/llms/pyproject.toml src/llms/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/llms/ ./
RUN uv sync --frozen --no-dev
COPY --from=web /build/src/llms/llms/proxy/static ./llms/proxy/static
ENV DATA_DIR=/data ZEN_GATEWAY_PORT=8789
EXPOSE 8789
VOLUME /data
CMD ["uv", "run", "uvicorn", "llms.proxy.main:app", "--host", "0.0.0.0", "--port", "8789"]
