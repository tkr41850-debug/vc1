# Single image: node builds web/llms -> python serves gateway + SPA on :8789,
# with warp-cli installed so llms can supervise its own warp exits in-process.
FROM node:24-slim AS web
WORKDIR /build/web/llms
COPY web/llms/package.json web/llms/package-lock.json ./
RUN npm ci && npx tsc --version
COPY web/llms/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app/src/llms
ENV DEBIAN_FRONTEND=noninteractive
# Cloudflare WARP repo (warp-cli + warp-svc). Suite is dynamic so this works
# on both Debian- and Ubuntu-based python images; verified 200s for
# trixie/bookworm/bullseye + jammy/noble/focal (same flow as vsp).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg lsb-release ca-certificates iproute2 iptables \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSl https://pkg.cloudflareclient.com/pubkey.gpg | gpg --yes --dearmor --output /usr/share/keyrings/cloudflare-warp-archive-keyring.gpg \
 && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/cloudflare-warp-archive-keyring.gpg] https://pkg.cloudflareclient.com/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/cloudflare-client.list \
 && apt-get update && apt-get install -y --no-install-recommends cloudflare-warp \
 && rm -rf /var/lib/apt/lists/* \
 && warp-cli --version
RUN pip install --no-cache-dir uv
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY src/llms/pyproject.toml src/llms/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/llms/ ./
RUN uv sync --frozen --no-dev
COPY --from=web /build/src/llms/llms/proxy/static ./llms/proxy/static
ENV DATA_DIR=/data ZEN_GATEWAY_PORT=8789
EXPOSE 8789
VOLUME /data
ENTRYPOINT ["/entrypoint.sh"]
