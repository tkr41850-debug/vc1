from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from proxy.config import Settings, get_settings
from proxy.logging import setup_logging
from proxy.routes import router as health_router
from proxy.routes.responses import router as responses_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    async with httpx.AsyncClient(
        base_url=settings.zen_base_url, timeout=settings.request_timeout_s
    ) as client:
        app.state.upstream_client = client
        yield


def create_app(settings: Settings | None = None) -> FastAPI:
    setup_logging()
    app = FastAPI(title="zen-responses-proxy")
    app.state.settings = settings or get_settings()
    app.include_router(health_router)
    app.include_router(responses_router)
    return app


app = create_app()
