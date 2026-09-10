from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from llms.proxy.buckets import BucketTable
from llms.proxy.config import Settings, get_settings
from llms.proxy.egress import DirectEgress
from llms.proxy.logging import setup_logging
from llms.proxy.middleware import AffinityMiddleware
from llms.proxy.routes import router as health_router
from llms.proxy.routes.chat import router as chat_router
from llms.proxy.routes.messages import router as messages_router
from llms.proxy.routes.responses import router as responses_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "egress", None) is not None:
        yield
        return
    settings: Settings = app.state.settings
    async with httpx.AsyncClient(
        base_url=settings.zen_base_url, timeout=settings.request_timeout_s
    ) as client:
        app.state.egress = DirectEgress(client)
        app.state.bucket_table = BucketTable(
            num_buckets=settings.num_buckets,
            num_slots=app.state.egress.num_slots(),
            slot_cooldown_s=settings.slot_cooldown_s,
        )
        yield


def create_app(settings: Settings | None = None) -> FastAPI:
    setup_logging()
    app = FastAPI(title="llm-server", lifespan=lifespan)
    app.state.settings = settings or get_settings()
    app.add_middleware(AffinityMiddleware)
    app.include_router(health_router)
    app.include_router(responses_router)
    app.include_router(chat_router)
    app.include_router(messages_router)
    return app


app = create_app()
