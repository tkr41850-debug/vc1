from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from llms.proxy.buckets import BucketTable
from llms.proxy.config import Settings, get_settings
from llms.proxy.egress import DirectEgress, ProviderEgress
from llms.proxy.logging import setup_logging
from llms.proxy.middleware import GateMiddleware
from llms.proxy.providers import ProviderRegistry
from llms.proxy.routes import router as health_router
from llms.proxy.routes.admin import router as admin_router
from llms.proxy.routes.auth import router as auth_router
from llms.proxy.routes.chat import router as chat_router
from llms.proxy.routes.messages import router as messages_router
from llms.proxy.routes.models import router as models_router
from llms.proxy.routes.providers import router as providers_router
from llms.proxy.routes.responses import router as responses_router
from llms.proxy.usage import UsageTracker
from llms.proxy.warp import WarpSupervisor

USAGE_FLUSH_INTERVAL_S = 60.0

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    if getattr(app.state, "usage", None) is None:
        app.state.usage = UsageTracker()
        app.state.usage.load_file(settings.data_dir)
    if getattr(app.state, "egress", None) is not None:
        yield
        return
    logger.info(
        "effective settings port=%s buckets=%s egress=%s aliases=%s defaults=%s/%s/%s auth=%s",
        settings.port,
        settings.num_buckets,
        settings.egress_mode,
        settings.model_aliases or "none",
        settings.default_model,
        settings.default_chat_model,
        settings.default_messages_model,
        "key" if settings.zen_api_key else "anonymous",
    )
    stop = asyncio.Event()

    async def _flush_loop():
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=USAGE_FLUSH_INTERVAL_S)
                break
            except TimeoutError:
                app.state.usage.save_file(settings.data_dir)

    task = asyncio.create_task(_flush_loop())
    if getattr(app.state, "warp", None) is None:
        app.state.warp = WarpSupervisor(settings.data_dir)
    if getattr(app.state, "providers", None) is None:
        app.state.providers = ProviderRegistry(
            data_dir=settings.data_dir,
            settings=settings,
            supervisor=app.state.warp,
        )
    else:
        app.state.providers._settings = settings
        if app.state.providers._supervisor is None:
            app.state.providers._supervisor = app.state.warp
    # Boot supervised pools for enabled warp providers so registrations
    # persist and heal across restarts (state dirs live under DATA_DIR).
    try:
        for provider in app.state.providers.load():
            if provider.kind == "warp" and provider.enabled:
                await app.state.providers.ensure_pool(provider)
    except Exception as exc:
        logger.warning("warp supervisor boot failed: %s", exc)
    async with httpx.AsyncClient(
        base_url=settings.zen_base_url, timeout=settings.request_timeout_s
    ) as client:
        app.state.egress = ProviderEgress(
            DirectEgress(client), registry=app.state.providers
        )
        app.state.bucket_table = BucketTable(
            num_buckets=settings.num_buckets,
            num_slots=app.state.egress.num_slots(),
            slot_cooldown_s=settings.slot_cooldown_s,
        )
        try:
            yield
        finally:
            stop.set()
            await task
            app.state.usage.save_file(settings.data_dir)
            await app.state.providers.aclose()
            supervisor = getattr(app.state, "warp", None)
            if supervisor is not None:
                await supervisor.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    setup_logging()
    app = FastAPI(title="llm-server", lifespan=lifespan)
    app.state.settings = settings or get_settings()
    app.state.usage = UsageTracker()
    app.state.usage.load_file(app.state.settings.data_dir)
    app.state.warp = WarpSupervisor(app.state.settings.data_dir)
    app.state.providers = ProviderRegistry(
        data_dir=app.state.settings.data_dir,
        settings=app.state.settings,
        supervisor=app.state.warp,
    )
    # Starlette executes middleware in reverse insertion order, so Gate runs
    # last (outermost) and sees the session populated by SessionMiddleware.
    # Without a session secret the admin UI cannot work: sign with a random
    # per-process key (admin sessions simply won't survive restarts) rather
    # than a publicly-known fallback. Set ADMIN_SESSION_SECRET in production
    # so logins persist and can't be forged after a restart.
    import secrets as _secrets

    session_secret = app.state.settings.admin_session_secret or _secrets.token_hex(32)
    app.add_middleware(GateMiddleware)
    app.add_middleware(SessionMiddleware, secret_key=session_secret)
    app.include_router(health_router)
    app.include_router(auth_router)
    app.include_router(admin_router)
    app.include_router(providers_router)
    app.include_router(models_router)
    app.include_router(responses_router)
    app.include_router(chat_router)
    app.include_router(messages_router)
    static_dir = Path(app.state.settings.static_dir)
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
    return app


app = create_app()
