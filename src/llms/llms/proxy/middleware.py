from __future__ import annotations

from pathlib import Path

from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from llms.proxy.affinity import parse_affinity_prefix
from llms.proxy.auth import require_admin, session_login
from llms.proxy.keys import resolve_secret_key
from llms.proxy.store import StoreError

OPEN_PATHS = {"/healthz", "/api/hello"}
OPEN_ADMIN_PREFIXES = (
    "/api/admin/login",
    "/api/admin/callback",
    "/api/admin/logout",
)


def is_open_path(path: str) -> bool:
    return path in OPEN_PATHS


def is_oauth_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in OPEN_ADMIN_PREFIXES)


def is_admin_path(path: str) -> bool:
    return path == "/api/admin" or path.startswith("/api/admin/")


def is_ui_path(path: str) -> bool:
    if path == "/" or path == "/index.html":
        return True
    if path.startswith("/assets/"):
        return True
    suffix = Path(path).suffix.lower()
    return bool(suffix) and suffix not in {".json"} and not is_admin_path(path)


class GateMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # ak- affinity lives in the path: unauthenticated, hashed onto a bucket
        # so the bucket can pick an egress pool. It is never an API key.
        affinity, stripped = parse_affinity_prefix(request.url.path)
        request.state.affinity = affinity
        request.state.secret_key = None
        settings = request.app.state.settings
        path = request.url.path

        if is_open_path(path) or is_oauth_path(path):
            if affinity is not None:
                request.scope["path"] = stripped
            return await call_next(request)

        if is_admin_path(path):
            # dependency_overrides is the documented FastAPI seam for tests;
            # only honor it when explicitly enabled so prod never consults a
            # test hook in its hot path.
            import os as _os

            override = request.app.dependency_overrides.get(require_admin)
            if _os.getenv("ALLOW_ADMIN_OVERRIDE", "") != "1":
                override = None
            if (
                path
                in (
                    "/api/admin/login",
                    "/api/admin/callback",
                    "/api/admin/logout",
                )
                or override is not None
                or session_login(request, settings) is not None
            ):
                if affinity is not None:
                    request.scope["path"] = stripped
                return await call_next(request)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "admin login required"}},
            )

        if is_ui_path(path):
            login = session_login(request, settings)
            if login is None:
                return RedirectResponse(url="/api/admin/login", status_code=302)
            return await call_next(request)

        # sk- secret keys live on the Authorization header. The ak- affinity
        # prefix (when present) is unauthenticated bucket routing only.
        try:
            secret_key = resolve_secret_key(request, settings)
        except StoreError as exc:
            request.app.state.store_error = str(exc)
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "key store unavailable"}},
            )
        else:
            request.app.state.store_error = None
        if secret_key is None:
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "missing or invalid secret key"}},
            )
        request.state.secret_key = secret_key
        if affinity is not None:
            request.scope["path"] = stripped
        return await call_next(request)


AffinityMiddleware = GateMiddleware
