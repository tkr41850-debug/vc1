from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from llms.proxy.affinity import parse_affinity_prefix


class AffinityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        affinity, stripped = parse_affinity_prefix(request.url.path)
        request.state.affinity = affinity
        if affinity is not None:
            request.scope["path"] = stripped
        return await call_next(request)
