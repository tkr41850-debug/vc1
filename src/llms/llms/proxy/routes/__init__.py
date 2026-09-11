from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    body: dict[str, str] = {"status": "ok"}
    store_error = getattr(request.app.state, "store_error", None)
    if store_error:
        body = {"status": "degraded", "store_error": store_error}
    return body
