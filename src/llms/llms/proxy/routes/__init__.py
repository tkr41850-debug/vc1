from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> dict[str, str]:
    body: dict[str, str] = {"status": "ok"}
    store_error = getattr(request.app.state, "store_error", None)
    if store_error:
        # Unauthenticated path: never reflect raw parser text (YAML
        # errors echo the offending line, which can carry key/config
        # material). Degraded status is the signal; detail stays in logs.
        body = {"status": "degraded", "store_error": "key store unavailable"}
    return body


@router.api_route("/api/hello", methods=["GET", "HEAD"])
async def api_hello() -> dict[str, str]:
    return {"status": "ok"}
