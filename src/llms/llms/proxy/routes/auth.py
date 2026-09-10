from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, RedirectResponse

from llms.proxy.auth import build_authorize_url, login_via_code
from llms.proxy.config import Settings, settings_from_app

router = APIRouter()


@router.get("/api/admin/login")
async def admin_login(
    request: Request, settings: Settings = Depends(settings_from_app)
):
    if not settings.github_client_id:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "github oauth not configured"}},
        )
    return RedirectResponse(url=build_authorize_url(settings), status_code=302)


@router.get("/api/admin/callback")
async def admin_callback(
    request: Request, settings: Settings = Depends(settings_from_app)
):
    code = request.query_params.get("code", "")
    if not code:
        return JSONResponse(
            status_code=400, content={"error": {"message": "missing code"}}
        )
    await login_via_code(request, settings, code)
    return RedirectResponse(url="/", status_code=302)


@router.post("/api/admin/logout")
async def admin_logout(request: Request):
    session = request.scope.get("session")
    if isinstance(session, dict):
        session.clear()
    return {"status": "ok"}
