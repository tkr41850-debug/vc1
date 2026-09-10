from __future__ import annotations

import urllib.parse

import httpx
from fastapi import HTTPException, Request

from llms.proxy.config import Settings
from llms.proxy.logging import setup_logging

logger = setup_logging()

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"


def build_authorize_url(settings: Settings) -> str:
    params = {
        "client_id": settings.github_client_id,
        "redirect_uri": settings.github_redirect_uri,
        "scope": "read:user",
    }
    return f"{GITHUB_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


async def exchange_code(code: str, settings: Settings) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            GITHUB_TOKEN_URL,
            json={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret,
                "code": code,
                "redirect_uri": settings.github_redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
    resp.raise_for_status()
    return resp.json().get("access_token", "")


async def fetch_login(access_token: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            GITHUB_USER_URL,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
        )
    resp.raise_for_status()
    return (resp.json().get("login") or "").strip().lower()


def session_login(request: Request, settings: Settings) -> str | None:
    session = request.scope.get("session")
    if not isinstance(session, dict):
        return None
    login = (session.get("admin_user") or "").strip().lower()
    if not login:
        return None
    if login not in settings.admin_github_users:
        return None
    return login


def require_admin(request: Request) -> str:
    settings: Settings = request.app.state.settings
    login = session_login(request, settings)
    if login is None:
        raise HTTPException(status_code=401, detail="admin login required")
    return login


async def login_via_code(request: Request, settings: Settings, code: str) -> str:
    try:
        token = await exchange_code(code, settings)
        if not token:
            raise ValueError("empty access token")
        login = await fetch_login(token)
    except Exception as exc:
        logger.error("github oauth exchange failed: %s", exc)
        raise HTTPException(status_code=502, detail="github oauth failed")
    if not login or login not in settings.admin_github_users:
        raise HTTPException(status_code=403, detail="not an admin user")
    request.session["admin_user"] = login
    return login
