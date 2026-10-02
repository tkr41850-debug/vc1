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

# sk- guessing brake: per-(source, key-prefix) leaky bucket on auth
# failures only (successes and open paths never count). N failures
# inside WINDOW_S earn a COOLDOWN_S 429 instead of the 401, so online
# enumeration pays one cooldown per guess instead of guessing at request
# cost. Bucketed by source IP *and* the attempted key's short prefix
# (first 8 chars: "sk-xxxx…"), so one bad actor behind shared egress
# cannot 429 legitimate users guessing different keys — only the same
# (source, prefix) pair trips. Memory-only (single process, no
# persistence): worst case on restart is the brake starts empty, never
# a lockout that survives one. Idle buckets are pruned on write so
# quiescent sources never accumulate.
_AUTH_FAIL_WINDOW_S = 60.0
_AUTH_FAIL_THRESHOLD = 20
_AUTH_FAIL_COOLDOWN_S = 60.0
_AUTH_KEY_PREFIX_LEN = 8
_auth_failures: dict[str, list[float]] = {}
_auth_blocked_until: dict[str, float] = {}


def _auth_attempted_prefix(request: Request) -> str:
    """Short prefix of the presented (invalid) key for brake bucketing.

    Uses the presented value, not the resolved one (resolution failed —
    that is why we are here). Non-sk values collapse to one bucket; the
    full secret never enters the key (a dict key is not a log line, but
    minimal exposure is still the rule).
    """
    try:
        from llms.proxy.auth import presented_secret_key

        presented = presented_secret_key(request) or ""
    except Exception:
        presented = ""
    try:
        raw = request.headers.get("authorization", "") or request.headers.get(
            "x-api-key", ""
        )
        presented = presented or raw.strip().split()[-1]
    except Exception:
        pass
    return (presented.strip()[:_AUTH_KEY_PREFIX_LEN]) or "none"


def _auth_bucket(request: Request, source: str) -> str:
    return f"{source}\x00{_auth_attempted_prefix(request)}"


def _auth_source(request: Request) -> str:
    try:
        client = request.client
        if client is not None and getattr(client, "host", None):
            return str(client.host)
    except Exception:
        pass
    return "unknown"


def _auth_failed(bucket: str, now: float) -> bool:
    """Record an auth failure; True when the bucket is now rate-limited."""
    hits = _auth_failures.get(bucket)
    if hits is None:
        hits = _auth_failures[bucket] = []
    cutoff = now - _AUTH_FAIL_WINDOW_S
    while hits and hits[0] < cutoff:
        hits.pop(0)
    hits.append(now)
    # Prune idle buckets on write: a source that went quiet keeps no
    # residue (bounded memory under a distributed spray).
    if len(_auth_failures) > 4096:
        stale = [k for k, v in _auth_failures.items() if not v or v[-1] < cutoff][:1024]
        for k in stale:
            _auth_failures.pop(k, None)
            _auth_blocked_until.pop(k, None)
    if len(hits) >= _AUTH_FAIL_THRESHOLD:
        _auth_blocked_until[bucket] = now + _AUTH_FAIL_COOLDOWN_S
        return True
    return False


def _auth_limited(bucket: str, now: float) -> bool:
    until = _auth_blocked_until.get(bucket, 0.0)
    if until and now < until:
        return True
    if until and now >= until:
        _auth_blocked_until.pop(bucket, None)
        _auth_failures.pop(bucket, None)
    return False


def is_open_path(path: str) -> bool:
    return path in OPEN_PATHS


def is_oauth_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in OPEN_ADMIN_PREFIXES)


def is_admin_path(path: str) -> bool:
    return path == "/api/admin" or path.startswith("/api/admin/")


def is_ui_path(path: str) -> bool:
    if path == "/" or path == "/index.html":
        return True
    if path == "/ui" or path.startswith("/ui/"):
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
        import time as _time

        _now = _time.monotonic()
        _source = _auth_source(request)
        _bucket = _auth_bucket(request, _source)
        if _auth_limited(_bucket, _now):
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "too many auth failures"}},
                headers={"retry-after": str(int(_AUTH_FAIL_COOLDOWN_S))},
            )
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
            if _auth_failed(_bucket, _now):
                return JSONResponse(
                    status_code=429,
                    content={"error": {"message": "too many auth failures"}},
                    headers={"retry-after": str(int(_AUTH_FAIL_COOLDOWN_S))},
                )
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "missing or invalid secret key"}},
            )
        request.state.secret_key = secret_key
        if affinity is not None:
            request.scope["path"] = stripped
        return await call_next(request)


AffinityMiddleware = GateMiddleware
