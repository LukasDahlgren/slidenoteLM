from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING

from fastapi import Request, Response
from pydantic import BaseModel

if TYPE_CHECKING:
    from .config import Settings

SESSION_COOKIE_NAME = "slidenotelm_session"
SESSION_MAX_AGE_SECONDS = 30 * 24 * 3600  # 30 days
_SESSION_VERSION = "v1"


class AuthStatusResponse(BaseModel):
    auth_required: bool
    authenticated: bool


class AuthLoginRequest(BaseModel):
    password: str


def derive_signing_key(app_password: str, app_session_secret: str = "") -> bytes:
    """Return the HMAC key for signing session cookies.

    Uses APP_SESSION_SECRET when configured, or deterministically derives a
    SHA-256 key from APP_PASSWORD so restarts keep existing cookies valid with
    a single environment variable.
    """
    if app_session_secret:
        return app_session_secret.encode("utf-8")
    return hashlib.sha256(f"slidenotelm:session-key:{app_password}".encode("utf-8")).digest()


def verify_password(candidate: str, expected: str) -> bool:
    """Constant-time comparison of a submitted password against APP_PASSWORD."""
    if not expected:
        return True
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def create_session_token(app_password: str, app_session_secret: str = "") -> str:
    """Create a deterministic HMAC-SHA256 signed session cookie value."""
    if not app_password:
        raise ValueError("Cannot create session token when app_password is empty.")
    key = derive_signing_key(app_password, app_session_secret)
    pw_digest = hashlib.sha256(app_password.encode("utf-8")).hexdigest()
    payload = f"{_SESSION_VERSION}:{pw_digest}".encode("utf-8")
    signature = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return f"{_SESSION_VERSION}.{signature}"


def verify_session_token(
    token: str | None,
    app_password: str,
    app_session_secret: str = "",
) -> bool:
    """Verify a signed session cookie value in constant time."""
    if not app_password or not token:
        return False
    expected = create_session_token(app_password, app_session_secret)
    return hmac.compare_digest(token.encode("utf-8"), expected.encode("utf-8"))


def is_https_request(request: Request) -> bool:
    """Detect whether the external client request arrived over HTTPS."""
    forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if forwarded_proto:
        first_proto = forwarded_proto.split(",")[0].strip().lower()
        if first_proto == "https":
            return True
    return request.url.scheme.lower() == "https"


def is_request_authenticated(request: Request, settings: Settings) -> bool:
    """Return True if auth is disabled or the request carries a valid session cookie."""
    if not settings.app_password:
        return True
    cookie_token = request.cookies.get(SESSION_COOKIE_NAME)
    return verify_session_token(
        cookie_token,
        settings.app_password,
        settings.app_session_secret,
    )


def set_session_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=SESSION_MAX_AGE_SECONDS,
        path="/",
        httponly=True,
        samesite="lax",
        secure=is_https_request(request),
    )


def clear_session_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=is_https_request(request),
    )
