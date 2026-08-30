from __future__ import annotations

import hashlib
import hmac
import time
from typing import Awaitable, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

from .config import get_settings

SIGNATURE_HEADER = "x-internal-signature"
TIMESTAMP_HEADER = "x-internal-timestamp"
USER_ROLE_HEADER = "x-user-role"
USER_SUB_HEADER = "x-user-sub"

EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}


def _expected_signature(secret: str, method: str, path: str, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret.encode("utf-8"), digestmod=hashlib.sha256)
    mac.update(method.upper().encode("utf-8"))
    mac.update(b"\n")
    mac.update(path.encode("utf-8"))
    mac.update(b"\n")
    mac.update(timestamp.encode("utf-8"))
    mac.update(b"\n")
    mac.update(body)
    return mac.hexdigest()


class HMACAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        settings = get_settings()
        secret = settings.internal_hmac_secret
        signature = request.headers.get(SIGNATURE_HEADER)
        timestamp = request.headers.get(TIMESTAMP_HEADER)

        if not signature or not timestamp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing internal signature headers",
            )

        try:
            ts_int = int(timestamp)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid timestamp",
            )

        skew = abs(int(time.time()) - ts_int)
        if skew > settings.hmac_max_skew_seconds:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Request timestamp out of allowed skew",
            )

        body = await request.body()
        expected = _expected_signature(
            secret=secret,
            method=request.method,
            path=request.url.path,
            timestamp=timestamp,
            body=body,
        )

        if not hmac.compare_digest(expected, signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature",
            )

        return await call_next(request)


def require_admin(request: Request) -> str:
    role = request.headers.get(USER_ROLE_HEADER, "").lower()
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return role


def install_hmac_middleware(app: ASGIApp) -> None:
    app.add_middleware(HMACAuthMiddleware)  # type: ignore[arg-type]
