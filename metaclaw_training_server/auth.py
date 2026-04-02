"""Bearer token authentication middleware."""

from __future__ import annotations

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Validate Bearer token on every request (skip /v1/health)."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Health check is always open
        if request.url.path == "/v1/health":
            return await call_next(request)

        if self.api_key:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != self.api_key:
                raise HTTPException(status_code=401, detail="Invalid API key")

        return await call_next(request)
