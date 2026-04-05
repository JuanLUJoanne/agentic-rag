"""
API Key authentication middleware.

APIKeyMiddleware — Starlette BaseHTTPMiddleware
  Reads the X-API-Key header.  Missing or unknown keys → 401.
  Exempt paths: /health, /metrics, /docs, /openapi.json

load_api_keys() — reads API_KEYS env var (comma-separated).
  Falls back to {"dev-key"} with a warning when unset.
"""
from __future__ import annotations

import json
import os

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = structlog.get_logger()

_EXEMPT_PATHS = {"/health", "/metrics", "/docs", "/openapi.json"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Authenticate every non-exempt request via the X-API-Key header."""

    def __init__(self, app: ASGIApp, keys: set[str]) -> None:
        super().__init__(app)
        self._keys = keys

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key", "")
        if not api_key or api_key not in self._keys:
            body = json.dumps({"detail": "Invalid or missing API key"}).encode()
            return Response(
                content=body,
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


def load_api_keys() -> set[str]:
    """
    Load API keys from the API_KEYS environment variable (comma-separated).

    Falls back to {"dev-key"} and logs a warning when the variable is absent.
    """
    raw = os.environ.get("API_KEYS", "")
    if not raw:
        logger.warning(
            "api_keys_not_configured",
            fallback="dev-key",
            hint="Set API_KEYS env var (comma-separated) before deploying",
        )
        return {"dev-key"}
    keys = {k.strip() for k in raw.split(",") if k.strip()}
    if not keys:
        logger.warning("api_keys_empty_after_parse", fallback="dev-key")
        return {"dev-key"}
    return keys
