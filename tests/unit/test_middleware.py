"""
Tests for src/api/middleware.py — APIKeyMiddleware and load_api_keys.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import PlainTextResponse

from src.api.middleware import APIKeyMiddleware, load_api_keys


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_app(keys: set[str] | None = None) -> TestClient:
    """Create a minimal FastAPI app with APIKeyMiddleware and a /ping route."""
    if keys is None:
        keys = {"secret-key"}

    app = FastAPI()
    app.add_middleware(APIKeyMiddleware, keys=keys)

    @app.get("/ping")
    async def ping():
        return PlainTextResponse("pong")

    @app.get("/health")
    async def health():
        return PlainTextResponse("ok")

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse("metrics")

    return TestClient(app, raise_server_exceptions=True)


# ── Auth tests ─────────────────────────────────────────────────────────────────

def test_valid_key_passes():
    """A request with a correct X-API-Key header must reach the endpoint."""
    client = _make_app(keys={"good-key"})
    resp = client.get("/ping", headers={"X-API-Key": "good-key"})
    assert resp.status_code == 200
    assert resp.text == "pong"


def test_missing_key_returns_401():
    """A request without X-API-Key must receive a 401."""
    client = _make_app()
    resp = client.get("/ping")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or missing API key"


def test_invalid_key_returns_401():
    """A request with a wrong X-API-Key must receive a 401."""
    client = _make_app(keys={"good-key"})
    resp = client.get("/ping", headers={"X-API-Key": "bad-key"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid or missing API key"


def test_health_exempt_from_auth():
    """/health must return 200 even without an API key."""
    client = _make_app()
    resp = client.get("/health")
    assert resp.status_code == 200


def test_metrics_exempt_from_auth():
    """/metrics must return 200 even without an API key."""
    client = _make_app()
    resp = client.get("/metrics")
    assert resp.status_code == 200


# ── load_api_keys ──────────────────────────────────────────────────────────────

def test_load_api_keys_from_env():
    """load_api_keys() parses a comma-separated API_KEYS env var."""
    with patch.dict(os.environ, {"API_KEYS": "key1,key2, key3 "}):
        keys = load_api_keys()
    assert keys == {"key1", "key2", "key3"}


def test_load_api_keys_fallback():
    """load_api_keys() falls back to {'dev-key'} when API_KEYS is not set."""
    env = {k: v for k, v in os.environ.items() if k != "API_KEYS"}
    with patch.dict(os.environ, env, clear=True):
        keys = load_api_keys()
    assert keys == {"dev-key"}
