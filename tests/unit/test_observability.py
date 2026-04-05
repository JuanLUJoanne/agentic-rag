"""
Tests for src/observability/metrics.py and the /metrics endpoint.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry


# ── Singleton ──────────────────────────────────────────────────────────────────

def test_metrics_singleton():
    """get_metrics() must return the same instance on every call."""
    from src.observability.metrics import get_metrics

    # Do not reset the global registry singleton — just verify idempotency
    m1 = get_metrics()
    m2 = get_metrics()
    assert m1 is m2


# ── Active requests gauge ──────────────────────────────────────────────────────

def test_active_requests_gauge_increments():
    """active_requests gauge should increment by 1 when inc() is called."""
    from src.observability.metrics import RequestMetrics

    # Use a private registry so we don't clash with the process-wide singleton
    registry = CollectorRegistry()
    metrics = RequestMetrics(registry=registry)

    before = metrics.active_requests._value.get()
    metrics.active_requests.inc()
    after = metrics.active_requests._value.get()
    assert after == before + 1
    # Clean up
    metrics.active_requests.dec()


# ── /metrics endpoint ──────────────────────────────────────────────────────────

def test_metrics_endpoint_returns_200():
    """GET /metrics must return 200 with text/plain content type."""
    # Patch generate_latest to avoid collecting real metrics state
    fake_body = b"# HELP active_requests\nactive_requests 0.0\n"

    with patch("src.api.main.generate_latest", return_value=fake_body):
        from src.api.main import app
        from fastapi.testclient import TestClient

        # /metrics is exempt from auth
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
