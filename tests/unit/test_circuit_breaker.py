"""Unit tests for the circuit breaker (closed/open/half_open state machine)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from src.retrieval.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    _registry,
    get_circuit_breaker,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _ok(value=42):
    return value


async def _fail():
    raise RuntimeError("boom")


def _make_breaker(threshold: int = 3, timeout: float = 30.0) -> CircuitBreaker:
    """Return a fresh, isolated CircuitBreaker (not in the global registry)."""
    return CircuitBreaker(name="_test_", failure_threshold=threshold, recovery_timeout=timeout)


# ── Closed-state tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_closed_state_allows_calls():
    breaker = _make_breaker()
    result = await breaker.call(_ok(99))
    assert result == 99
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_closed_state_passes_through_exception():
    breaker = _make_breaker(threshold=10)
    with pytest.raises(RuntimeError, match="boom"):
        await breaker.call(_fail())
    # Still closed — not yet at threshold
    assert breaker.state == "closed"


# ── Opening the circuit ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_opens_after_threshold_failures():
    breaker = _make_breaker(threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail())
    assert breaker.state == "open"


@pytest.mark.asyncio
async def test_open_state_raises_circuit_open_error():
    breaker = _make_breaker(threshold=1)
    with pytest.raises(RuntimeError):
        await breaker.call(_fail())
    assert breaker.state == "open"

    with pytest.raises(CircuitOpenError):
        await breaker.call(_ok())


@pytest.mark.asyncio
async def test_open_state_does_not_increment_total_calls_on_rejection():
    """Calls rejected in OPEN state still increment total_calls (they were attempted)."""
    breaker = _make_breaker(threshold=1)
    with pytest.raises(RuntimeError):
        await breaker.call(_fail())

    before = breaker.stats()["total_calls"]
    with pytest.raises(CircuitOpenError):
        await breaker.call(_ok())
    # The rejected call still increments total_calls
    assert breaker.stats()["total_calls"] == before + 1


# ── Recovery (HALF_OPEN) ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recovery_after_timeout():
    """After recovery_timeout, state becomes half_open; success → closed."""
    breaker = _make_breaker(threshold=1, timeout=10.0)

    # Open the circuit
    with pytest.raises(RuntimeError):
        await breaker.call(_fail())
    assert breaker.state == "open"

    # Advance time past recovery_timeout
    with patch("src.retrieval.circuit_breaker.time.monotonic", return_value=time.monotonic() + 11.0):
        assert breaker.state == "half_open"
        result = await breaker.call(_ok(7))

    assert result == 7
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_half_open_failure_reopens():
    """A failed probe in half_open transitions back to open."""
    breaker = _make_breaker(threshold=1, timeout=10.0)

    # Open the circuit
    with pytest.raises(RuntimeError):
        await breaker.call(_fail())
    assert breaker.state == "open"

    # Advance time to half_open
    with patch("src.retrieval.circuit_breaker.time.monotonic", return_value=time.monotonic() + 11.0):
        assert breaker.state == "half_open"
        with pytest.raises(CircuitOpenError):
            await breaker.call(_fail())

    assert breaker.state == "open"


# ── Stats ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_tracks_counts():
    breaker = _make_breaker(threshold=5)

    await breaker.call(_ok())
    await breaker.call(_ok())
    for _ in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(_fail())

    s = breaker.stats()
    assert s["name"] == "_test_"
    assert s["state"] == "closed"
    assert s["failures"] == 3
    assert s["total_calls"] == 5
    assert s["open_count"] == 0

    # Trigger open
    with pytest.raises(RuntimeError):
        await breaker.call(_fail())
    with pytest.raises(RuntimeError):
        await breaker.call(_fail())

    s2 = breaker.stats()
    assert s2["state"] == "open"
    assert s2["open_count"] == 1


# ── Registry ─────────────────────────────────────────────────────────────────


def test_registry_returns_same_instance():
    # Use a unique name to avoid interference with other tests
    name = "_registry_test_unique_42_"
    _registry.pop(name, None)  # clean up if a previous run left it

    b1 = get_circuit_breaker(name)
    b2 = get_circuit_breaker(name)
    assert b1 is b2

    # Cleanup
    _registry.pop(name, None)
