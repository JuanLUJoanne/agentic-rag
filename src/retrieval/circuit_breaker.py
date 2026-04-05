"""
Per-retriever circuit breaker — fault-isolation pattern.

Implements the classic three-state circuit breaker:
  CLOSED    — normal operation; failures are counted toward the threshold.
  OPEN      — all calls rejected immediately with CircuitOpenError;
              after recovery_timeout the breaker moves to HALF_OPEN.
  HALF_OPEN — one probe call is allowed; success → CLOSED, failure → OPEN.

Usage::

    breaker = get_circuit_breaker("bm25")
    result = await breaker.call(some_coroutine)
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable

import structlog

logger = structlog.get_logger()

_REGISTRY: dict[str, "CircuitBreaker"] = {}
_REGISTRY_LOCK = asyncio.Lock() if False else None  # populated lazily


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is in the OPEN state."""


class CircuitBreaker:
    """
    Async circuit breaker for a single named service.

    Parameters
    ----------
    name:
        Human-readable identifier (e.g. ``"bm25"``).
    failure_threshold:
        Number of consecutive failures before the circuit opens.
    recovery_timeout:
        Seconds to wait in OPEN state before transitioning to HALF_OPEN.
    """

    _STATE_CLOSED = "closed"
    _STATE_OPEN = "open"
    _STATE_HALF_OPEN = "half_open"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        self._state = self._STATE_CLOSED
        self._failures = 0
        self._total_calls = 0
        self._open_count = 0          # how many times the circuit has opened
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current state: ``"closed"`` | ``"open"`` | ``"half_open"``."""
        self._maybe_transition_to_half_open()
        return self._state

    async def call(self, coro: Awaitable[Any]) -> Any:
        """
        Execute *coro* subject to the circuit breaker policy.

        Raises
        ------
        CircuitOpenError
            Immediately when the circuit is OPEN (or stays OPEN after a
            failed probe in HALF_OPEN).
        """
        # Check state *before* acquiring lock for the fast-path rejection.
        self._maybe_transition_to_half_open()

        async with self._lock:
            self._total_calls += 1

            if self._state == self._STATE_OPEN:
                raise CircuitOpenError(f"Circuit '{self.name}' is OPEN")

            if self._state == self._STATE_HALF_OPEN:
                # Allow exactly one probe; serialise via lock (already held).
                return await self._probe(coro)

            # CLOSED — normal path
            return await self._execute(coro)

    def stats(self) -> dict:
        """Return a snapshot of current circuit-breaker metrics."""
        return {
            "name": self.name,
            "state": self.state,
            "failures": self._failures,
            "total_calls": self._total_calls,
            "open_count": self._open_count,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _maybe_transition_to_half_open(self) -> None:
        """Promote OPEN → HALF_OPEN once recovery_timeout has elapsed."""
        if (
            self._state == self._STATE_OPEN
            and self._opened_at is not None
            and time.monotonic() - self._opened_at >= self.recovery_timeout
        ):
            self._state = self._STATE_HALF_OPEN
            logger.info("circuit_half_open", name=self.name)

    async def _execute(self, coro: Awaitable[Any]) -> Any:
        """Run *coro* in CLOSED state, tracking success/failure."""
        try:
            result = await coro
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    async def _probe(self, coro: Awaitable[Any]) -> Any:
        """Run the single probe in HALF_OPEN state."""
        try:
            result = await coro
            self._reset()
            logger.info("circuit_closed", name=self.name)
            return result
        except Exception:
            self._trip()
            raise CircuitOpenError(f"Circuit '{self.name}' re-opened after failed probe")

    def _on_success(self) -> None:
        self._failures = 0

    def _on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = self._STATE_OPEN
        self._opened_at = time.monotonic()
        self._open_count += 1
        logger.warning("circuit_opened", name=self.name, failures=self._failures)

    def _reset(self) -> None:
        self._state = self._STATE_CLOSED
        self._failures = 0
        self._opened_at = None


# ── Registry ─────────────────────────────────────────────────────────────────

_registry: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """
    Return the singleton ``CircuitBreaker`` for *name*, creating it on first call.

    Uses a plain dict — safe because Python dict operations are GIL-protected
    for CPython and we only ever add (never remove) entries.
    """
    if name not in _registry:
        _registry[name] = CircuitBreaker(name=name)
    return _registry[name]
