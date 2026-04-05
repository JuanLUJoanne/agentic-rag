"""
Tests for src/api/dedup.py — RequestDeduplicator.
"""
from __future__ import annotations

import asyncio

import pytest

import src.api.dedup as _dedup_mod
from src.api.dedup import RequestDeduplicator, get_deduplicator


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fresh_dedup() -> RequestDeduplicator:
    """Return a brand-new RequestDeduplicator (bypasses singleton)."""
    return RequestDeduplicator()


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dedup_runs_once_for_concurrent_calls():
    """
    5 concurrent calls with the same key must trigger the factory exactly once.
    """
    call_count = 0

    async def factory():
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # simulate async work
        return "result"

    dedup = _fresh_dedup()
    tasks = [dedup.get_or_run("same-key", factory) for _ in range(5)]
    results = await asyncio.gather(*tasks)

    assert call_count == 1
    assert all(r == "result" for r in results)


@pytest.mark.asyncio
async def test_dedup_different_keys_run_independently():
    """Two different keys must each execute their own factory."""
    call_log: list[str] = []

    async def factory_a():
        call_log.append("a")
        return "result-a"

    async def factory_b():
        call_log.append("b")
        return "result-b"

    dedup = _fresh_dedup()
    r_a, r_b = await asyncio.gather(
        dedup.get_or_run("key-a", factory_a),
        dedup.get_or_run("key-b", factory_b),
    )

    assert r_a == "result-a"
    assert r_b == "result-b"
    assert "a" in call_log
    assert "b" in call_log


@pytest.mark.asyncio
async def test_dedup_cleans_up_after_completion():
    """After the coroutine resolves, the key must be removed from _inflight."""
    dedup = _fresh_dedup()

    async def factory():
        return 42

    await dedup.get_or_run("cleanup-key", factory)
    assert "cleanup-key" not in dedup._inflight


# ── Singleton ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dedup_singleton():
    """get_deduplicator() must return the same instance."""
    _dedup_mod._instance = None  # reset for test isolation
    d1 = get_deduplicator()
    d2 = get_deduplicator()
    assert d1 is d2
