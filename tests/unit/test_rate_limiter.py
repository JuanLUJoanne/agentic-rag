"""Unit tests for TokenBucketRateLimiter."""
from __future__ import annotations

import asyncio
import time

import pytest

from src.gateway.rate_limiter import TokenBucketRateLimiter

# ── Helpers ────────────────────────────────────────────────────────────────────

def _drain(limiter: TokenBucketRateLimiter, model: str) -> None:
    """Drain the bucket to zero so the next call must wait."""
    bucket = limiter._get_bucket(model)
    bucket._request_tokens = 0.0
    bucket._text_tokens = 0.0
    bucket._last_refill = time.monotonic()


def _fill(limiter: TokenBucketRateLimiter, model: str, tokens: float = 1000.0) -> None:
    """Restore enough capacity for a single request."""
    bucket = limiter._get_bucket(model)
    bucket._request_tokens = 5.0
    bucket._text_tokens = tokens
    bucket._last_refill = time.monotonic()


# ── Tests ──────────────────────────────────────────────────────────────────────


async def test_capacity_decreases_after_use() -> None:
    """Consuming capacity should reduce available tokens."""
    limiter = TokenBucketRateLimiter()
    bucket = limiter._get_bucket("gpt-4o-mini")

    before_text = bucket._text_tokens
    before_req = bucket._request_tokens

    await limiter.wait_for_capacity("gpt-4o-mini", 1_000)

    # Tokens should have decreased
    assert bucket._text_tokens < before_text
    assert bucket._request_tokens < before_req


async def test_wait_blocks_when_empty() -> None:
    """wait_for_capacity must not return immediately when the bucket is empty."""
    limiter = TokenBucketRateLimiter()
    _drain(limiter, "gpt-4o")

    done = False

    async def _waiter() -> None:
        nonlocal done
        await limiter.wait_for_capacity("gpt-4o", 1)
        done = True

    task = asyncio.create_task(_waiter())
    # Give the event loop one turn — task should still be blocked
    await asyncio.sleep(0)
    assert not done, "wait_for_capacity returned immediately on empty bucket"

    # Refill the bucket; the next poll (≤0.05 s) should unblock the task
    _fill(limiter, "gpt-4o")
    await asyncio.wait_for(task, timeout=1.0)
    assert done


async def test_429_halves_rate() -> None:
    """on_429 should cap available tokens to half the model limits."""
    limiter = TokenBucketRateLimiter()
    bucket = limiter._get_bucket("gpt-4o")

    # Start fully loaded
    assert bucket._request_tokens == pytest.approx(bucket.rpm, abs=1)
    assert bucket._text_tokens == pytest.approx(bucket.tpm, abs=1)

    limiter.on_429("gpt-4o")

    # Tokens capped to half
    assert bucket._request_tokens <= bucket.rpm / 2 + 1
    assert bucket._text_tokens <= bucket.tpm / 2 + 1

    # Effective rates are halved while throttled
    assert bucket.effective_rpm == pytest.approx(bucket.rpm / 2)
    assert bucket.effective_tpm == pytest.approx(bucket.tpm / 2)


async def test_throttled_flag() -> None:
    """is_throttled returns True immediately after on_429."""
    limiter = TokenBucketRateLimiter()

    assert not limiter.is_throttled("gpt-4o-mini")

    limiter.on_429("gpt-4o-mini")
    assert limiter.is_throttled("gpt-4o-mini")


async def test_unknown_model_uses_default_config() -> None:
    """Unknown model IDs should fall back to the default config."""
    limiter = TokenBucketRateLimiter()
    # Should not raise
    await limiter.wait_for_capacity("my-custom-model", 10)
    bucket = limiter._get_bucket("my-custom-model")
    assert bucket.rpm == 60
    assert bucket.tpm == 40_000


async def test_capacity_consumed_matches_tokens() -> None:
    """Text tokens consumed should equal the requested token count."""
    limiter = TokenBucketRateLimiter()
    bucket = limiter._get_bucket("gpt-4o-mini")

    before = bucket._text_tokens
    # Force a known starting point
    bucket._last_refill = time.monotonic()

    await limiter.wait_for_capacity("gpt-4o-mini", 5_000)

    # Elapsed since we set _last_refill is tiny; refill is negligible
    assert before - bucket._text_tokens == pytest.approx(5_000, abs=50)


async def test_concurrent_requests_do_not_double_consume() -> None:
    """Concurrent wait_for_capacity calls must not both consume the same tokens."""
    limiter = TokenBucketRateLimiter()
    bucket = limiter._get_bucket("gpt-4o-mini")
    bucket._last_refill = time.monotonic()
    initial = bucket._text_tokens

    await asyncio.gather(
        limiter.wait_for_capacity("gpt-4o-mini", 1_000),
        limiter.wait_for_capacity("gpt-4o-mini", 1_000),
    )

    consumed = initial - bucket._text_tokens
    # Two requests × 1000 tokens each, small refill tolerance
    assert consumed == pytest.approx(2_000, abs=100)
