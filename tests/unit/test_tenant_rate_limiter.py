"""
Tests for per-tenant rate limiting (src/gateway/rate_limiter.py).
"""
from __future__ import annotations

import pytest

from src.gateway.rate_limiter import TenantAwareRateLimiter, _Bucket


class TestTenantIsolation:
    """Tenant A's exhausted bucket must not block tenant B."""

    @pytest.mark.asyncio
    async def test_tenant_isolation(self):
        limiter = TenantAwareRateLimiter(default_rpm=1, default_tpm=100)

        # Drain tenant A's request quota (rpm=1 means 1 request token available)
        await limiter.wait_for_capacity("tenantA", "gpt-4o-mini", 10)

        # Tenant A is now exhausted — but tenant B should still have full capacity
        # We verify by checking that tenant B's bucket has capacity independently
        bucket_b = limiter._get_bucket("tenantB", "gpt-4o-mini")
        assert bucket_b.has_capacity(10), "Tenant B should have capacity unaffected by tenant A"

    @pytest.mark.asyncio
    async def test_same_tenant_shares_bucket(self):
        """Two calls from the same tenant share the rate limit."""
        limiter = TenantAwareRateLimiter(default_rpm=100, default_tpm=1000)

        # Make one call to initialise the bucket
        await limiter.wait_for_capacity("tenantX", "gpt-4o-mini", 50)

        # The bucket should have been decremented
        bucket = limiter._get_bucket("tenantX", "gpt-4o-mini")
        # After 1 request, _request_tokens should be rpm - 1 = 99
        assert bucket._request_tokens < 100, "Bucket tokens should decrease after first call"
        # Text tokens should also be decremented by 50
        assert bucket._text_tokens < 1000, "Text tokens should decrease after first call"

    def test_on_429_affects_only_tenant(self):
        """on_429 for tenantA does not change tenantB's bucket state."""
        limiter = TenantAwareRateLimiter(default_rpm=60, default_tpm=40_000)

        # Force bucket creation for both tenants
        bucket_a = limiter._get_bucket("tenantA", "gpt-4o-mini")
        bucket_b = limiter._get_bucket("tenantB", "gpt-4o-mini")

        # Record initial state for B
        initial_b_rpm = bucket_b._request_tokens
        initial_b_tpm = bucket_b._text_tokens

        # Trigger 429 for A
        limiter.on_429("tenantA", "gpt-4o-mini")

        # A should be throttled
        assert bucket_a.is_throttled(), "Tenant A should be throttled after 429"

        # B should be unaffected
        assert not bucket_b.is_throttled(), "Tenant B should not be throttled"
        assert bucket_b._request_tokens == initial_b_rpm, "Tenant B rpm tokens unchanged"
        assert bucket_b._text_tokens == initial_b_tpm, "Tenant B tpm tokens unchanged"
