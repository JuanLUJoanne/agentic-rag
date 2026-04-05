"""
Token-bucket rate limiter for LLM API calls.

Each model has two buckets — requests-per-minute (RPM) and tokens-per-minute
(TPM) — that refill continuously based on elapsed wall-clock time.

Design choices:
  - Time-based refill: no background task needed; tokens accumulate on each
    ``has_capacity`` check based on ``time.monotonic()`` delta.
  - asyncio.Lock: ensures check-and-consume is atomic inside the event loop
    without blocking the thread.
  - on_429: immediately halves the effective rate for 60 s; the _Bucket caps
    current tokens to the new lower limit so the halved rate takes effect at
    once rather than after a full drain.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

_MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "gpt-4o-mini": {"rpm": 500, "tpm": 200_000},
    "gpt-4o": {"rpm": 100, "tpm": 30_000},
}

_DEFAULT_CONFIG: dict[str, int] = {"rpm": 60, "tpm": 40_000}

_THROTTLE_SECONDS = 60.0
_POLL_INTERVAL = 0.05  # seconds between capacity checks


@dataclass
class _Bucket:
    """Single token-bucket pair for one model."""

    rpm: int
    tpm: int
    _request_tokens: float = field(init=False)
    _text_tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _throttle_until: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._request_tokens = float(self.rpm)
        self._text_tokens = float(self.tpm)
        self._last_refill = time.monotonic()

    # ── Throttle state ──────────────────────────────────────────────────────

    def is_throttled(self) -> bool:
        return time.monotonic() < self._throttle_until

    @property
    def effective_rpm(self) -> float:
        return self.rpm / 2.0 if self.is_throttled() else float(self.rpm)

    @property
    def effective_tpm(self) -> float:
        return self.tpm / 2.0 if self.is_throttled() else float(self.tpm)

    # ── Token management ────────────────────────────────────────────────────

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        eff_rpm = self.effective_rpm
        eff_tpm = self.effective_tpm
        self._request_tokens = min(
            eff_rpm, self._request_tokens + elapsed * eff_rpm / 60.0
        )
        self._text_tokens = min(
            eff_tpm, self._text_tokens + elapsed * eff_tpm / 60.0
        )

    def has_capacity(self, tokens: int) -> bool:
        self._refill()
        return self._request_tokens >= 1.0 and self._text_tokens >= float(tokens)

    def consume(self, tokens: int) -> None:
        self._request_tokens -= 1.0
        self._text_tokens -= float(tokens)

    def on_429(self) -> None:
        self._throttle_until = time.monotonic() + _THROTTLE_SECONDS
        # Immediately cap current tokens to the halved limits
        self._request_tokens = min(self._request_tokens, self.rpm / 2.0)
        self._text_tokens = min(self._text_tokens, self.tpm / 2.0)


class TokenBucketRateLimiter:
    """
    Per-model token-bucket rate limiter.

    Usage::

        limiter = TokenBucketRateLimiter()
        await limiter.wait_for_capacity("gpt-4o-mini", estimated_tokens=300)
        # ... make LLM call ...
        # On HTTP 429:
        limiter.on_429("gpt-4o-mini")
    """

    def __init__(self, configs: dict[str, dict[str, int]] | None = None) -> None:
        self._configs = configs or _MODEL_CONFIGS
        self._buckets: dict[str, _Bucket] = {}
        self._lock = asyncio.Lock()

    def _get_bucket(self, model_id: str) -> _Bucket:
        if model_id not in self._buckets:
            cfg = self._configs.get(model_id, _DEFAULT_CONFIG)
            self._buckets[model_id] = _Bucket(rpm=cfg["rpm"], tpm=cfg["tpm"])
        return self._buckets[model_id]

    async def wait_for_capacity(self, model_id: str, estimated_tokens: int) -> None:
        """Block until the model bucket has capacity, then consume tokens."""
        bucket = self._get_bucket(model_id)
        while True:
            async with self._lock:
                if bucket.has_capacity(estimated_tokens):
                    bucket.consume(estimated_tokens)
                    logger.debug(
                        "capacity_wait",
                        model=model_id,
                        tokens=estimated_tokens,
                        status="acquired",
                    )
                    return
            await asyncio.sleep(_POLL_INTERVAL)

    def on_429(self, model_id: str) -> None:
        """React to an HTTP 429 by halving the effective rate for 60 s."""
        bucket = self._get_bucket(model_id)
        bucket.on_429()
        logger.warning(
            "throttle_start",
            model=model_id,
            duration_seconds=_THROTTLE_SECONDS,
            new_rpm=bucket.effective_rpm,
            new_tpm=bucket.effective_tpm,
        )

    def is_throttled(self, model_id: str) -> bool:
        """Return True while the model is in the post-429 throttle window."""
        throttled = self._get_bucket(model_id).is_throttled()
        if not throttled and model_id in self._buckets:
            # Log throttle end exactly once by checking the prior state
            bucket = self._buckets[model_id]
            if bucket._throttle_until > 0 and not bucket.is_throttled():
                logger.info("throttle_end", model=model_id)
        return throttled


# ── Module-level singleton ──────────────────────────────────────────────────

_default_limiter: TokenBucketRateLimiter | None = None


def get_default_rate_limiter() -> TokenBucketRateLimiter:
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = TokenBucketRateLimiter()
    return _default_limiter


# ── Tenant-aware rate limiter ───────────────────────────────────────────────


class TenantAwareRateLimiter:
    """
    Per-tenant, per-model token-bucket rate limiter.

    Each tenant gets its own isolated set of buckets. Tenants are created
    lazily on first access. A per-tenant asyncio.Lock ensures atomicity of
    check-and-consume without cross-tenant contention.

    Usage::

        limiter = TenantAwareRateLimiter()
        await limiter.wait_for_capacity("tenant_abc", "gpt-4o-mini", 300)
        # On HTTP 429 from this tenant:
        limiter.on_429("tenant_abc", "gpt-4o-mini")
    """

    def __init__(
        self,
        configs: dict | None = None,
        default_rpm: int = 60,
        default_tpm: int = 40_000,
    ) -> None:
        self._configs = configs or {}
        self._default_rpm = default_rpm
        self._default_tpm = default_tpm
        # {tenant_id: {model_id: _Bucket}}
        self._tenant_buckets: dict[str, dict[str, _Bucket]] = {}
        # per-tenant asyncio.Lock (created on first access)
        self._tenant_locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, tenant_id: str) -> asyncio.Lock:
        if tenant_id not in self._tenant_locks:
            self._tenant_locks[tenant_id] = asyncio.Lock()
        return self._tenant_locks[tenant_id]

    def _get_bucket(self, tenant_id: str, model_id: str) -> _Bucket:
        if tenant_id not in self._tenant_buckets:
            self._tenant_buckets[tenant_id] = {}
        tenant = self._tenant_buckets[tenant_id]
        if model_id not in tenant:
            # Tenant-level config overrides, then model-level defaults, then global default
            tenant_cfg = self._configs.get(tenant_id, {})
            model_cfg = tenant_cfg.get(model_id, {})
            rpm = model_cfg.get("rpm", self._default_rpm)
            tpm = model_cfg.get("tpm", self._default_tpm)
            tenant[model_id] = _Bucket(rpm=rpm, tpm=tpm)
        return tenant[model_id]

    async def wait_for_capacity(
        self, tenant_id: str, model_id: str, estimated_tokens: int
    ) -> None:
        """Block until the tenant+model bucket has capacity, then consume tokens."""
        lock = self._get_lock(tenant_id)
        while True:
            async with lock:
                bucket = self._get_bucket(tenant_id, model_id)
                if bucket.has_capacity(estimated_tokens):
                    bucket.consume(estimated_tokens)
                    logger.debug(
                        "tenant_capacity_acquired",
                        tenant=tenant_id,
                        model=model_id,
                        tokens=estimated_tokens,
                    )
                    return
            await asyncio.sleep(_POLL_INTERVAL)

    def on_429(self, tenant_id: str, model_id: str) -> None:
        """React to an HTTP 429 for a specific tenant+model."""
        bucket = self._get_bucket(tenant_id, model_id)
        bucket.on_429()
        logger.warning(
            "tenant_throttle_start",
            tenant=tenant_id,
            model=model_id,
            duration_seconds=_THROTTLE_SECONDS,
            new_rpm=bucket.effective_rpm,
            new_tpm=bucket.effective_tpm,
        )

    def tenant_stats(self, tenant_id: str) -> dict:
        """Return per-model bucket state for the given tenant."""
        if tenant_id not in self._tenant_buckets:
            return {}
        result = {}
        for model_id, bucket in self._tenant_buckets[tenant_id].items():
            result[model_id] = {
                "rpm": bucket.rpm,
                "tpm": bucket.tpm,
                "request_tokens": bucket._request_tokens,
                "text_tokens": bucket._text_tokens,
                "throttled": bucket.is_throttled(),
                "effective_rpm": bucket.effective_rpm,
                "effective_tpm": bucket.effective_tpm,
            }
        return result


# ── Tenant singleton ────────────────────────────────────────────────────────

_tenant_limiter: TenantAwareRateLimiter | None = None


def get_tenant_rate_limiter() -> TenantAwareRateLimiter:
    global _tenant_limiter
    if _tenant_limiter is None:
        _tenant_limiter = TenantAwareRateLimiter()
    return _tenant_limiter
