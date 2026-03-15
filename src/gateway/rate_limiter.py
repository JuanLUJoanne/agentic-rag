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
