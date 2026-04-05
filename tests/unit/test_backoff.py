"""
Tests for exponential backoff + jitter on LLM calls (src/utils/llm.py).
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a thin callable that wraps _llm_retry for testing purposes.
# We do NOT rely on the internal structure of DummyLLM or ChatOpenAI; instead
# we create a standalone function decorated with _llm_retry and mock it.
# ---------------------------------------------------------------------------


def _get_retryable_exceptions():
    """Return the retryable exception tuple from llm module."""
    from src.utils.llm import _RETRYABLE_EXCEPTIONS
    return _RETRYABLE_EXCEPTIONS


def _make_decorated_fn(side_effects):
    """
    Return a function decorated with _llm_retry whose calls raise the given
    side_effects in sequence (last item is returned if not an exception).
    """
    from src.utils.llm import _llm_retry

    call_count = {"n": 0}
    effects = list(side_effects)

    @_llm_retry
    def _fn():
        idx = call_count["n"]
        call_count["n"] += 1
        effect = effects[idx] if idx < len(effects) else effects[-1]
        if isinstance(effect, Exception):
            raise effect
        return effect

    return _fn, call_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackoffRetry:
    def test_retry_on_rate_limit(self):
        """Mock LLM call raises RateLimitError twice then succeeds; called 3 times total."""
        retryable = _get_retryable_exceptions()
        if not retryable:
            pytest.skip("openai not installed; retry decorator is a no-op")

        from openai import RateLimitError

        # RateLimitError requires a message, response, and body
        err1 = RateLimitError(
            "rate limit",
            response=_mock_response(429),
            body={"error": {"message": "rate limit"}},
        )
        err2 = RateLimitError(
            "rate limit",
            response=_mock_response(429),
            body={"error": {"message": "rate limit"}},
        )

        fn, call_count = _make_decorated_fn([err1, err2, "success"])
        result = fn()
        assert result == "success"
        assert call_count["n"] == 3

    def test_no_retry_on_value_error(self):
        """ValueError should not trigger retry — raises immediately."""
        fn, call_count = _make_decorated_fn([ValueError("bad input"), "never_reached"])
        with pytest.raises(ValueError, match="bad input"):
            fn()
        assert call_count["n"] == 1

    def test_max_retries_exhausted_reraises(self):
        """After 3 consecutive failures the original exception propagates."""
        retryable = _get_retryable_exceptions()
        if not retryable:
            pytest.skip("openai not installed; retry decorator is a no-op")

        from openai import RateLimitError

        def _make_err():
            return RateLimitError(
                "rate limit",
                response=_mock_response(429),
                body={"error": {"message": "rate limit"}},
            )

        fn, call_count = _make_decorated_fn([_make_err(), _make_err(), _make_err()])
        with pytest.raises(RateLimitError):
            fn()
        assert call_count["n"] == 3


# ---------------------------------------------------------------------------
# Minimal mock helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int):
    """Return a minimal httpx.Response-like object for OpenAI error construction."""
    try:
        import httpx

        return httpx.Response(status_code, request=httpx.Request("POST", "https://api.openai.com"))
    except ImportError:
        # Fallback minimal mock
        class _FakeResponse:
            def __init__(self, code):
                self.status_code = code
                self.headers = {}
                self.request = type("Req", (), {"method": "POST", "url": "https://api.openai.com"})()

        return _FakeResponse(status_code)
