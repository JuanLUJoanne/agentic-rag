"""
LLM factory with DummyLLM fallback for CI and tests.

DummyLLM exists so the full workflow can run without API keys — critical
for CI pipelines and local development. Each agent checks isinstance(llm, DummyLLM)
to short-circuit to deterministic hardcoded responses instead of calling a real model.
"""
from __future__ import annotations

import os
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src.observability.tracing import get_tracer, set_span_ok

logger = structlog.get_logger()

# Import OpenAI exception types; fall back to an empty tuple if openai is not installed.
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError

    _RETRYABLE_EXCEPTIONS: tuple[type[Exception], ...] = (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
    )
except ImportError:  # pragma: no cover
    RateLimitError = None  # type: ignore[assignment,misc]
    APIConnectionError = None  # type: ignore[assignment,misc]
    APITimeoutError = None  # type: ignore[assignment,misc]
    _RETRYABLE_EXCEPTIONS = ()


def _log_retry(retry_state: Any) -> None:
    """Log each retry attempt with attempt number and upcoming wait duration."""
    wait = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        "llm_retry",
        attempt=retry_state.attempt_number,
        wait=round(wait, 3),
    )


def _make_retry_decorator() -> Any:
    """Build a tenacity retry decorator that only retries on retryable exceptions."""
    if not _RETRYABLE_EXCEPTIONS:
        # No openai installed — return a no-op decorator
        def _noop(fn: Any) -> Any:
            return fn

        return _noop

    return retry(
        wait=wait_exponential(multiplier=1, min=1, max=60) + wait_random(0, 2),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
        after=_log_retry,
        reraise=True,
    )


_llm_retry = _make_retry_decorator()


class DummyLLM(BaseChatModel):
    """
    Deterministic no-op LLM for tests and CI environments.

    Returns a fixed string so callers can exercise the full agent/graph
    logic without network calls. Agents detect this class via isinstance
    and substitute hardcoded, realistic-looking responses.
    """

    model_name: str = "dummy"
    default_response: str = "Placeholder response"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        with get_tracer().start_as_current_span("llm_call") as span:
            span.set_attribute("llm.model", self.model_name)
            span.set_attribute("llm.input_messages", len(messages))
            message = AIMessage(content=self.default_response)
            span.set_attribute("llm.output_tokens_est", len(self.default_response.split()))
            set_span_ok(span)
            return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop=stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "dummy"


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> BaseChatModel:
    """
    Return the best available LLM given the current environment.

    Falls back to DummyLLM when OPENAI_API_KEY is absent or set to the
    placeholder value from .env.example — keeping CI green without secrets.
    Lazy-imports ChatOpenAI to avoid import cost in test environments.
    """
    _PLACEHOLDER_PREFIXES = ("sk-xxx", "sk-test", "sk-fake", "sk-placeholder", "sk-dummy")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and not any(api_key.startswith(p) for p in _PLACEHOLDER_PREFIXES):
        from langchain_openai import ChatOpenAI

        logger.info("llm_initialized", model=model, provider="openai")
        return ChatOpenAI(model=model, temperature=temperature)

    logger.info("llm_initialized", provider="dummy", reason="no_api_key_configured")
    return DummyLLM()
