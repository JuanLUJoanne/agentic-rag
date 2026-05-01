"""
Unified LLM Gateway — single entry point for all LLM calls.

Routes requests to the correct provider based on model name, applies
middleware (logging, cost tracking, rate limiting), and integrates with
the existing circuit breaker infrastructure.

Usage::

    gw = get_gateway()
    response = await gw.complete("Summarise this document.", model="gpt-4o-mini")

The gateway wraps existing provider implementations (``src/utils/llm.py``)
so all current code continues to work unchanged.
"""
from __future__ import annotations

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

import structlog

from src.gateway.cost_tracker import CostTracker, get_default_tracker
from src.gateway.model_registry import ModelInfo, get_model_info
from src.gateway.rate_limiter import TokenBucketRateLimiter, get_default_rate_limiter
from src.observability.tracing import get_tracer, set_span_error, set_span_ok
from src.retrieval.circuit_breaker import get_circuit_breaker
from src.utils.llm import get_llm

logger = structlog.get_logger()


# ── Response ──────────────────────────────────────────────────────────────────


@dataclass
class GatewayResponse:
    """Unified response from any LLM provider."""

    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: Decimal
    correlation_id: str


# ── Stats collector ───────────────────────────────────────────────────────────


class _GatewayStats:
    """In-memory stats aggregated per model."""

    def __init__(self) -> None:
        self.request_count: dict[str, int] = defaultdict(int)
        self.error_count: dict[str, int] = defaultdict(int)
        self.total_latency_ms: dict[str, float] = defaultdict(float)
        self.total_cost: Decimal = Decimal("0")

    def record(self, model: str, latency_ms: float, cost: Decimal) -> None:
        self.request_count[model] += 1
        self.total_latency_ms[model] += latency_ms
        self.total_cost += cost

    def record_error(self, model: str) -> None:
        self.error_count[model] += 1

    def snapshot(self) -> dict:
        models = {}
        for model in set(self.request_count) | set(self.error_count):
            count = self.request_count.get(model, 0)
            models[model] = {
                "requests": count,
                "errors": self.error_count.get(model, 0),
                "avg_latency_ms": round(
                    self.total_latency_ms.get(model, 0) / count, 2
                )
                if count
                else 0,
            }
        return {
            "models": models,
            "total_cost": float(self.total_cost),
        }


# ── Middleware protocol ───────────────────────────────────────────────────────


@dataclass
class GatewayRequest:
    """Internal request object passed through the middleware chain."""

    prompt: str
    model: str
    model_info: ModelInfo
    correlation_id: str
    kwargs: dict = field(default_factory=dict)


class Middleware:
    """Base class for gateway middleware."""

    async def before(self, request: GatewayRequest) -> None:
        pass

    async def after(
        self, request: GatewayRequest, response: GatewayResponse
    ) -> None:
        pass

    async def on_error(
        self, request: GatewayRequest, exc: BaseException
    ) -> None:
        pass


class LoggingMiddleware(Middleware):
    """Log every request and response with correlation ID."""

    async def before(self, request: GatewayRequest) -> None:
        logger.info(
            "gateway_request",
            correlation_id=request.correlation_id,
            model=request.model,
            provider=request.model_info.provider,
            prompt_len=len(request.prompt),
        )

    async def after(
        self, request: GatewayRequest, response: GatewayResponse
    ) -> None:
        logger.info(
            "gateway_response",
            correlation_id=response.correlation_id,
            model=response.model,
            latency_ms=response.latency_ms,
            output_len=len(response.content),
            cost=float(response.cost),
        )

    async def on_error(
        self, request: GatewayRequest, exc: BaseException
    ) -> None:
        logger.error(
            "gateway_error",
            correlation_id=request.correlation_id,
            model=request.model,
            error=str(exc),
        )


class CostMiddleware(Middleware):
    """Accumulate costs in the global CostTracker."""

    def __init__(self, tracker: CostTracker | None = None) -> None:
        self._tracker = tracker

    def _get_tracker(self) -> CostTracker:
        return self._tracker or get_default_tracker()

    async def after(
        self, request: GatewayRequest, response: GatewayResponse
    ) -> None:
        from src.gateway.cost_tracker import BudgetExceededError

        try:
            self._get_tracker().record_usage(
                model_id=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                query_id=request.correlation_id,
            )
        except BudgetExceededError:
            logger.warning(
                "gateway_budget_exceeded",
                correlation_id=request.correlation_id,
                model=response.model,
            )


class RateLimitMiddleware(Middleware):
    """Enforce per-model rate limits before each call."""

    def __init__(self, limiter: TokenBucketRateLimiter | None = None) -> None:
        self._limiter = limiter

    def _get_limiter(self) -> TokenBucketRateLimiter:
        return self._limiter or get_default_rate_limiter()

    async def before(self, request: GatewayRequest) -> None:
        estimated_tokens = len(request.prompt.split()) + 100
        await self._get_limiter().wait_for_capacity(
            request.model, estimated_tokens
        )


# ── Gateway ───────────────────────────────────────────────────────────────────


class LLMGateway:
    """
    Unified proxy for all LLM providers.

    Wraps existing ``get_llm()`` and adds middleware, circuit breakers,
    and per-model stats collection.
    """

    def __init__(
        self,
        middleware: list[Middleware] | None = None,
        default_model: str = "gpt-4o-mini",
    ) -> None:
        self._middleware = middleware or [
            LoggingMiddleware(),
            RateLimitMiddleware(),
            CostMiddleware(),
        ]
        self._default_model = default_model
        self._stats = _GatewayStats()

    # ── Public API ────────────────────────────────────────────────────────

    async def complete(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: object,
    ) -> GatewayResponse:
        """Send a prompt to the resolved provider and return a unified response."""
        model = model or self._default_model
        model_info = get_model_info(model)
        correlation_id = uuid.uuid4().hex[:12]

        request = GatewayRequest(
            prompt=prompt,
            model=model,
            model_info=model_info,
            correlation_id=correlation_id,
            kwargs=dict(kwargs),
        )

        with get_tracer().start_as_current_span("gateway_complete") as span:
            span.set_attribute("gateway.model", model)
            span.set_attribute("gateway.provider", model_info.provider)
            span.set_attribute("gateway.correlation_id", correlation_id)

            # Run before-middleware
            for mw in self._middleware:
                await mw.before(request)

            try:
                response = await self._call_provider(request, span)
            except Exception as exc:
                self._stats.record_error(model)
                for mw in self._middleware:
                    await mw.on_error(request, exc)
                set_span_error(span, exc)
                raise

            # Run after-middleware
            for mw in self._middleware:
                await mw.after(request, response)

            self._stats.record(model, response.latency_ms, response.cost)
            span.set_attribute("gateway.latency_ms", response.latency_ms)
            span.set_attribute("gateway.cost", float(response.cost))
            set_span_ok(span)
            return response

    async def complete_structured(
        self,
        prompt: str,
        schema: dict,
        model: str | None = None,
        **kwargs: object,
    ) -> GatewayResponse:
        """Complete with a JSON schema hint appended to the prompt.

        For DummyLLM this just passes through to ``complete``.
        For real providers, append schema instructions to the prompt.
        """
        schema_hint = f"\n\nRespond with JSON matching this schema: {schema}"
        return await self.complete(prompt + schema_hint, model=model, **kwargs)

    def stats(self) -> dict:
        """Return aggregated gateway statistics."""
        snapshot = self._stats.snapshot()

        # Add circuit breaker states for known models
        cb_states = {}
        for model in MODEL_REGISTRY:
            try:
                cb = get_circuit_breaker(f"llm_{model}")
                cb_states[model] = cb.state
            except Exception:
                pass
        snapshot["circuit_breakers"] = cb_states
        return snapshot

    # ── Internal ──────────────────────────────────────────────────────────

    async def _call_provider(
        self, request: GatewayRequest, span: object
    ) -> GatewayResponse:
        """Dispatch to the appropriate LLM provider via circuit breaker."""
        model = request.model
        provider = request.model_info.provider
        cb = get_circuit_breaker(f"llm_{model}")

        start = time.monotonic()

        async def _invoke() -> str:
            llm = get_llm(model=model)
            result = await llm.ainvoke(request.prompt)
            return result.content

        content = await cb.call(_invoke())
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)

        # Estimate tokens (real provider would return actual counts)
        input_tokens = len(request.prompt.split())
        output_tokens = len(content.split())
        cost = (
            request.model_info.cost_per_1k_input
            * Decimal(str(input_tokens))
            / Decimal("1000")
            + request.model_info.cost_per_1k_output
            * Decimal(str(output_tokens))
            / Decimal("1000")
        )

        return GatewayResponse(
            content=content,
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
            cost=cost,
            correlation_id=request.correlation_id,
        )


# ── Registry import (avoid circular) ─────────────────────────────────────────
from src.gateway.model_registry import MODEL_REGISTRY  # noqa: E402

# ── Module-level singleton ────────────────────────────────────────────────────

_gateway: LLMGateway | None = None


def get_gateway() -> LLMGateway:
    """Return (or lazily create) the module-level LLM gateway."""
    global _gateway
    if _gateway is None:
        _gateway = LLMGateway()
    return _gateway
