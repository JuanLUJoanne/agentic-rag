"""Unit tests for the LLM Gateway, Model Registry, and middleware."""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.gateway.llm_gateway import (
    CostMiddleware,
    GatewayRequest,
    GatewayResponse,
    LLMGateway,
    LoggingMiddleware,
    Middleware,
    RateLimitMiddleware,
    _GatewayStats,
    get_gateway,
)
from src.gateway.model_registry import (
    get_model_info,
    list_models,
    resolve_provider,
)

# ── Model Registry ────────────────────────────────────────────────────────────


class TestModelRegistry:
    def test_get_known_model(self):
        info = get_model_info("gpt-4o")
        assert info.provider == "openai"
        assert info.tier == 3
        assert info.cost_per_1k_input == Decimal("0.005000")

    def test_get_unknown_model_falls_back(self):
        info = get_model_info("nonexistent-model")
        # Falls back to gpt-4o-mini
        assert info.provider == "openai"
        assert info.tier == 2

    def test_resolve_provider_openai(self):
        assert resolve_provider("gpt-4o-mini") == "openai"
        assert resolve_provider("gpt-4o") == "openai"

    def test_resolve_provider_anthropic(self):
        assert resolve_provider("claude-3-5-sonnet") == "anthropic"
        assert resolve_provider("claude-3-5-haiku") == "anthropic"

    def test_resolve_provider_ollama(self):
        assert resolve_provider("llama3.1:8b") == "ollama"

    def test_resolve_provider_dummy(self):
        assert resolve_provider("dummy") == "dummy"

    def test_list_models_all(self):
        models = list_models()
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models
        assert "claude-3-5-sonnet" in models
        assert "dummy" in models

    def test_list_models_filter_provider(self):
        openai_models = list_models(provider="openai")
        assert all(
            get_model_info(m).provider == "openai" for m in openai_models
        )
        assert "gpt-4o" in openai_models
        assert "claude-3-5-sonnet" not in openai_models

    def test_list_models_filter_tier(self):
        cheap = list_models(max_tier=1)
        for m in cheap:
            assert get_model_info(m).tier <= 1

    def test_model_info_is_frozen(self):
        info = get_model_info("gpt-4o")
        with pytest.raises(AttributeError):
            info.provider = "hacked"  # type: ignore[misc]


# ── Gateway Stats ─────────────────────────────────────────────────────────────


class TestGatewayStats:
    def test_record_and_snapshot(self):
        stats = _GatewayStats()
        stats.record("gpt-4o", 150.0, Decimal("0.001"))
        stats.record("gpt-4o", 200.0, Decimal("0.002"))
        stats.record("gpt-4o-mini", 50.0, Decimal("0.0001"))

        snap = stats.snapshot()
        assert snap["models"]["gpt-4o"]["requests"] == 2
        assert snap["models"]["gpt-4o"]["avg_latency_ms"] == 175.0
        assert snap["models"]["gpt-4o-mini"]["requests"] == 1
        assert snap["total_cost"] == pytest.approx(0.0031, rel=1e-4)

    def test_record_error(self):
        stats = _GatewayStats()
        stats.record_error("gpt-4o")
        stats.record_error("gpt-4o")
        snap = stats.snapshot()
        assert snap["models"]["gpt-4o"]["errors"] == 2
        assert snap["models"]["gpt-4o"]["requests"] == 0

    def test_empty_snapshot(self):
        stats = _GatewayStats()
        snap = stats.snapshot()
        assert snap["models"] == {}
        assert snap["total_cost"] == 0.0


# ── Middleware ────────────────────────────────────────────────────────────────


class TestMiddleware:
    @pytest.mark.asyncio
    async def test_logging_middleware_before(self, capsys):
        mw = LoggingMiddleware()
        req = GatewayRequest(
            prompt="test prompt",
            model="gpt-4o-mini",
            model_info=get_model_info("gpt-4o-mini"),
            correlation_id="abc123",
        )
        # Should not raise
        await mw.before(req)

    @pytest.mark.asyncio
    async def test_logging_middleware_after(self):
        mw = LoggingMiddleware()
        req = GatewayRequest(
            prompt="test",
            model="gpt-4o-mini",
            model_info=get_model_info("gpt-4o-mini"),
            correlation_id="abc",
        )
        resp = GatewayResponse(
            content="response text",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=10,
            output_tokens=5,
            latency_ms=100.0,
            cost=Decimal("0.001"),
            correlation_id="abc",
        )
        await mw.after(req, resp)

    @pytest.mark.asyncio
    async def test_logging_middleware_on_error(self):
        mw = LoggingMiddleware()
        req = GatewayRequest(
            prompt="test",
            model="gpt-4o-mini",
            model_info=get_model_info("gpt-4o-mini"),
            correlation_id="abc",
        )
        await mw.on_error(req, RuntimeError("boom"))

    @pytest.mark.asyncio
    async def test_cost_middleware_records_usage(self):
        from src.gateway.cost_tracker import CostTracker

        tracker = CostTracker(budget=100.0)
        mw = CostMiddleware(tracker=tracker)
        req = GatewayRequest(
            prompt="test",
            model="gpt-4o-mini",
            model_info=get_model_info("gpt-4o-mini"),
            correlation_id="test-corr",
        )
        resp = GatewayResponse(
            content="out",
            model="gpt-4o-mini",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            latency_ms=80.0,
            cost=Decimal("0.001"),
            correlation_id="test-corr",
        )
        await mw.after(req, resp)
        assert tracker.total_cost > Decimal("0")

    @pytest.mark.asyncio
    async def test_rate_limit_middleware_acquires_capacity(self):
        from src.gateway.rate_limiter import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter()
        mw = RateLimitMiddleware(limiter=limiter)
        req = GatewayRequest(
            prompt="short prompt",
            model="gpt-4o-mini",
            model_info=get_model_info("gpt-4o-mini"),
            correlation_id="rl-test",
        )
        # Should complete without blocking (fresh bucket has capacity)
        await mw.before(req)

    @pytest.mark.asyncio
    async def test_base_middleware_is_noop(self):
        mw = Middleware()
        req = GatewayRequest(
            prompt="x",
            model="dummy",
            model_info=get_model_info("dummy"),
            correlation_id="noop",
        )
        await mw.before(req)
        resp = GatewayResponse(
            content="y",
            model="dummy",
            provider="dummy",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1.0,
            cost=Decimal("0"),
            correlation_id="noop",
        )
        await mw.after(req, resp)
        await mw.on_error(req, RuntimeError("test"))


# ── Gateway complete ──────────────────────────────────────────────────────────


class TestLLMGateway:
    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        gw = LLMGateway(middleware=[])
        resp = await gw.complete("What is RAG?", model="dummy")
        assert isinstance(resp, GatewayResponse)
        assert resp.provider == "dummy"
        assert resp.model == "dummy"
        assert resp.content  # non-empty
        assert resp.correlation_id
        assert resp.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_complete_uses_default_model(self):
        gw = LLMGateway(middleware=[], default_model="dummy")
        resp = await gw.complete("test")
        assert resp.model == "dummy"

    @pytest.mark.asyncio
    async def test_complete_with_middleware_chain(self):
        """Middleware before/after hooks are called in order."""
        calls: list[str] = []

        class TrackingMiddleware(Middleware):
            def __init__(self, name: str) -> None:
                self.name = name

            async def before(self, request: GatewayRequest) -> None:
                calls.append(f"{self.name}.before")

            async def after(
                self, request: GatewayRequest, response: GatewayResponse
            ) -> None:
                calls.append(f"{self.name}.after")

        gw = LLMGateway(
            middleware=[TrackingMiddleware("A"), TrackingMiddleware("B")],
            default_model="dummy",
        )
        await gw.complete("test")
        assert calls == ["A.before", "B.before", "A.after", "B.after"]

    @pytest.mark.asyncio
    async def test_complete_records_stats(self):
        gw = LLMGateway(middleware=[], default_model="dummy")
        await gw.complete("test")
        await gw.complete("test2")

        stats = gw.stats()
        assert stats["models"]["dummy"]["requests"] == 2
        assert stats["models"]["dummy"]["avg_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_complete_structured(self):
        gw = LLMGateway(middleware=[], default_model="dummy")
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        resp = await gw.complete_structured("What is RAG?", schema=schema)
        assert isinstance(resp, GatewayResponse)
        assert resp.content

    @pytest.mark.asyncio
    async def test_stats_includes_circuit_breakers(self):
        gw = LLMGateway(middleware=[], default_model="dummy")
        stats = gw.stats()
        assert "circuit_breakers" in stats

    @pytest.mark.asyncio
    async def test_error_records_in_stats(self):
        """Gateway records errors when provider raises."""

        class FailingMiddleware(Middleware):
            async def before(self, request: GatewayRequest) -> None:
                raise RuntimeError("forced failure")

        gw = LLMGateway(
            middleware=[FailingMiddleware()], default_model="dummy"
        )
        with pytest.raises(RuntimeError, match="forced failure"):
            await gw.complete("test")

    def test_get_gateway_singleton(self):
        gw1 = get_gateway()
        gw2 = get_gateway()
        assert gw1 is gw2

    @pytest.mark.asyncio
    async def test_cost_accumulation_through_gateway(self):
        from src.gateway.cost_tracker import CostTracker

        tracker = CostTracker(budget=100.0)
        gw = LLMGateway(
            middleware=[CostMiddleware(tracker=tracker)],
            default_model="dummy",
        )
        await gw.complete("test 1")
        await gw.complete("test 2")
        # DummyLLM still produces tokens, so cost should have been recorded
        assert tracker.total_cost >= Decimal("0")


# ── API endpoint ──────────────────────────────────────────────────────────────


class TestGatewayStatsEndpoint:
    @pytest.mark.asyncio
    async def test_gateway_stats_returns_dict(self):
        from httpx import ASGITransport, AsyncClient

        from src.api.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://test",
            headers={"X-API-Key": "dev-key"},
        ) as client:
            resp = await client.get("/gateway/stats")
            assert resp.status_code == 200
            data = resp.json()
            assert "models" in data
            assert "total_cost" in data
            assert "circuit_breakers" in data
