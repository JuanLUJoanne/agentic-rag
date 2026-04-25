"""Unit tests for OpenTelemetry tracing setup and span instrumentation."""
from __future__ import annotations

from collections.abc import Sequence

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

# ---------------------------------------------------------------------------
# In-memory exporter (opentelemetry.sdk.trace.export.in_memory may not be
# installed, so we define a minimal one here)
# ---------------------------------------------------------------------------


class _InMemoryExporter(SpanExporter):
    """Collects finished spans for test assertions."""

    def __init__(self) -> None:
        self.spans: list = []

    def export(self, spans: Sequence) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.spans.clear()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_tracer():
    """Reset the module-level tracer cache before/after each test."""
    import src.observability.tracing as mod

    original = mod._tracer
    mod._tracer = None
    yield
    mod._tracer = original
    # Reset global tracer provider so subsequent tests can install their own
    otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]


@pytest.fixture
def exporter():
    """Create an InMemoryExporter and install a TracerProvider with it."""
    exp = _InMemoryExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exp))

    # Reset global state so set_tracer_provider works
    otel_trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    otel_trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    otel_trace.set_tracer_provider(provider)

    # Point the module-level tracer at this provider
    import src.observability.tracing as mod

    mod._tracer = provider.get_tracer("agentic-rag")

    yield exp
    exp.clear()


# ---------------------------------------------------------------------------
# setup_tracing tests
# ---------------------------------------------------------------------------


def test_setup_tracing_creates_provider():
    """setup_tracing should return a TracerProvider."""
    from src.observability.tracing import setup_tracing

    provider = setup_tracing("test-service")
    assert isinstance(provider, TracerProvider)


def test_setup_tracing_respects_service_name_env(monkeypatch):
    """OTEL_SERVICE_NAME env var should override the argument."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")
    from src.observability.tracing import setup_tracing

    provider = setup_tracing("arg-service")
    assert isinstance(provider, TracerProvider)
    # Resource should contain the env-specified name
    resource = provider.resource
    assert resource.attributes.get("service.name") == "env-service"


def test_setup_tracing_dev_mode_uses_simple_processor(monkeypatch):
    """OTEL_DEV_MODE=true should use SimpleSpanProcessor."""
    monkeypatch.setenv("OTEL_DEV_MODE", "true")
    from src.observability.tracing import setup_tracing

    provider = setup_tracing("test")
    assert isinstance(provider, TracerProvider)
    # Verify at least one SimpleSpanProcessor was added
    processors = provider._active_span_processor._span_processors
    assert any(isinstance(p, SimpleSpanProcessor) for p in processors)


# ---------------------------------------------------------------------------
# get_tracer / NoOp fallback tests
# ---------------------------------------------------------------------------


def test_get_tracer_returns_tracer():
    """get_tracer should return an object with start_as_current_span."""
    from src.observability.tracing import get_tracer

    tracer = get_tracer()
    assert hasattr(tracer, "start_as_current_span")


def test_noop_span_context_manager():
    """_NoOpSpan should work as a context manager and accept all methods."""
    from src.observability.tracing import _NoOpSpan

    span = _NoOpSpan()
    with span as s:
        s.set_attribute("key", "value")
        s.record_exception(RuntimeError("test"))
        s.set_status("OK")


def test_noop_tracer_yields_noop_span():
    """_NoOpTracer.start_as_current_span should yield a _NoOpSpan."""
    from src.observability.tracing import _NoOpSpan, _NoOpTracer

    tracer = _NoOpTracer()
    with tracer.start_as_current_span("test") as span:
        assert isinstance(span, _NoOpSpan)


# ---------------------------------------------------------------------------
# Context propagation helpers
# ---------------------------------------------------------------------------


def test_context_helpers_with_otel():
    """get_current_context / attach / detach should work without error."""
    from src.observability.tracing import (
        attach_context,
        detach_context,
        get_current_context,
    )

    ctx = get_current_context()
    assert ctx is not None
    token = attach_context(ctx)
    detach_context(token)


def test_context_helpers_none_safe():
    """attach/detach should handle None gracefully."""
    from src.observability.tracing import attach_context, detach_context

    assert attach_context(None) is None
    detach_context(None)  # should not raise


# ---------------------------------------------------------------------------
# Span status helpers
# ---------------------------------------------------------------------------


def test_set_span_ok(exporter):
    """set_span_ok should set the span status to OK."""
    from src.observability.tracing import get_tracer, set_span_ok

    with get_tracer().start_as_current_span("test_ok") as span:
        set_span_ok(span)

    assert len(exporter.spans) == 1
    assert exporter.spans[0].name == "test_ok"


def test_set_span_error(exporter):
    """set_span_error should record the exception on the span."""
    from src.observability.tracing import get_tracer, set_span_error

    with get_tracer().start_as_current_span("test_err") as span:
        set_span_error(span, RuntimeError("boom"))

    assert len(exporter.spans) == 1
    assert exporter.spans[0].name == "test_err"
    # Should have recorded events (the exception)
    assert len(exporter.spans[0].events) > 0


# ---------------------------------------------------------------------------
# Supervisor node span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supervisor_node_creates_span(exporter):
    """supervisor_node should create a span with decision attributes."""
    from src.graph.multi_agent_workflow import supervisor_node

    state = {
        "query": "What is LangGraph?",
        "agents_called": [],
        "iteration_count": 0,
        "cost_so_far": 0.0,
        "retrieved_docs": [],
        "generation": "",
        "answer_quality": None,
        "supervisor_decision": None,
        "mode": "multi_agent",
        "agent_trace": [],
    }
    result = await supervisor_node(state)
    assert "supervisor_decision" in result

    spans = [s for s in exporter.spans if s.name == "supervisor_node"]
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["rag.query"] == "What is LangGraph?"
    assert attrs["agent.iteration_count"] == 0
    assert "agent.decision" in attrs


# ---------------------------------------------------------------------------
# Research subgraph node spans
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_research_retrieve_node_span(exporter):
    """_retrieve_node should create a span with doc_count."""
    from src.graph.research_subgraph import _retrieve_node

    state = {
        "query": "test query",
        "retrieved_docs": [],
        "rewrite_count": 0,
        "max_rewrites": 2,
        "chat_history": [],
        "query_type": None,
        "sub_queries": [],
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "dense",
        "generation": "",
        "citations": [],
        "docs_relevant": None,
        "is_hallucinated": None,
        "answer_quality": None,
        "retry_count": 0,
        "max_retries": 2,
        "should_rewrite_query": False,
        "final_answer": None,
        "query_embedding": None,
        "cost_so_far": 0.0,
        "agent_trace": [],
    }
    await _retrieve_node(state)

    spans = [s for s in exporter.spans if s.name == "research_retrieve"]
    assert len(spans) == 1
    assert "retrieval.doc_count" in spans[0].attributes


@pytest.mark.asyncio
async def test_research_grade_node_span(exporter):
    """_grade_node should create a span with docs_relevant."""
    from src.graph.research_subgraph import _grade_node

    state = {
        "query": "test",
        "retrieved_docs": [
            {"id": "1", "content": "LangGraph is...", "source": "s", "score": 0.9}
        ],
        "rewrite_count": 0,
        "max_rewrites": 2,
        "chat_history": [],
        "query_type": None,
        "sub_queries": [],
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "dense",
        "generation": "",
        "citations": [],
        "docs_relevant": None,
        "is_hallucinated": None,
        "answer_quality": None,
        "retry_count": 0,
        "max_retries": 2,
        "should_rewrite_query": False,
        "final_answer": None,
        "query_embedding": None,
        "cost_so_far": 0.0,
        "agent_trace": [],
    }
    await _grade_node(state)

    spans = [s for s in exporter.spans if s.name == "research_grade"]
    assert len(spans) == 1
    assert "grade.docs_relevant" in spans[0].attributes


@pytest.mark.asyncio
async def test_research_rewrite_node_span(exporter):
    """_rewrite_node should create a span with rewrite iteration."""
    from src.graph.research_subgraph import _rewrite_node

    state = {
        "query": "original query",
        "rewrite_count": 0,
        "max_rewrites": 2,
        "retrieved_docs": [],
        "chat_history": [],
        "query_type": None,
        "sub_queries": [],
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "dense",
        "generation": "",
        "citations": [],
        "docs_relevant": None,
        "is_hallucinated": None,
        "answer_quality": None,
        "retry_count": 0,
        "max_retries": 2,
        "should_rewrite_query": False,
        "final_answer": None,
        "query_embedding": None,
        "cost_so_far": 0.0,
        "agent_trace": [],
    }
    await _rewrite_node(state)

    spans = [s for s in exporter.spans if s.name == "research_rewrite"]
    assert len(spans) == 1
    assert spans[0].attributes["rewrite.iteration"] == 1


@pytest.mark.asyncio
async def test_research_synthesize_node_span(exporter):
    """_synthesize_node should create a span with output length."""
    from src.graph.research_subgraph import _synthesize_node

    state = {
        "query": "test",
        "retrieved_docs": [
            {"id": "1", "content": "Some content here", "source": "s", "score": 0.9}
        ],
        "rewrite_count": 0,
        "max_rewrites": 2,
        "chat_history": [],
        "query_type": None,
        "sub_queries": [],
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "dense",
        "generation": "",
        "citations": [],
        "docs_relevant": None,
        "is_hallucinated": None,
        "answer_quality": None,
        "retry_count": 0,
        "max_retries": 2,
        "should_rewrite_query": False,
        "final_answer": None,
        "query_embedding": None,
        "cost_so_far": 0.0,
        "agent_trace": [],
    }
    await _synthesize_node(state)

    spans = [s for s in exporter.spans if s.name == "research_synthesize"]
    assert len(spans) == 1
    assert spans[0].attributes["synthesis.output_length"] > 0


# ---------------------------------------------------------------------------
# LLM call span
# ---------------------------------------------------------------------------


def test_dummy_llm_creates_span(exporter):
    """DummyLLM._generate should create an llm_call span."""
    from src.utils.llm import DummyLLM

    llm = DummyLLM()
    from langchain_core.messages import HumanMessage

    llm.invoke([HumanMessage(content="hello")])

    spans = [s for s in exporter.spans if s.name == "llm_call"]
    assert len(spans) >= 1
    assert spans[0].attributes["llm.model"] == "dummy"


# ---------------------------------------------------------------------------
# Parallel dispatch context propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_dispatch_creates_parent_span(exporter):
    """parallel_dispatch_node should create a parent span."""
    from src.graph.multi_agent_workflow import parallel_dispatch_node

    state = {
        "query": "What is LangGraph?",
        "agents_called": [],
        "iteration_count": 0,
        "cost_so_far": 0.0,
        "retrieved_docs": [],
        "generation": "",
        "answer_quality": None,
        "supervisor_decision": {"next_agents": ["research", "analysis"]},
        "mode": "multi_agent",
        "agent_trace": [],
        "chat_history": [],
        "query_type": None,
        "sub_queries": [],
        "graph_context": [],
        "relevance_scores": [],
        "retrieval_strategy": "dense",
        "citations": [],
        "docs_relevant": None,
        "is_hallucinated": None,
        "retry_count": 0,
        "max_retries": 2,
        "should_rewrite_query": False,
        "final_answer": None,
        "query_embedding": None,
    }
    result = await parallel_dispatch_node(state)
    assert "research" in result.get("agents_called", [])
    assert "analysis" in result.get("agents_called", [])

    parent_spans = [s for s in exporter.spans if s.name == "parallel_dispatch_node"]
    assert len(parent_spans) == 1
    assert parent_spans[0].attributes["parallel.agents"] == "research,analysis"

    # Child spans should exist for both agents
    child_names = {s.name for s in exporter.spans}
    assert "parallel_research" in child_names
    assert "parallel_analysis" in child_names
