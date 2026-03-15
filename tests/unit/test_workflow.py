"""Unit tests for the simple Corrective RAG workflow."""
from __future__ import annotations

import uuid

import pytest

from src.graph.simple_workflow import (
    finalize,
    get_initial_state,
    graph,
    route_after_grading,
    route_after_hallucination,
)

# ── Routing function tests (pure, synchronous) ─────────────────────────────────


def test_route_after_grading_all_relevant():
    state = {"docs_relevant": "all_relevant", "should_rewrite_query": False}
    assert route_after_grading(state) == "generate"


def test_route_after_grading_partial_first_time():
    """First partial result should trigger a query rewrite."""
    state = {"docs_relevant": "partial", "should_rewrite_query": False}
    assert route_after_grading(state) == "rewrite_query"


def test_route_after_grading_partial_already_rewrote():
    """After a rewrite attempt, partial should go straight to generate."""
    state = {"docs_relevant": "partial", "should_rewrite_query": True}
    assert route_after_grading(state) == "generate"


def test_route_after_grading_none_first_time():
    """No relevant docs on first attempt → try web search."""
    state = {"docs_relevant": "none", "should_rewrite_query": False}
    assert route_after_grading(state) == "web_search"


def test_route_after_grading_none_already_tried():
    """After web search attempt, none should fall back to generate."""
    state = {"docs_relevant": "none", "should_rewrite_query": True}
    assert route_after_grading(state) == "generate"


def test_route_after_hallucination_grounded():
    state = {"is_hallucinated": False, "retry_count": 0, "max_retries": 2}
    assert route_after_hallucination(state) == "finalize"


def test_route_after_hallucination_retry_first():
    state = {"is_hallucinated": True, "retry_count": 1, "max_retries": 2}
    assert route_after_hallucination(state) == "generate"


def test_route_after_hallucination_max_retries_reached():
    """When retry_count >= max_retries, force finalize even if hallucinated."""
    state = {"is_hallucinated": True, "retry_count": 2, "max_retries": 2}
    assert route_after_hallucination(state) == "finalize"


def test_route_after_hallucination_exceeds_max():
    state = {"is_hallucinated": True, "retry_count": 5, "max_retries": 2}
    assert route_after_hallucination(state) == "finalize"


# ── finalize node tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_finalize_sets_final_answer():
    state = {"query": "test", "generation": "This is the answer.", "agent_trace": []}
    result = await finalize(state)
    assert result["final_answer"] == "This is the answer."


@pytest.mark.asyncio
async def test_finalize_fallback_when_no_generation():
    state = {"query": "What is RAG?", "generation": "", "agent_trace": []}
    result = await finalize(state)
    assert "What is RAG?" in result["final_answer"]
    assert result["final_answer"]  # Must be non-empty


# ── End-to-end workflow tests (DummyLLM) ──────────────────────────────────────


@pytest.mark.asyncio
async def test_full_workflow_happy_path():
    """Full pipeline with DummyLLM must complete and return a non-empty answer."""
    initial_state = get_initial_state("What is LangGraph?")
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state["final_answer"] is not None
    assert len(final_state["final_answer"]) > 0


@pytest.mark.asyncio
async def test_full_workflow_complex_query():
    """Complex query should route through query_analyzer with sub-query decomposition."""
    initial_state = get_initial_state("Compare BM25 and dense retrieval methods")
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state["final_answer"] is not None
    assert final_state["query_type"] == "complex"


@pytest.mark.asyncio
async def test_full_workflow_agent_trace_populated():
    """Every node should append to agent_trace."""
    initial_state = get_initial_state("What is RAG?")
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    final_state = await graph.ainvoke(initial_state, config=config)

    trace_nodes = {step["node"] for step in final_state.get("agent_trace", [])}
    # These nodes must always run in the happy path
    assert "query_router" in trace_nodes
    assert "retrieve" in trace_nodes
    assert "grade_documents" in trace_nodes
    assert "generate" in trace_nodes
    assert "finalize" in trace_nodes


@pytest.mark.asyncio
async def test_full_workflow_docs_relevant_set():
    initial_state = get_initial_state("What is LangGraph?")
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state["docs_relevant"] in ("all_relevant", "partial", "none")


@pytest.mark.asyncio
async def test_full_workflow_hallucination_false_with_dummy():
    """DummyLLM always returns is_hallucinated=False so no retries should occur."""
    initial_state = get_initial_state("What is corrective RAG?")
    config = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}

    final_state = await graph.ainvoke(initial_state, config=config)

    assert final_state["is_hallucinated"] is False
    assert final_state["retry_count"] == 0
