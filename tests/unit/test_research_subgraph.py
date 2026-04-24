"""
Tests for the research subgraph (retrieve → grade → rewrite loop → synthesize).

Covers:
  - Subgraph compilation and basic execution
  - State mapping (to_research_state / from_research_state)
  - Rewrite loop respects max_rewrites cap
  - Routing logic (_route_after_grade)
  - Integration with the multi-agent workflow (research_agent_node uses subgraph)
"""
from __future__ import annotations

import uuid

import pytest

from src.graph.research_subgraph import (
    ResearchState,
    _route_after_grade,
    build_research_subgraph,
    from_research_state,
    research_compiled,
    to_research_state,
)


def _minimal_research_state(**overrides) -> ResearchState:
    """Build a minimal ResearchState for testing."""
    base = {
        "query": "What is BM25?",
        "chat_history": [],
        "query_type": "simple",
        "sub_queries": [],
        "retrieved_docs": [],
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
        "rewrite_count": 0,
        "max_rewrites": 2,
    }
    return ResearchState(**{**base, **overrides})


# ── Routing tests ─────────────────────────────────────────────────────────────


def test_route_after_grade_all_relevant():
    state = _minimal_research_state(docs_relevant="all_relevant", rewrite_count=0)
    assert _route_after_grade(state) == "synthesize"


def test_route_after_grade_partial_first_attempt():
    state = _minimal_research_state(docs_relevant="partial", rewrite_count=0)
    assert _route_after_grade(state) == "rewrite"


def test_route_after_grade_none_first_attempt():
    state = _minimal_research_state(docs_relevant="none", rewrite_count=0)
    assert _route_after_grade(state) == "rewrite"


def test_route_after_grade_partial_exhausted_rewrites():
    state = _minimal_research_state(docs_relevant="partial", rewrite_count=2, max_rewrites=2)
    assert _route_after_grade(state) == "synthesize"


def test_route_after_grade_none_exhausted_rewrites():
    state = _minimal_research_state(docs_relevant="none", rewrite_count=2, max_rewrites=2)
    assert _route_after_grade(state) == "synthesize"


def test_route_after_grade_rewrite_under_cap():
    state = _minimal_research_state(docs_relevant="partial", rewrite_count=1, max_rewrites=2)
    assert _route_after_grade(state) == "rewrite"


# ── State mapping tests ───────────────────────────────────────────────────────


def test_to_research_state_preserves_query():
    outer = {"query": "What is BERT?", "retrieved_docs": [{"id": "d1"}]}
    rs = to_research_state(outer)
    assert rs["query"] == "What is BERT?"
    assert rs["rewrite_count"] == 0
    assert rs["max_rewrites"] == 2


def test_to_research_state_defaults():
    outer = {"query": "test"}
    rs = to_research_state(outer)
    assert rs["retrieved_docs"] == []
    assert rs["agent_trace"] == []
    assert rs["rewrite_count"] == 0


def test_from_research_state_maps_required_fields():
    research_result = {
        "retrieved_docs": [{"id": "d1", "content": "doc text"}],
        "relevance_scores": [0.9],
        "docs_relevant": "all_relevant",
        "retrieval_strategy": "parallel_simple",
        "agent_trace": [{"node": "research_retrieve"}],
    }
    outer = from_research_state(research_result)
    assert outer["retrieved_docs"] == research_result["retrieved_docs"]
    assert outer["relevance_scores"] == [0.9]
    assert outer["docs_relevant"] == "all_relevant"
    assert outer["agents_called"] == ["research"]


def test_from_research_state_includes_subgraph_marker():
    result = from_research_state({"agent_trace": []})
    trace_nodes = [t.get("node") for t in result["agent_trace"]]
    assert "research_agent" in trace_nodes


# ── Subgraph compilation tests ────────────────────────────────────────────────


def test_subgraph_compiles():
    """build_research_subgraph produces a valid StateGraph."""
    sg = build_research_subgraph()
    compiled = sg.compile()
    assert compiled is not None


def test_research_compiled_is_available():
    """Module-level research_compiled should be importable and not None."""
    assert research_compiled is not None


# ── End-to-end subgraph execution ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subgraph_end_to_end_produces_docs():
    """Running the subgraph should retrieve and grade documents."""
    state = _minimal_research_state(query="What is BM25 retrieval?")
    result = await research_compiled.ainvoke(state)
    # DummyLLM grades all docs as relevant → should have retrieved_docs
    assert result.get("retrieved_docs") is not None


@pytest.mark.asyncio
async def test_subgraph_end_to_end_sets_docs_relevant():
    state = _minimal_research_state(query="What is BM25 retrieval?")
    result = await research_compiled.ainvoke(state)
    assert result.get("docs_relevant") in ("all_relevant", "partial", "none")


@pytest.mark.asyncio
async def test_subgraph_end_to_end_has_trace():
    state = _minimal_research_state(query="What is BM25?")
    result = await research_compiled.ainvoke(state)
    trace = result.get("agent_trace", [])
    trace_nodes = [t.get("node") for t in trace]
    assert "retrieve" in trace_nodes or any("retrieve" in str(t) for t in trace)


@pytest.mark.asyncio
async def test_subgraph_rewrite_count_stays_at_zero_when_all_relevant():
    """When docs are all_relevant, no rewrite should happen."""
    state = _minimal_research_state(query="What is BM25 retrieval?")
    result = await research_compiled.ainvoke(state)
    # DummyLLM marks all docs as relevant → no rewrite needed
    assert result.get("rewrite_count", 0) == 0


# ── Integration with multi_agent_workflow ─────────────────────────────────────


@pytest.mark.asyncio
async def test_research_agent_node_uses_subgraph():
    """research_agent_node in multi_agent_workflow delegates to the subgraph."""
    from src.graph.multi_agent_workflow import (
        SupervisorState,
        research_agent_node,
    )

    state = SupervisorState(
        query="What is dense retrieval?",
        chat_history=[],
        query_type="simple",
        sub_queries=[],
        retrieved_docs=[],
        graph_context=[],
        relevance_scores=[],
        retrieval_strategy="dense",
        generation="",
        citations=[],
        docs_relevant=None,
        is_hallucinated=None,
        answer_quality=None,
        retry_count=0,
        max_retries=2,
        should_rewrite_query=False,
        final_answer=None,
        query_embedding=None,
        cost_so_far=0.0,
        agent_trace=[],
        supervisor_decision=None,
        iteration_count=0,
        agents_called=[],
        mode="multi_agent",
    )

    result = await research_agent_node(state)
    assert "research" in result.get("agents_called", [])
    assert result.get("retrieved_docs") is not None
    assert result.get("iteration_count") == 1
