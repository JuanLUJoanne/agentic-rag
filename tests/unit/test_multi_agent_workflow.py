"""Unit tests for the multi-agent workflow and API mode routing."""
from __future__ import annotations

import uuid

import pytest

from src.graph.multi_agent_workflow import (
    get_initial_supervisor_state,
    route_supervisor,
)
from src.graph.multi_agent_workflow import (
    graph as multi_agent_graph,
)
from src.graph.simple_workflow import get_initial_state
from src.graph.simple_workflow import graph as simple_graph

# ── route_supervisor pure-function tests ───────────────────────────────────────


def test_route_supervisor_research():
    state = {"supervisor_decision": {"next_agent": "research"}, "answer_quality": None}
    assert route_supervisor(state) == "research_agent"


def test_route_supervisor_analysis():
    state = {"supervisor_decision": {"next_agent": "analysis"}, "answer_quality": None}
    assert route_supervisor(state) == "analysis_agent"


def test_route_supervisor_quality():
    state = {"supervisor_decision": {"next_agent": "quality"}, "answer_quality": None}
    assert route_supervisor(state) == "quality_agent"


def test_route_supervisor_done_high_quality():
    """done + quality >= 0.7 → skip human review, go straight to finalize."""
    state = {"supervisor_decision": {"next_agent": "done"}, "answer_quality": 0.9}
    assert route_supervisor(state) == "finalize"


def test_route_supervisor_done_low_quality():
    """done + quality < 0.7 → route to human_review."""
    state = {"supervisor_decision": {"next_agent": "done"}, "answer_quality": 0.5}
    assert route_supervisor(state) == "human_review"


def test_route_supervisor_done_no_quality():
    """done + no quality score → finalize (treat as acceptable)."""
    state = {"supervisor_decision": {"next_agent": "done"}, "answer_quality": None}
    assert route_supervisor(state) == "finalize"


def test_route_supervisor_unknown_agent():
    """Unrecognised next_agent falls back to finalize."""
    state = {"supervisor_decision": {"next_agent": "unknown_agent"}, "answer_quality": None}
    assert route_supervisor(state) == "finalize"


def test_route_supervisor_missing_decision():
    state = {"supervisor_decision": None, "answer_quality": None}
    assert route_supervisor(state) == "finalize"


# ── End-to-end workflow tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_agent_end_to_end():
    """Full multi-agent pipeline must complete and produce a non-empty answer."""
    state = get_initial_supervisor_state("What is LangGraph?")
    config = {"configurable": {"thread_id": f"test-ma-{uuid.uuid4()}"}}

    final = await multi_agent_graph.ainvoke(state, config=config)

    assert final["final_answer"]
    assert len(final["final_answer"]) > 0


@pytest.mark.asyncio
async def test_multi_agent_all_agents_called():
    """DummyLLM sequence must dispatch research → analysis → quality."""
    state = get_initial_supervisor_state("What is RAG?")
    config = {"configurable": {"thread_id": f"test-ma-{uuid.uuid4()}"}}

    final = await multi_agent_graph.ainvoke(state, config=config)

    agents = final.get("agents_called", [])
    assert "research" in agents
    assert "analysis" in agents
    assert "quality" in agents


@pytest.mark.asyncio
async def test_multi_agent_iteration_count_increments():
    """iteration_count should equal the number of agent dispatches."""
    state = get_initial_supervisor_state("What is corrective RAG?")
    config = {"configurable": {"thread_id": f"test-ma-{uuid.uuid4()}"}}

    final = await multi_agent_graph.ainvoke(state, config=config)

    # 3 agents dispatched → iteration_count == 3
    assert final.get("iteration_count") == 3


@pytest.mark.asyncio
async def test_multi_agent_trace_populated():
    """Supervisor and agent nodes must all append to agent_trace."""
    state = get_initial_supervisor_state("Explain multi-hop reasoning")
    config = {"configurable": {"thread_id": f"test-ma-{uuid.uuid4()}"}}

    final = await multi_agent_graph.ainvoke(state, config=config)

    trace_nodes = {step["node"] for step in final.get("agent_trace", [])}
    assert "supervisor" in trace_nodes
    assert "research_agent" in trace_nodes
    assert "analysis_agent" in trace_nodes
    assert "quality_agent" in trace_nodes
    assert "finalize" in trace_nodes


@pytest.mark.asyncio
async def test_multi_agent_quality_score_set():
    state = get_initial_supervisor_state("What is dense retrieval?")
    config = {"configurable": {"thread_id": f"test-ma-{uuid.uuid4()}"}}

    final = await multi_agent_graph.ainvoke(state, config=config)

    assert final.get("answer_quality") is not None
    assert 0.0 <= final["answer_quality"] <= 1.0


@pytest.mark.asyncio
async def test_simple_mode_still_works():
    """Batch 1 simple workflow must be unaffected by Batch 2 changes."""
    state = get_initial_state("What is LangGraph?")
    config = {"configurable": {"thread_id": f"test-s-{uuid.uuid4()}"}}

    final = await simple_graph.ainvoke(state, config=config)

    assert final["final_answer"]


# ── API mode-routing tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_api_simple_mode():
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
        headers={"X-API-Key": "dev-key"},
    ) as client:
        response = await client.post(
            "/query", json={"query": "What is RAG?", "mode": "simple"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "simple"
    assert data["answer"]


@pytest.mark.asyncio
async def test_api_multi_agent_mode():
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
        headers={"X-API-Key": "dev-key"},
    ) as client:
        response = await client.post(
            "/query", json={"query": "What is LangGraph?", "mode": "multi_agent"}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "multi_agent"
    assert data["answer"]
    assert data["iteration_count"] > 0


@pytest.mark.asyncio
async def test_api_default_mode_is_simple():
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
        headers={"X-API-Key": "dev-key"},
    ) as client:
        # No mode field — should default to simple
        response = await client.post("/query", json={"query": "What is RAG?"})

    assert response.status_code == 200
    assert response.json()["mode"] == "simple"


@pytest.mark.asyncio
async def test_api_health():
    from httpx import ASGITransport, AsyncClient

    from src.api.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
