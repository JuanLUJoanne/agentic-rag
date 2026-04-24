"""Unit tests for the Supervisor orchestrator."""
from __future__ import annotations

import pytest

from src.agents.analysis_agent import AnalysisAgent
from src.agents.base import AgentRegistry
from src.agents.quality_agent import QualityAgent
from src.agents.research_agent import ResearchAgent
from src.agents.supervisor import Supervisor

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register(ResearchAgent())
    reg.register(AnalysisAgent())
    reg.register(QualityAgent())
    return reg


@pytest.fixture
def supervisor(registry: AgentRegistry) -> Supervisor:
    return Supervisor(registry=registry, max_iterations=5, budget=0.05)


def _state(**kwargs) -> dict:
    """Build a minimal supervisor state dict with sensible defaults."""
    defaults = {
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
    return {**defaults, **kwargs}


# ── DummyLLM dispatch sequence ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_picks_research_first(supervisor):
    """New query with no agents called → parallel dispatch research + analysis."""
    decision = await supervisor.decide(_state())
    assert decision["next_agents"] == ["research", "analysis"]


@pytest.mark.asyncio
async def test_supervisor_picks_analysis_after_research(supervisor):
    """Research done → dispatch analysis."""
    decision = await supervisor.decide(_state(
        agents_called=["research"],
        retrieved_docs=[{"id": "d1", "content": "...", "source": "s", "score": 0.9}],
    ))
    assert decision["next_agent"] == "analysis"


@pytest.mark.asyncio
async def test_supervisor_picks_quality_after_analysis(supervisor):
    """Research + analysis done → dispatch quality."""
    decision = await supervisor.decide(_state(
        agents_called=["research", "analysis"],
        generation="LangGraph is a framework for building stateful LLM apps.",
    ))
    assert decision["next_agent"] == "quality"


@pytest.mark.asyncio
async def test_supervisor_done_after_quality(supervisor):
    """All three agents done → declare done."""
    decision = await supervisor.decide(_state(
        agents_called=["research", "analysis", "quality"],
        answer_quality=0.95,
    ))
    assert decision["next_agent"] == "done"


@pytest.mark.asyncio
async def test_supervisor_done_includes_reasoning(supervisor):
    decision = await supervisor.decide(_state(
        agents_called=["research", "analysis", "quality"],
        answer_quality=0.95,
    ))
    assert "reasoning" in decision
    assert decision["reasoning"]


# ── Guard-clause tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_stops_at_max_iterations(supervisor):
    """iteration_count >= max_iterations must short-circuit to done."""
    decision = await supervisor.decide(_state(iteration_count=5))
    assert decision["next_agent"] == "done"
    assert "max_iterations" in decision["reasoning"]


@pytest.mark.asyncio
async def test_supervisor_stops_before_max_iterations(supervisor):
    """iteration_count < max_iterations should NOT trigger the guard."""
    decision = await supervisor.decide(_state(iteration_count=4))
    next_agent = decision.get("next_agent")
    # Just verify it doesn't erroneously fire the guard at iteration 4
    # (it may be "done" for other reasons, so we just check the guard reasoning)
    if next_agent == "done":
        assert "max_iterations" not in decision["reasoning"]


@pytest.mark.asyncio
async def test_supervisor_stops_on_budget_exceeded(supervisor):
    """cost_so_far >= budget must short-circuit to done."""
    decision = await supervisor.decide(_state(cost_so_far=0.10))
    assert decision["next_agent"] == "done"
    assert "budget" in decision["reasoning"]


@pytest.mark.asyncio
async def test_supervisor_budget_not_triggered_below_limit(supervisor):
    """cost_so_far < budget must not trigger budget guard."""
    decision = await supervisor.decide(_state(cost_so_far=0.01))
    # Should proceed with normal dispatch (parallel research+analysis in dummy mode)
    assert decision.get("next_agents") == ["research", "analysis"]


# ── Decision structure ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_decision_has_required_keys(supervisor):
    decision = await supervisor.decide(_state())
    # First call returns parallel dispatch (next_agents) instead of next_agent
    assert "next_agents" in decision or "next_agent" in decision
    assert "required_skill" in decision
    assert "reasoning" in decision


@pytest.mark.asyncio
async def test_supervisor_research_decision_has_skill(supervisor):
    decision = await supervisor.decide(_state())
    assert decision["required_skill"]  # non-empty skill for parallel dispatch
