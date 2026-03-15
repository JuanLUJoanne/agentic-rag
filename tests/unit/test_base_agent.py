"""Unit tests for BaseAgent and AgentRegistry."""
from __future__ import annotations

import pytest

from src.agents.analysis_agent import AnalysisAgent
from src.agents.base import AgentRegistry, BaseAgent
from src.agents.quality_agent import QualityAgent
from src.agents.research_agent import ResearchAgent

# ── Concrete test agent ────────────────────────────────────────────────────────


class ConcreteAgent(BaseAgent):
    name = "test_agent"
    skills = ["skill_a", "skill_b"]
    description = "A test agent for unit testing"

    async def execute(self, state):
        return {"agent_trace": [{"node": "test_agent"}]}


# ── Registration tests ─────────────────────────────────────────────────────────


def test_agent_registers_in_registry():
    registry = AgentRegistry()
    registry.register(ConcreteAgent())
    assert len(registry.list_agents()) == 1


def test_registered_agent_name_preserved():
    registry = AgentRegistry()
    registry.register(ConcreteAgent())
    assert registry.list_agents()[0].name == "test_agent"


def test_multiple_agents_register_independently():
    registry = AgentRegistry()
    registry.register(ResearchAgent())
    registry.register(AnalysisAgent())
    registry.register(QualityAgent())
    assert len(registry.list_agents()) == 3


# ── find_by_skill tests ────────────────────────────────────────────────────────


def test_find_by_skill_returns_matching_agents():
    registry = AgentRegistry()
    registry.register(ConcreteAgent())
    result = registry.find_by_skill("skill_a")
    assert len(result) == 1
    assert result[0].name == "test_agent"


def test_find_by_skill_returns_empty_for_unknown():
    registry = AgentRegistry()
    registry.register(ConcreteAgent())
    assert registry.find_by_skill("nonexistent_skill") == []


def test_find_by_skill_returns_multiple_matches():
    registry = AgentRegistry()

    class AgentA(BaseAgent):
        name = "agent_a"
        skills = ["shared_skill", "unique_a"]
        description = "Agent A"
        async def execute(self, state): return {}

    class AgentB(BaseAgent):
        name = "agent_b"
        skills = ["shared_skill", "unique_b"]
        description = "Agent B"
        async def execute(self, state): return {}

    registry.register(AgentA())
    registry.register(AgentB())

    result = registry.find_by_skill("shared_skill")
    names = {a.name for a in result}
    assert names == {"agent_a", "agent_b"}


def test_find_by_skill_partial_match_only():
    """find_by_skill should not match substrings — 'retrieval' ≠ 'document_retrieval'."""
    registry = AgentRegistry()
    registry.register(ResearchAgent())
    assert registry.find_by_skill("retrieval") == []
    assert len(registry.find_by_skill("document_retrieval")) == 1


# ── get_all_capabilities tests ─────────────────────────────────────────────────


def test_get_all_capabilities_contains_agent_name():
    registry = AgentRegistry()
    registry.register(ConcreteAgent())
    caps = registry.get_all_capabilities()
    assert "test_agent" in caps


def test_get_all_capabilities_contains_skills():
    registry = AgentRegistry()
    registry.register(ConcreteAgent())
    caps = registry.get_all_capabilities()
    assert "skill_a" in caps
    assert "skill_b" in caps


def test_get_all_capabilities_contains_description():
    registry = AgentRegistry()
    registry.register(ConcreteAgent())
    caps = registry.get_all_capabilities()
    assert "A test agent" in caps


def test_get_all_capabilities_empty_registry():
    registry = AgentRegistry()
    caps = registry.get_all_capabilities()
    assert "none" in caps.lower() or "Available agents" in caps


def test_new_agent_auto_discovered_in_capabilities():
    """
    Adding a new agent updates get_all_capabilities() with no supervisor changes.

    This is the key property of the skills-based design: open for extension.
    """
    registry = AgentRegistry()
    registry.register(ConcreteAgent())

    class BrandNewAgent(BaseAgent):
        name = "brand_new"
        skills = ["cutting_edge_skill"]
        description = "A brand new capability"
        async def execute(self, state): return {}

    registry.register(BrandNewAgent())

    caps = registry.get_all_capabilities()
    assert "brand_new" in caps
    assert "cutting_edge_skill" in caps
    assert len(registry.list_agents()) == 2


# ── Real agent skills tests ────────────────────────────────────────────────────


def test_research_agent_skills():
    agent = ResearchAgent()
    assert "document_retrieval" in agent.skills
    assert "web_search" in agent.skills


def test_analysis_agent_skills():
    agent = AnalysisAgent()
    assert "grounded_generation" in agent.skills


def test_quality_agent_skills():
    agent = QualityAgent()
    assert "hallucination_check" in agent.skills
    assert "eval_scoring" in agent.skills


@pytest.mark.asyncio
async def test_research_agent_execute_returns_docs(base_state):
    agent = ResearchAgent()
    result = await agent.execute(base_state)
    assert "retrieved_docs" in result
    assert "docs_relevant" in result


@pytest.mark.asyncio
async def test_analysis_agent_execute_returns_generation(base_state):
    agent = AnalysisAgent()
    result = await agent.execute(base_state)
    assert "generation" in result
    assert result["generation"]


@pytest.mark.asyncio
async def test_quality_agent_execute_returns_score(base_state):
    agent = QualityAgent()
    result = await agent.execute(base_state)
    assert "answer_quality" in result
    assert 0.0 <= result["answer_quality"] <= 1.0
