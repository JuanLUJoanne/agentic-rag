"""
BaseAgent abstract class and AgentRegistry.

Skills-based design: the supervisor reads capabilities from the registry
at runtime, so adding a new agent only requires register() — no supervisor
code changes. This is the open/closed principle applied to multi-agent routing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import structlog

from src.graph.state import AgentState

logger = structlog.get_logger()


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Each subclass declares its skills as a class variable. The supervisor
    reads these at decision time to match required capabilities to available
    agents without any hardcoded agent names in the supervisor prompt.
    """

    name: ClassVar[str] = ""
    skills: ClassVar[list[str]] = []
    description: ClassVar[str] = ""

    @abstractmethod
    async def execute(self, state: AgentState) -> dict:
        """
        Run this agent's task against the current state.

        Returns a partial state dict — only the keys this agent produces.
        LangGraph merges this into the full state via reducers.
        """
        ...


class AgentRegistry:
    """
    Runtime registry of available agents and their capabilities.

    The supervisor calls get_all_capabilities() to build its prompt
    dynamically — new agents are auto-discovered with no other changes needed.
    """

    def __init__(self) -> None:
        self._agents: list[BaseAgent] = []

    def register(self, agent: BaseAgent) -> None:
        """Add an agent to the registry."""
        self._agents.append(agent)
        logger.debug("agent_registered", name=agent.name, skills=agent.skills)

    def find_by_skill(self, skill: str) -> list[BaseAgent]:
        """Return all agents that have the requested skill."""
        return [a for a in self._agents if skill in a.skills]

    def get_all_capabilities(self) -> str:
        """
        Format agent capabilities for injection into the supervisor prompt.

        Layout kept human-readable so the LLM can reliably parse it:
          Available agents:
          - research: skill_a, skill_b
            Description text
        """
        if not self._agents:
            return "Available agents:\n(none registered)"
        lines = ["Available agents:"]
        for agent in self._agents:
            skills_str = ", ".join(agent.skills)
            lines.append(f"- {agent.name}: {skills_str}")
            lines.append(f"  {agent.description}")
        return "\n".join(lines)

    def list_agents(self) -> list[BaseAgent]:
        """Return all registered agents."""
        return list(self._agents)
