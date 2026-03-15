"""
Analysis Agent — Prompt Chaining pattern (specialist implementation).

Wraps the three-step grounded generator. The supervisor dispatches here
when grounded_generation is needed and documents are already in state.
"""
from __future__ import annotations

import structlog

from src.agents.base import BaseAgent
from src.agents.generator import generate
from src.graph.state import AgentState

logger = structlog.get_logger()


class AnalysisAgent(BaseAgent):
    name = "analysis"
    skills = ["summarization", "multi_hop_reasoning", "grounded_generation", "comparison"]
    description = "Generates grounded answers with citations from retrieved documents"

    async def execute(self, state: AgentState) -> dict:
        """
        Run the three-step grounded generation pipeline.

        Delegates entirely to generator.generate so the prompt-chain logic
        lives in one place — analysis_agent is purely a skills wrapper.
        """
        result = await generate(state)

        citation_count = len(result.get("citations", []))
        logger.info("analysis_agent_complete", citation_count=citation_count)

        return {
            "generation": result["generation"],
            "citations": result["citations"],
            "retry_count": state.get("retry_count", 0),
            "agent_trace": [{"node": "analysis_agent", "citation_count": citation_count}],
        }
