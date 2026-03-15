"""
Quality Agent — Reflection pattern (specialist implementation).

Combines hallucination checking with a simple eval score so the supervisor
gets a single quality signal back rather than raw boolean + separate metric.
"""
from __future__ import annotations

import structlog

from src.agents.base import BaseAgent
from src.agents.hallucination_checker import check_hallucination
from src.graph.state import AgentState

logger = structlog.get_logger()


class QualityAgent(BaseAgent):
    name = "quality"
    skills = ["hallucination_check", "citation_verify", "factual_grounding", "eval_scoring"]
    description = "Verifies answer quality and checks for hallucinations against source docs"

    async def execute(self, state: AgentState) -> dict:
        """
        Check hallucination then compute a scalar quality score.

        Score formula: 0.7 weight on groundedness + 0.3 weight on citation
        coverage (capped at the first 3 source docs).  A hallucinated answer
        immediately scores 0.0 regardless of citation count.
        """
        hc_result = await check_hallucination(state)
        is_hallucinated = hc_result.get("is_hallucinated", False)

        # Base score: 1.0 if grounded, 0.0 if hallucinated
        base_score = 0.0 if is_hallucinated else 1.0

        # Adjust for citation coverage — rewarding answers that cite sources
        citations = state.get("citations", [])
        docs = state.get("retrieved_docs", [])
        if docs and citations and not is_hallucinated:
            coverage = len(citations) / min(len(docs), 3)
            answer_quality = round(base_score * 0.7 + min(coverage, 1.0) * 0.3, 3)
        else:
            answer_quality = base_score

        passed = not is_hallucinated
        logger.info(
            "quality_agent_complete",
            passed=passed,
            answer_quality=answer_quality,
            is_hallucinated=is_hallucinated,
        )

        return {
            "is_hallucinated": is_hallucinated,
            "answer_quality": answer_quality,
            "retry_count": hc_result.get("retry_count", state.get("retry_count", 0)),
            "agent_trace": [
                {"node": "quality_agent", "passed": passed, "score": answer_quality}
            ],
        }
