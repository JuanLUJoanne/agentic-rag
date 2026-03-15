"""
Skills-based Supervisor — Multi-Agent pattern.

The supervisor reads agent capabilities from AgentRegistry at decision time,
so the prompt is always up-to-date without code changes when new agents are
added. This is the key design choice that makes the multi-agent system
open for extension: register a new agent → it's immediately available.

Three safety guards prevent infinite loops and runaway costs:
  - max_iterations: caps total agent dispatches per query
  - budget: stops execution when cost_so_far exceeds the per-query limit
  - CostGuardrail: pre-flight cost check before each dispatch
"""
from __future__ import annotations

import json

import structlog

from src.agents.base import AgentRegistry
from src.gateway.cost_tracker import get_default_tracker
from src.gateway.guardrails import CostGuardrail
from src.utils.llm import DummyLLM, get_llm

logger = structlog.get_logger()

# Shared guardrail; uses the module-level CostTracker singleton so costs
# recorded by individual agent steps are visible to the supervisor.
_guardrail = CostGuardrail(
    tracker=get_default_tracker(),
    per_request_limit=0.05,
    per_query_limit=0.10,
)

_SUPERVISOR_PROMPT = """\
You coordinate a team of specialist agents to answer user queries accurately.

{capabilities}

Current state:
- query: {query}
- documents retrieved: {doc_count}
- answer generated: {has_answer}
- quality score: {quality_score}

Decide which agent to dispatch next, or declare done if the answer is ready.
Respond with valid JSON only — no explanation outside the JSON block.

If dispatching an agent:
{{"next_agent": "<agent_name>", "required_skill": "<skill>", "reasoning": "<one sentence>"}}

If finished:
{{"next_agent": "done", "required_skill": "", "reasoning": "<one sentence>"}}"""


class Supervisor:
    """
    Orchestrates specialist agents using a dynamic capability prompt.

    DummyLLM path: deterministic research → analysis → quality → done
    sequence that exercises the full graph without an API key.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        max_iterations: int = 5,
        budget: float = 0.05,
    ) -> None:
        self.registry = registry
        self.max_iterations = max_iterations
        self.budget = budget

    async def decide(self, state: dict) -> dict:
        """
        Choose the next agent or declare the pipeline complete.

        Checks budget and iteration limits before consulting the LLM so
        runaway loops always terminate even if the model misbehaves.
        """
        iteration_count = state.get("iteration_count", 0)
        cost_so_far = state.get("cost_so_far", 0.0)

        if iteration_count >= self.max_iterations:
            decision = {
                "next_agent": "done",
                "required_skill": "",
                "reasoning": "max_iterations_reached",
            }
            logger.info("supervisor_decision", **decision, iteration_count=iteration_count)
            return decision

        if cost_so_far >= self.budget:
            decision = {
                "next_agent": "done",
                "required_skill": "",
                "reasoning": "budget_exceeded",
            }
            logger.info("supervisor_decision", **decision, cost_so_far=cost_so_far)
            return decision

        # CostGuardrail: pre-flight check before each agent dispatch
        query_id = state.get("query", "")[:64]
        guardrail_result = _guardrail.check(
            model_id="gpt-4o-mini",
            estimated_tokens=500,
            query_id=query_id,
        )
        if not guardrail_result.allowed:
            decision = {
                "next_agent": "done",
                "required_skill": "",
                "reasoning": f"guardrail_blocked: {guardrail_result.reason}",
            }
            logger.warning(
                "supervisor_guardrail_blocked",
                reason=guardrail_result.reason,
                iteration_count=iteration_count,
            )
            return decision

        llm = get_llm()
        if isinstance(llm, DummyLLM):
            decision = self._dummy_decide(state)
        else:
            decision = await self._llm_decide(state)

        logger.info(
            "supervisor_decision",
            next_agent=decision.get("next_agent"),
            skill=decision.get("required_skill", ""),
            reasoning=decision.get("reasoning", "")[:100],
        )
        return decision

    # ── Decision strategies ────────────────────────────────────────────────────

    def _dummy_decide(self, state: dict) -> dict:
        """
        Fixed dispatch sequence for DummyLLM environments.

        Follows research → analysis → quality → done regardless of state
        quality, because DummyLLM always produces valid-looking results.
        Checks agents_called so it never re-dispatches a completed agent.
        """
        agents_called: list[str] = state.get("agents_called", [])

        if "research" not in agents_called:
            return {
                "next_agent": "research",
                "required_skill": "document_retrieval",
                "reasoning": "Need to retrieve relevant documents before generating an answer",
            }
        if "analysis" not in agents_called:
            return {
                "next_agent": "analysis",
                "required_skill": "grounded_generation",
                "reasoning": "Documents retrieved; generate a grounded answer with citations",
            }
        if "quality" not in agents_called:
            return {
                "next_agent": "quality",
                "required_skill": "hallucination_check",
                "reasoning": "Answer generated; verify it is factually grounded",
            }
        return {
            "next_agent": "done",
            "required_skill": "",
            "reasoning": "All pipeline stages complete; answer is ready",
        }

    async def _llm_decide(self, state: dict) -> dict:
        """Query the LLM for a dispatch decision; fall back to dummy on parse error."""
        prompt = _SUPERVISOR_PROMPT.format(
            capabilities=self.registry.get_all_capabilities(),
            query=state.get("query", ""),
            doc_count=len(state.get("retrieved_docs", [])),
            has_answer=bool(state.get("generation", "")),
            quality_score=state.get("answer_quality", "not_checked"),
        )
        llm = get_llm()
        response = await llm.ainvoke(prompt)

        try:
            return json.loads(response.content.strip())
        except (json.JSONDecodeError, ValueError):
            # Parse failure → fall back to deterministic sequence
            logger.warning("supervisor_llm_parse_error", raw=response.content[:200])
            return self._dummy_decide(state)
