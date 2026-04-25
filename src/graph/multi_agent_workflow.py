"""
Multi-agent Corrective RAG workflow — Skills-based Supervisor pattern.

Batch 3 adds memory_check as the first node — if QueryMemory has a
high-quality cached answer the entire agent pipeline is bypassed.

Batch 4 adds production hardening:
  - sanitize_input (new entry point): blocks prompt-injection attempts.
  - Rate limiter check before each specialist-agent LLM call.
  - Per-query cost tracking via get_default_tracker().
  - audit_log (new terminal node): appends a JSONL record after finalize.

Batch 5 adds observability:
  - supervisor_node instrumented with an OTel child span.

Batch 6 upgrades:
  - research_agent replaced with a LangGraph subgraph (retrieve → grade →
    rewrite loop → synthesize) for multi-step retrieval.
  - Parallel dispatch via asyncio.gather when supervisor returns
    ``next_agents`` list (e.g. research + analysis concurrently).
    Latency = max(agent_latencies) instead of sum(agent_latencies).

Graph topology:
  sanitize_input ──► memory_check ──► supervisor ──┬──► research_agent ──► supervisor
       │ injection         │ hit             ▲     ├──► analysis_agent ──► supervisor
       └──► finalize       └──► finalize     │     ├──► quality_agent  ──► supervisor
                                             │     └──► parallel_dispatch ──► supervisor
                                             │              (asyncio.gather: research + analysis)
                                             └───────── human_review ──► finalize ──► audit_log ──► END
"""
from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.analysis_agent import AnalysisAgent
from src.agents.base import AgentRegistry
from src.agents.quality_agent import QualityAgent
from src.agents.research_agent import ResearchAgent
from src.agents.supervisor import Supervisor
from src.gateway.cost_tracker import get_default_tracker
from src.gateway.rate_limiter import get_default_rate_limiter
from src.graph.research_subgraph import (
    from_research_state,
    research_compiled,
    to_research_state,
)
from src.graph.simple_workflow import (
    audit_log,
    finalize,
    get_initial_state,
    memory_check,
    route_after_sanitize,
    sanitize_input,
)
from src.graph.state import AgentState
from src.observability.tracing import (
    attach_context,
    detach_context,
    get_current_context,
    get_tracer,
    set_span_ok,
)

logger = structlog.get_logger()


# ── Extended state ─────────────────────────────────────────────────────────────


class SupervisorState(AgentState):
    """
    AgentState extended with multi-agent orchestration fields.

    Keeps the corrective-RAG state intact so the multi-agent graph can
    reuse every node from simple_workflow without modification.
    """

    supervisor_decision: dict | None
    iteration_count: int
    # operator.add accumulates agent names across the pipeline
    agents_called: Annotated[list[str], operator.add]
    mode: str


# ── Registry & supervisor (built once at import time) ──────────────────────────

_registry = AgentRegistry()
_registry.register(ResearchAgent())
_registry.register(AnalysisAgent())
_registry.register(QualityAgent())

_supervisor = Supervisor(registry=_registry, max_iterations=5, budget=0.05)


# ── Graph nodes ────────────────────────────────────────────────────────────────


async def supervisor_node(state: SupervisorState) -> dict:
    """LangGraph node: ask the supervisor which agent to dispatch next.

    Instrumented with an OTel child span named 'supervisor_node'.
    """
    with get_tracer().start_as_current_span("supervisor_node") as span:
        query = state.get("query", "")
        span.set_attribute("rag.query", query[:100])
        span.set_attribute("agent.iteration_count", state.get("iteration_count", 0))
        decision = await _supervisor.decide(state)
        span.set_attribute(
            "agent.decision",
            decision.get("next_agent") or ",".join(decision.get("next_agents", [])),
        )
        span.set_attribute("agent.reasoning", decision.get("reasoning", "")[:200])
        set_span_ok(span)
        return {
            "supervisor_decision": decision,
            "agent_trace": [{"node": "supervisor", "decision": decision}],
        }


async def research_agent_node(state: SupervisorState) -> dict:
    """LangGraph node: run the research subgraph (retrieve → grade → rewrite loop → synthesize)."""
    with get_tracer().start_as_current_span("research_agent_node") as span:
        span.set_attribute("rag.query", state.get("query", "")[:100])
        span.set_attribute("agent.name", "research")
        span.set_attribute("agent.iteration_count", state.get("iteration_count", 0))
        await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 500)

        research_state = to_research_state(state)
        result = await research_compiled.ainvoke(research_state)
        outer_result = from_research_state(result)

        span.set_attribute("retrieval.doc_count", len(outer_result.get("retrieved_docs", [])))
        _record_agent_cost(model_id="gpt-4o-mini", input_t=300, output_t=200, state=state)
        set_span_ok(span)
        return {
            **outer_result,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


async def analysis_agent_node(state: SupervisorState) -> dict:
    """LangGraph node: delegate to AnalysisAgent and record the dispatch."""
    with get_tracer().start_as_current_span("analysis_agent_node") as span:
        span.set_attribute("rag.query", state.get("query", "")[:100])
        span.set_attribute("agent.name", "analysis")
        span.set_attribute("agent.iteration_count", state.get("iteration_count", 0))
        await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 500)
        result = await AnalysisAgent().execute(state)
        _record_agent_cost(model_id="gpt-4o-mini", input_t=400, output_t=300, state=state)
        set_span_ok(span)
        return {
            **result,
            "agents_called": ["analysis"],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


async def quality_agent_node(state: SupervisorState) -> dict:
    """LangGraph node: delegate to QualityAgent and record the dispatch."""
    with get_tracer().start_as_current_span("quality_agent_node") as span:
        span.set_attribute("rag.query", state.get("query", "")[:100])
        span.set_attribute("agent.name", "quality")
        span.set_attribute("agent.iteration_count", state.get("iteration_count", 0))
        await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 200)
        result = await QualityAgent().execute(state)
        quality = result.get("answer_quality")
        if quality is not None:
            span.set_attribute("quality.score", float(quality))
        _record_agent_cost(model_id="gpt-4o-mini", input_t=200, output_t=100, state=state)
        set_span_ok(span)
        return {
            **result,
            "agents_called": ["quality"],
            "iteration_count": state.get("iteration_count", 0) + 1,
        }


async def parallel_dispatch_node(state: SupervisorState) -> dict:
    """LangGraph node: run multiple agents concurrently via asyncio.gather.

    When the supervisor returns ``next_agents: ["research", "analysis"]``,
    this node runs both agents in parallel.  Latency becomes
    max(research_latency, analysis_latency) instead of sum.

    Uses asyncio.gather rather than LangGraph's Send() API because Send
    requires all parallel-written state fields to have Annotated reducers —
    asyncio.gather lets us merge results in application code without
    restructuring the entire state schema.
    """
    with get_tracer().start_as_current_span("parallel_dispatch_node") as span:
        decision = state.get("supervisor_decision") or {}
        next_agents = decision.get("next_agents", [])
        span.set_attribute("rag.query", state.get("query", "")[:100])
        span.set_attribute("parallel.agents", ",".join(next_agents))

        agent_map = {
            "research": _run_research,
            "analysis": _run_analysis,
        }

        # Capture parent context so child tasks inherit the trace
        parent_ctx = get_current_context()

        tasks = []
        for agent_name in next_agents:
            runner = agent_map.get(agent_name)
            if runner:
                tasks.append(runner(state, parent_ctx=parent_ctx))

        if not tasks:
            logger.warning("parallel_dispatch_no_valid_agents", next_agents=next_agents)
            return {"agent_trace": [{"node": "parallel_dispatch", "agents": []}]}

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results: combine all agent outputs
        merged: dict = {
            "agents_called": [],
            "agent_trace": [{"node": "parallel_dispatch", "agents": list(next_agents)}],
        }
        succeeded = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_name = next_agents[i] if i < len(next_agents) else "unknown"
                logger.warning("parallel_agent_failed", agent=agent_name, error=str(result))
                continue
            succeeded.append(next_agents[i] if i < len(next_agents) else "unknown")
            for key, value in result.items():
                if key == "agents_called":
                    merged["agents_called"].extend(value)
                elif key == "agent_trace":
                    merged["agent_trace"].extend(value)
                elif key == "iteration_count":
                    pass  # handled below
                else:
                    merged[key] = value

        merged["iteration_count"] = state.get("iteration_count", 0) + 1
        span.set_attribute("parallel.succeeded", ",".join(succeeded))
        set_span_ok(span)

        logger.info(
            "parallel_dispatch_complete",
            agents_dispatched=next_agents,
            agents_succeeded=succeeded,
        )
        return merged


async def _run_research(
    state: SupervisorState, *, parent_ctx: object = None
) -> dict:
    """Run research subgraph (for parallel dispatch) with context propagation."""
    token = attach_context(parent_ctx)
    try:
        with get_tracer().start_as_current_span("parallel_research") as span:
            span.set_attribute("agent.name", "research")
            await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 500)
            research_state = to_research_state(state)
            result = await research_compiled.ainvoke(research_state)
            outer_result = from_research_state(result)
            span.set_attribute("retrieval.doc_count", len(outer_result.get("retrieved_docs", [])))
            _record_agent_cost(model_id="gpt-4o-mini", input_t=300, output_t=200, state=state)
            set_span_ok(span)
            return outer_result
    finally:
        detach_context(token)


async def _run_analysis(
    state: SupervisorState, *, parent_ctx: object = None
) -> dict:
    """Run analysis agent (for parallel dispatch) with context propagation."""
    token = attach_context(parent_ctx)
    try:
        with get_tracer().start_as_current_span("parallel_analysis") as span:
            span.set_attribute("agent.name", "analysis")
            await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 500)
            result = await AnalysisAgent().execute(state)
            _record_agent_cost(model_id="gpt-4o-mini", input_t=400, output_t=300, state=state)
            set_span_ok(span)
            return {
                **result,
                "agents_called": ["analysis"],
            }
    finally:
        detach_context(token)


def _record_agent_cost(
    model_id: str, input_t: int, output_t: int, state: SupervisorState
) -> None:
    """Record estimated cost for an agent dispatch; swallow BudgetExceededError."""
    from src.gateway.cost_tracker import BudgetExceededError

    try:
        query_id = state.get("query", "")[:64]
        get_default_tracker().record_usage(
            model_id, input_t, output_t, query_id=query_id
        )
    except BudgetExceededError:
        logger.warning("per_query_budget_exceeded", model=model_id)


async def human_review(state: SupervisorState) -> dict:
    """
    LangGraph node: route low-confidence answers to the human review queue.

    When quality_agent scores answer_quality < 0.7, the supervisor routes
    here.  We submit the answer for human review and finalize with a pending
    message so the caller gets an immediate (if incomplete) response while the
    reviewer works in the background.

    Human-in-the-Loop pattern (Batch 5): human reviewers access the queue
    via GET /review/pending and approve/reject via POST /review/{id}/approve.
    Approved answers are stored in QueryMemory for future queries.
    """
    from src.api.human_review import submit_for_review

    with get_tracer().start_as_current_span("human_review_node") as span:
        quality = state.get("answer_quality", 0.0)
        query = state.get("query", "")
        answer = state.get("generation", "") or state.get("final_answer", "")
        span.set_attribute("rag.query", query[:100])
        span.set_attribute("quality.score", float(quality) if quality else 0.0)

        review_id = submit_for_review(
            query=query,
            answer=answer,
            confidence=float(quality) if quality else 0.0,
            reason=f"quality_agent score {quality:.2f} below threshold 0.70",
        )

        span.set_attribute("review.id", review_id)
        set_span_ok(span)

        logger.info(
            "human_review_submitted",
            review_id=review_id,
            answer_quality=quality,
            query=query[:80],
        )
        return {
            "final_answer": f"Answer pending human review (ID: {review_id})",
            "agent_trace": [
                {
                    "node": "human_review",
                    "quality": quality,
                    "review_id": review_id,
                    "status": "submitted",
                }
            ],
        }


# ── Routing ────────────────────────────────────────────────────────────────────


def route_after_sanitize_ma(
    state: SupervisorState,
) -> Literal["memory_check", "finalize"]:
    """Route to finalize on injection, otherwise proceed to memory check."""
    return route_after_sanitize(state)  # type: ignore[return-value]


def route_after_memory_check_ma(
    state: SupervisorState,
) -> Literal["finalize", "supervisor"]:
    """Route to finalize on memory hit, otherwise continue to supervisor."""
    return "finalize" if state.get("final_answer") else "supervisor"


def route_supervisor(
    state: SupervisorState,
) -> Literal[
    "research_agent", "analysis_agent", "quality_agent",
    "parallel_dispatch", "human_review", "finalize",
]:
    """
    Translate the supervisor's decision into a graph edge.

    Supports two dispatch modes:
      - Single agent: ``{"next_agent": "research"}`` → edge to "research_agent"
      - Parallel: ``{"next_agents": ["research", "analysis"]}``
        → edge to "parallel_dispatch" (asyncio.gather internally)

    'done' routes to human_review when quality < 0.7 (low-confidence answers
    need human sign-off), otherwise straight to finalize.
    """
    decision = state.get("supervisor_decision") or {}

    # ── Parallel dispatch ───────────────────────────────────────────────────
    next_agents = decision.get("next_agents")
    if next_agents and isinstance(next_agents, list) and len(next_agents) > 1:
        return "parallel_dispatch"

    # ── Single dispatch ─────────────────────────────────────────────────────
    next_agent = decision.get("next_agent")

    # Handle next_agents with single element
    if not next_agent and next_agents and len(next_agents) == 1:
        next_agent = next_agents[0]

    if next_agent == "research":
        return "research_agent"
    if next_agent == "analysis":
        return "analysis_agent"
    if next_agent == "quality":
        return "quality_agent"

    # 'done' or any unrecognised value → check quality threshold
    quality = state.get("answer_quality")
    if quality is not None and quality < 0.7:
        return "human_review"
    return "finalize"


# ── Graph construction ─────────────────────────────────────────────────────────


def build_multi_agent_workflow() -> StateGraph:
    """Assemble and return the multi-agent StateGraph (uncompiled)."""
    workflow = StateGraph(SupervisorState)

    # Nodes
    workflow.add_node("sanitize_input", sanitize_input)
    workflow.add_node("memory_check", memory_check)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("research_agent", research_agent_node)
    workflow.add_node("analysis_agent", analysis_agent_node)
    workflow.add_node("quality_agent", quality_agent_node)
    workflow.add_node("parallel_dispatch", parallel_dispatch_node)
    workflow.add_node("human_review", human_review)
    workflow.add_node("finalize", finalize)
    workflow.add_node("audit_log", audit_log)

    # Entry via sanitizer → memory check
    workflow.set_entry_point("sanitize_input")
    workflow.add_conditional_edges(
        "sanitize_input",
        route_after_sanitize_ma,
        {"memory_check": "memory_check", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "memory_check",
        route_after_memory_check_ma,
        {"finalize": "finalize", "supervisor": "supervisor"},
    )

    # Supervisor branches to specialist agents, parallel dispatch, or terminates
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research_agent": "research_agent",
            "analysis_agent": "analysis_agent",
            "quality_agent": "quality_agent",
            "parallel_dispatch": "parallel_dispatch",
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )

    # Every dispatch path reports back to the supervisor for the next decision
    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("analysis_agent", "supervisor")
    workflow.add_edge("quality_agent", "supervisor")
    workflow.add_edge("parallel_dispatch", "supervisor")

    # Human review passes through to finalize (resume handled in Batch 5)
    workflow.add_edge("human_review", "finalize")
    # finalize → audit → END
    workflow.add_edge("finalize", "audit_log")
    workflow.add_edge("audit_log", END)

    return workflow


# ── Public API ─────────────────────────────────────────────────────────────────
_checkpointer = MemorySaver()
graph = build_multi_agent_workflow().compile(checkpointer=_checkpointer)


def get_initial_supervisor_state(
    query: str,
    mode: str = "multi_agent",
    max_retries: int = 2,
) -> SupervisorState:
    """Return a fully-populated initial SupervisorState for a new query."""
    base = get_initial_state(query, max_retries=max_retries)
    return {
        **base,
        "supervisor_decision": None,
        "iteration_count": 0,
        "agents_called": [],
        "mode": mode,
    }
