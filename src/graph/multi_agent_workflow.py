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

Graph topology:
  sanitize_input ──► memory_check ──► supervisor ──► research_agent ──┐
       │ injection         │ hit             ▲        analysis_agent ──┤
       └──► finalize       └──► finalize     └────────quality_agent  ──┘
                                                      human_review   ──► finalize ──► audit_log ──► END
"""
from __future__ import annotations

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
from src.graph.simple_workflow import (
    audit_log,
    finalize,
    get_initial_state,
    memory_check,
    route_after_sanitize,
    sanitize_input,
)
from src.graph.state import AgentState
from src.observability.tracing import get_tracer

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
    with get_tracer().start_as_current_span("supervisor_node"):
        decision = await _supervisor.decide(state)
        return {
            "supervisor_decision": decision,
            "agent_trace": [{"node": "supervisor", "decision": decision}],
        }


async def research_agent_node(state: SupervisorState) -> dict:
    """LangGraph node: delegate to ResearchAgent and record the dispatch."""
    await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 500)
    result = await ResearchAgent().execute(state)
    _record_agent_cost(model_id="gpt-4o-mini", input_t=300, output_t=200, state=state)
    return {
        **result,
        "agents_called": ["research"],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def analysis_agent_node(state: SupervisorState) -> dict:
    """LangGraph node: delegate to AnalysisAgent and record the dispatch."""
    await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 500)
    result = await AnalysisAgent().execute(state)
    _record_agent_cost(model_id="gpt-4o-mini", input_t=400, output_t=300, state=state)
    return {
        **result,
        "agents_called": ["analysis"],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def quality_agent_node(state: SupervisorState) -> dict:
    """LangGraph node: delegate to QualityAgent and record the dispatch."""
    await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 200)
    result = await QualityAgent().execute(state)
    _record_agent_cost(model_id="gpt-4o-mini", input_t=200, output_t=100, state=state)
    return {
        **result,
        "agents_called": ["quality"],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


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

    quality = state.get("answer_quality", 0.0)
    query = state.get("query", "")
    answer = state.get("generation", "") or state.get("final_answer", "")

    review_id = submit_for_review(
        query=query,
        answer=answer,
        confidence=float(quality) if quality else 0.0,
        reason=f"quality_agent score {quality:.2f} below threshold 0.70",
    )

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
) -> Literal["research_agent", "analysis_agent", "quality_agent", "human_review", "finalize"]:
    """
    Translate the supervisor's next_agent decision into a graph edge.

    'done' routes to human_review when quality < 0.7 (low-confidence answers
    need human sign-off), otherwise straight to finalize.
    """
    decision = state.get("supervisor_decision") or {}
    next_agent = decision.get("next_agent", "done")

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

    # Supervisor branches to any specialist agent or terminates
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research_agent": "research_agent",
            "analysis_agent": "analysis_agent",
            "quality_agent": "quality_agent",
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )

    # Every specialist agent reports back to the supervisor for the next decision
    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("analysis_agent", "supervisor")
    workflow.add_edge("quality_agent", "supervisor")

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
