"""
Single-graph Corrective RAG workflow.

Batch 3 adds memory_check as the first node — if QueryMemory has a
high-quality cached answer the full pipeline is skipped entirely.

Batch 4 adds production hardening:
  - sanitize_input (new entry point): blocks prompt-injection attempts.
  - gate_generate: awaits rate-limiter capacity before calling the LLM.
  - audit_log (new terminal node): appends a JSONL audit record after finalize.

Batch 5 adds observability:
  - memory_check, retrieve (gate_generate), generate_node instrumented with
    child OTel spans via get_tracer().start_as_current_span().

Graph topology:
  sanitize_input ──► memory_check ──► semantic_cache_check ──► query_router → query_analyzer → retrieve → grade_documents
       │ injection          │ hit             │ hit                    ↓ all_relevant              ↑ rewrite loops back
       └──► finalize        └──► finalize     └──► finalize gate_generate ◄── partial/rewrite ┘
                                      │     ↑ web_search feeds in on none
                                      │     check_hallucination
                                      │     ↓ grounded    ↓ hallucinated (retry ≤ max_retries)
                                      └──── finalize ──── gate_generate
                                                 ↓
                                             audit_log ──► END

Two anti-loop guards:
  - should_rewrite_query: True after first rewrite.
  - retry_count / max_retries: caps generation retries.
"""
from __future__ import annotations

from typing import Literal

import structlog
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.generator import generate as _generate
from src.agents.hallucination_checker import check_hallucination
from src.agents.query_analyzer import query_analyzer
from src.agents.query_rewriter import rewrite_query
from src.agents.query_router import query_router as _query_router_node
from src.agents.relevance_grader import grade_documents
from src.agents.retriever import retrieve
from src.agents.web_search import web_search
from src.gateway.rate_limiter import get_default_rate_limiter
from src.gateway.security import AuditLogger, InputSanitizer, PromptInjectionDetected
from src.graph.state import AgentState
from src.observability.tracing import get_tracer
from src.retrieval.memory import get_default_memory
from src.retrieval.semantic_cache import get_default_semantic_cache
from src.utils.llm import DummyLLM, get_llm

logger = structlog.get_logger()

_sanitizer = InputSanitizer()
_audit_logger = AuditLogger()


# ── Security: sanitize input ───────────────────────────────────────────────────


async def sanitize_input(state: AgentState) -> dict:
    """
    LangGraph node: first line of defence against prompt injection.

    If the query matches a known injection pattern, short-circuits to
    finalize with an error message so the user receives a clear refusal
    rather than a pipeline error.
    """
    try:
        _sanitizer.sanitize(state["query"])
        return {"agent_trace": [{"node": "sanitize_input", "clean": True}]}
    except PromptInjectionDetected as exc:
        logger.warning("injection_blocked", query_prefix=state["query"][:60])
        return {
            "final_answer": f"Request blocked: {exc}",
            "agent_trace": [
                {"node": "sanitize_input", "clean": False, "reason": str(exc)}
            ],
        }


def route_after_sanitize(
    state: AgentState,
) -> Literal["memory_check", "finalize"]:
    """Route to finalize immediately when injection was detected."""
    return "finalize" if state.get("final_answer") else "memory_check"


# ── Rate-limited generate wrapper ──────────────────────────────────────────────


async def gate_generate(state: AgentState) -> dict:
    """
    LangGraph node: acquire rate-limiter capacity, then generate.

    Wraps the agent ``generate`` function so we can enforce per-model RPM/TPM
    limits without modifying the agent itself.  Instrumented with an OTel span.
    """
    with get_tracer().start_as_current_span("generate_node"):
        await get_default_rate_limiter().wait_for_capacity("gpt-4o-mini", 500)
        return await _generate(state)


# ── Audit log node ─────────────────────────────────────────────────────────────


async def audit_log(state: AgentState) -> dict:
    """LangGraph node: append an audit record after every completed query."""
    model = "dummy" if isinstance(get_llm(), DummyLLM) else "gpt-4o-mini"
    nodes_visited = [t.get("node", "unknown") for t in state.get("agent_trace", [])]
    _audit_logger.log(
        input_text=state.get("query", ""),
        output_text=state.get("final_answer", ""),
        model=model,
        cost=state.get("cost_so_far", 0.0),
        agents_used=nodes_visited,
    )
    return {"agent_trace": [{"node": "audit_log"}]}


# ── Memory check node ──────────────────────────────────────────────────────────


async def memory_check(state: AgentState) -> dict:
    """
    LangGraph node: short-circuit the pipeline if QueryMemory has a
    high-quality cached answer for this exact query.

    Memory pattern: answering from memory costs zero LLM calls and zero
    retrieval latency. The quality bar (min_faithfulness=0.85) ensures
    we only serve answers we're confident in.  Instrumented with an OTel span.
    """
    with get_tracer().start_as_current_span("memory_check_node"):
        memory = get_default_memory()
        result = await memory.recall(state["query"])

        if result:
            logger.info(
                "memory_check_hit", query=state["query"][:80], score=result.eval_score
            )
            return {
                "final_answer": result.answer,
                "citations": result.citations,
                "agent_trace": [
                    {"node": "memory_check", "hit": True, "score": result.eval_score}
                ],
            }

        logger.info("memory_check_miss", query=state["query"][:80])
        return {
            "agent_trace": [{"node": "memory_check", "hit": False}],
        }


def route_after_memory_check(
    state: AgentState,
) -> Literal["finalize", "semantic_cache_check"]:
    """Route to finalize on cache hit, otherwise try the semantic cache."""
    return "finalize" if state.get("final_answer") else "semantic_cache_check"


# ── Semantic cache check node ──────────────────────────────────────────────────


async def semantic_cache_check(state: AgentState) -> dict:
    """
    LangGraph node: check the semantic cache before running full RAG.

    Accepts a pre-computed query embedding from AgentState (populated by
    earlier retrieval nodes) to avoid re-encoding the query.  On a cache hit
    the pipeline short-circuits directly to finalize.
    """
    cache = get_default_semantic_cache()
    embedding = state.get("query_embedding")
    result = await cache.get(state["query"], embedding=embedding)

    if result:
        logger.info(
            "semantic_cache_hit",
            similarity=result.similarity,
            query=state["query"][:80],
        )
        return {
            "final_answer": result.answer,
            "citations": result.citations,
            "agent_trace": [
                {
                    "node": "semantic_cache_check",
                    "hit": True,
                    "similarity": result.similarity,
                    "score": result.eval_score,
                }
            ],
        }

    logger.info("semantic_cache_miss", query=state["query"][:80])
    return {"agent_trace": [{"node": "semantic_cache_check", "hit": False}]}


def route_after_semantic_cache(
    state: AgentState,
) -> Literal["finalize", "query_router"]:
    """Route to finalize on semantic cache hit, otherwise run full RAG."""
    return "finalize" if state.get("final_answer") else "query_router"


# ── Retrieve node (instrumented) ───────────────────────────────────────────────


async def retrieve_node(state: AgentState) -> dict:
    """LangGraph node: run retrieval with an OTel child span."""
    with get_tracer().start_as_current_span("retrieve_node"):
        return await retrieve(state)


# ── Terminal node ──────────────────────────────────────────────────────────────


async def finalize(state: AgentState) -> dict:
    """
    LangGraph node: commit the best-available generation as final_answer.

    If final_answer is already populated (memory hit or prior finalize),
    we preserve it rather than overwriting with an empty generation.
    """
    if state.get("final_answer"):
        # Already set by memory_check or a previous pass — preserve it
        logger.info("finalized", source="preserved", answer_length=len(state["final_answer"]))
        return {"agent_trace": [{"node": "finalize", "source": "preserved"}]}

    generation = state.get("generation", "")
    query = state.get("query", "")
    final = generation or f"Unable to generate a grounded answer for: {query}"

    logger.info("finalized", answer_length=len(final), had_generation=bool(generation))
    return {
        "final_answer": final,
        "agent_trace": [{"node": "finalize"}],
    }


# ── Routing functions ──────────────────────────────────────────────────────────


def route_after_grading(
    state: AgentState,
) -> Literal["generate", "rewrite_query", "web_search"]:
    """
    Branch on document relevance.

    If we already rewrote the query once (should_rewrite_query=True), skip
    straight to generate on partial results to avoid an infinite rewrite loop.
    """
    docs_relevant = state.get("docs_relevant", "none")
    already_rewrote = state.get("should_rewrite_query", False)

    if docs_relevant == "all_relevant":
        return "generate"

    if docs_relevant == "partial":
        return "generate" if already_rewrote else "rewrite_query"

    # none
    return "generate" if already_rewrote else "web_search"


def route_after_hallucination(
    state: AgentState,
) -> Literal["finalize", "generate"]:
    """
    Retry generation when hallucinated, cap at max_retries to prevent loops.
    """
    is_hallucinated = state.get("is_hallucinated", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if not is_hallucinated:
        return "finalize"

    if retry_count < max_retries:
        return "generate"

    logger.warning(
        "max_retries_reached",
        retry_count=retry_count,
        max_retries=max_retries,
        forcing="finalize",
    )
    return "finalize"


# ── Graph construction ─────────────────────────────────────────────────────────


def build_simple_workflow() -> StateGraph:
    """Assemble and return the Corrective RAG StateGraph (uncompiled)."""
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("sanitize_input", sanitize_input)
    workflow.add_node("memory_check", memory_check)
    workflow.add_node("semantic_cache_check", semantic_cache_check)
    workflow.add_node("query_router", _query_router_node)
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", gate_generate)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("check_hallucination", check_hallucination)
    workflow.add_node("finalize", finalize)
    workflow.add_node("audit_log", audit_log)

    # Entry via sanitizer → memory check
    workflow.set_entry_point("sanitize_input")
    workflow.add_conditional_edges(
        "sanitize_input",
        route_after_sanitize,
        {"memory_check": "memory_check", "finalize": "finalize"},
    )
    workflow.add_conditional_edges(
        "memory_check",
        route_after_memory_check,
        {"finalize": "finalize", "semantic_cache_check": "semantic_cache_check"},
    )
    workflow.add_conditional_edges(
        "semantic_cache_check",
        route_after_semantic_cache,
        {"finalize": "finalize", "query_router": "query_router"},
    )

    # Linear path
    workflow.add_edge("query_router", "query_analyzer")
    workflow.add_edge("query_analyzer", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    # Branch after grading
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "web_search": "web_search",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_hallucination")

    # Branch after hallucination check
    workflow.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {"finalize": "finalize", "generate": "generate"},
    )

    # finalize → audit → END
    workflow.add_edge("finalize", "audit_log")
    workflow.add_edge("audit_log", END)

    return workflow


# ── Public API ─────────────────────────────────────────────────────────────────
_checkpointer = MemorySaver()
graph = build_simple_workflow().compile(checkpointer=_checkpointer)


def get_initial_state(query: str, max_retries: int = 2) -> AgentState:
    """Return a fully-populated zero-value initial state for a new query."""
    return AgentState(
        query=query,
        chat_history=[],
        query_type=None,
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
        max_retries=max_retries,
        should_rewrite_query=False,
        final_answer=None,
        query_embedding=None,
        cost_so_far=0.0,
        agent_trace=[],
    )
