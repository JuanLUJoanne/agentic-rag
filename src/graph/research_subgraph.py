"""
Research subgraph — multi-step retrieval with rewrite loop.

Replaces the single-call ResearchAgent node with a LangGraph subgraph
that has its own internal state and multi-step logic::

    retrieve → grade_relevance ──[all_relevant]──► synthesize → END
                                ──[partial/none]──► rewrite_query → retrieve  (max 2 rewrites)

Design decisions:
  - ResearchState is independent of SupervisorState — the subgraph
    receives only the fields it needs (query, retrieved_docs) and returns
    only what the outer graph consumes (retrieved_docs, relevance_scores,
    docs_relevant, retrieval_strategy, agent_trace).
  - State mapping is done via wrapper functions (research_subgraph_entry /
    research_subgraph_exit) so the compiled subgraph can be embedded
    directly as a node in the outer multi-agent workflow.
  - The rewrite loop is capped at 2 iterations via rewrite_count /
    max_rewrites — same anti-loop pattern as simple_workflow's
    should_rewrite_query but scoped to this subgraph.

This is the multi-hop layer: when the first retrieval pass returns
insufficient evidence, the query is reformulated and re-issued.
"""
from __future__ import annotations

from typing import Literal

import structlog
from langgraph.graph import END, StateGraph

from src.agents.relevance_grader import grade_documents
from src.agents.retriever import retrieve
from src.graph.state import AgentState
from src.observability.tracing import get_tracer, set_span_ok
from src.utils.llm import DummyLLM, get_llm

logger = structlog.get_logger()

_MAX_REWRITES = 2


# ── Internal subgraph state ──────────────────────────────────────────────────

class ResearchState(AgentState):
    """State local to the research subgraph.

    Extends AgentState so existing node functions (retrieve, grade_documents)
    work unmodified.  Adds rewrite_count to limit the internal rewrite loop.
    """

    rewrite_count: int
    max_rewrites: int


# ── Subgraph nodes ───────────────────────────────────────────────────────────

async def _retrieve_node(state: ResearchState) -> dict:
    """Run parallel retrieval with OTel span."""
    with get_tracer().start_as_current_span("research_retrieve") as span:
        span.set_attribute("rag.query", state.get("query", "")[:100])
        span.set_attribute("rewrite.count", state.get("rewrite_count", 0))
        result = await retrieve(state)
        doc_count = len(result.get("retrieved_docs", []))
        span.set_attribute("retrieval.doc_count", doc_count)
        set_span_ok(span)
        logger.info(
            "research_subgraph_retrieve",
            doc_count=doc_count,
            rewrite_count=state.get("rewrite_count", 0),
        )
        return result


async def _grade_node(state: ResearchState) -> dict:
    """Grade document relevance."""
    with get_tracer().start_as_current_span("research_grade") as span:
        span.set_attribute("retrieval.doc_count", len(state.get("retrieved_docs", [])))
        result = await grade_documents(state)
        span.set_attribute("grade.docs_relevant", result.get("docs_relevant", "unknown"))
        set_span_ok(span)
        return result


async def _rewrite_node(state: ResearchState) -> dict:
    """Rewrite the query for a better retrieval pass.

    Uses the same DummyLLM-compatible rewrite approach as simple_workflow:
    DummyLLM appends 'rewritten:' prefix; real LLM reformulates.
    """
    with get_tracer().start_as_current_span("research_rewrite") as span:
        query = state["query"]
        span.set_attribute("rewrite.iteration", state.get("rewrite_count", 0) + 1)
        llm = get_llm()

        if isinstance(llm, DummyLLM):
            new_query = f"rewritten: {query}"
        else:
            prompt = (
                f"Rewrite this search query to find more relevant documents. "
                f"Return only the rewritten query, nothing else.\n\n"
                f"Original query: {query}"
            )
            response = await llm.ainvoke(prompt)
            new_query = response.content.strip() or query

        rewrite_count = state.get("rewrite_count", 0) + 1
        set_span_ok(span)
        logger.info(
            "research_subgraph_rewrite",
            original=query[:80],
            rewritten=new_query[:80],
            rewrite_count=rewrite_count,
        )
        return {
            "query": new_query,
            "rewrite_count": rewrite_count,
            "agent_trace": [
                {
                    "node": "research_rewrite",
                    "original_query": query[:80],
                    "rewritten_query": new_query[:80],
                    "rewrite_count": rewrite_count,
                }
            ],
        }


async def _synthesize_node(state: ResearchState) -> dict:
    """Produce a research summary from the graded documents.

    For DummyLLM: returns a concatenation of doc contents.
    For real LLM: would produce a synthesis paragraph (future extension).
    """
    with get_tracer().start_as_current_span("research_synthesize") as span:
        docs = state.get("retrieved_docs", [])
        doc_count = len(docs)
        span.set_attribute("retrieval.doc_count", doc_count)

        # Build a summary from retrieved docs
        if docs:
            summary_parts = [
                doc.get("content", "")[:200] for doc in docs[:5]
            ]
            summary = " | ".join(summary_parts)
        else:
            summary = "No relevant documents found."

        span.set_attribute("synthesis.output_length", len(summary))
        set_span_ok(span)
        logger.info("research_subgraph_synthesize", doc_count=doc_count)
        return {
            "agent_trace": [
                {"node": "research_synthesize", "doc_count": doc_count}
            ],
        }


# ── Routing ──────────────────────────────────────────────────────────────────

def _route_after_grade(
    state: ResearchState,
) -> Literal["synthesize", "rewrite"]:
    """Branch on document relevance, respecting the rewrite cap."""
    docs_relevant = state.get("docs_relevant", "none")
    rewrite_count = state.get("rewrite_count", 0)
    max_rewrites = state.get("max_rewrites", _MAX_REWRITES)

    if docs_relevant == "all_relevant":
        return "synthesize"

    # partial or none — try rewrite if under the cap
    if rewrite_count < max_rewrites:
        return "rewrite"

    # Exhausted rewrites — synthesize with what we have
    logger.info(
        "research_subgraph_rewrite_exhausted",
        rewrite_count=rewrite_count,
        docs_relevant=docs_relevant,
    )
    return "synthesize"


# ── Graph construction ───────────────────────────────────────────────────────

def build_research_subgraph() -> StateGraph:
    """Assemble the research subgraph (uncompiled)."""
    sg = StateGraph(ResearchState)

    sg.add_node("retrieve", _retrieve_node)
    sg.add_node("grade", _grade_node)
    sg.add_node("rewrite", _rewrite_node)
    sg.add_node("synthesize", _synthesize_node)

    sg.set_entry_point("retrieve")
    sg.add_edge("retrieve", "grade")

    sg.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"synthesize": "synthesize", "rewrite": "rewrite"},
    )

    # Rewrite loops back to retrieve
    sg.add_edge("rewrite", "retrieve")
    sg.add_edge("synthesize", END)

    return sg


research_compiled = build_research_subgraph().compile()


# ── State mapping helpers ────────────────────────────────────────────────────

def to_research_state(outer_state: dict) -> ResearchState:
    """Map SupervisorState fields into ResearchState for subgraph entry."""
    return ResearchState(
        query=outer_state.get("query", ""),
        chat_history=outer_state.get("chat_history", []),
        query_type=outer_state.get("query_type"),
        sub_queries=outer_state.get("sub_queries", []),
        retrieved_docs=outer_state.get("retrieved_docs", []),
        graph_context=outer_state.get("graph_context", []),
        relevance_scores=outer_state.get("relevance_scores", []),
        retrieval_strategy=outer_state.get("retrieval_strategy", "dense"),
        generation=outer_state.get("generation", ""),
        citations=outer_state.get("citations", []),
        docs_relevant=outer_state.get("docs_relevant"),
        is_hallucinated=outer_state.get("is_hallucinated"),
        answer_quality=outer_state.get("answer_quality"),
        retry_count=outer_state.get("retry_count", 0),
        max_retries=outer_state.get("max_retries", 2),
        should_rewrite_query=False,
        final_answer=outer_state.get("final_answer"),
        query_embedding=outer_state.get("query_embedding"),
        cost_so_far=outer_state.get("cost_so_far", 0.0),
        agent_trace=[],
        rewrite_count=0,
        max_rewrites=_MAX_REWRITES,
    )


def from_research_state(research_result: dict) -> dict:
    """Map research subgraph output back to SupervisorState fields."""
    return {
        "retrieved_docs": research_result.get("retrieved_docs", []),
        "relevance_scores": research_result.get("relevance_scores", []),
        "docs_relevant": research_result.get("docs_relevant"),
        "retrieval_strategy": research_result.get("retrieval_strategy", "parallel_simple"),
        "agent_trace": [
            {"node": "research_agent", "subgraph": True},
            *research_result.get("agent_trace", []),
        ],
        "agents_called": ["research"],
    }
