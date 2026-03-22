"""
Central AgentState schema for the LangGraph RAG workflow.

A single TypedDict shared by every node means LangGraph can checkpoint
the full pipeline state at each step — enabling human-in-the-loop review,
retry on failure, and full audit traces without extra infrastructure.
"""
from __future__ import annotations

import operator
from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # ── Core query ────────────────────────────────────────────────────────────
    query: str
    # Accumulates message history across turns; add_messages reducer appends
    chat_history: Annotated[list, add_messages]

    # ── Routing ───────────────────────────────────────────────────────────────
    query_type: Literal["simple", "complex", "ambiguous"] | None
    sub_queries: list[str]

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_docs: list[dict]
    graph_context: list[dict]
    relevance_scores: list[float]
    retrieval_strategy: str

    # ── Generation ────────────────────────────────────────────────────────────
    generation: str
    citations: list[dict]

    # ── Self-correction signals ───────────────────────────────────────────────
    docs_relevant: Literal["all_relevant", "partial", "none"] | None
    is_hallucinated: bool | None
    answer_quality: float | None

    # ── Loop control ──────────────────────────────────────────────────────────
    retry_count: int
    max_retries: int
    # True after the first query rewrite attempt; prevents infinite rewrite loops
    should_rewrite_query: bool

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str | None

    # ── Semantic cache ────────────────────────────────────────────────────────
    query_embedding: list[float] | None

    # ── Observability ─────────────────────────────────────────────────────────
    cost_so_far: float
    # operator.add reducer accumulates trace entries across all nodes instead of overwriting
    agent_trace: Annotated[list[dict], operator.add]
