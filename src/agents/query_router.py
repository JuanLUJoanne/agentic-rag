"""
Heuristic query router — no LLM, no cost.

LLM-based routing added measurable latency and cost in early prototypes
without improving downstream recall. Keyword + word-count heuristics are
fast, deterministic, and sufficient to separate simple lookups from
multi-hop comparisons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import structlog

from src.graph.state import AgentState

logger = structlog.get_logger()

# Keywords that reliably signal a comparison or analysis query
_COMPLEX_KEYWORDS: frozenset[str] = frozenset([
    "compare", "contrast", "difference", "differences", "versus", "vs",
    "relationship", "analyze", "analyse", "explain", "how does", "why does",
    "trade-off", "tradeoff", "pros and cons", "advantages",
])


@dataclass
class QueryRoute:
    type: Literal["simple", "complex", "ambiguous"]
    sub_queries: list[str] = field(default_factory=list)
    strategy: str = "dense"


def _split_on_conjunction(query: str) -> list[str]:
    """Minimal sub-query decomposition: split on ' and ' without an LLM."""
    parts = [p.strip() for p in re.split(r"\s+and\s+", query, flags=re.IGNORECASE)]
    return [p for p in parts if p]


def classify(query: str) -> QueryRoute:
    """
    Classify a query into simple / complex / ambiguous and choose a retrieval strategy.

    Rules (applied in priority order):
      1. Very short queries (≤2 words) → ambiguous; caller should ask for clarification.
      2. Short question (< 15 words, contains '?') → simple, dense retrieval.
      3. Contains a comparison keyword → complex, hybrid retrieval.
      4. Long query (> 30 words) or contains ' and ' → complex with decomposition.
      5. Default → simple, dense retrieval.
    """
    q = query.strip()
    words = q.split()
    word_count = len(words)
    lower = q.lower()

    if word_count <= 2:
        return QueryRoute(type="ambiguous", sub_queries=[q], strategy="clarification")

    # Keyword check before short-question rule: "What is the difference between X and Y?"
    # is < 15 words with '?' but is clearly a comparison query.
    if any(kw in lower for kw in _COMPLEX_KEYWORDS):
        sub_queries = _split_on_conjunction(q) if " and " in lower else [q]
        return QueryRoute(type="complex", sub_queries=sub_queries, strategy="hybrid")

    if "?" in q and word_count < 15:
        return QueryRoute(type="simple", sub_queries=[q], strategy="dense")

    if word_count > 30 or " and " in lower:
        sub_queries = _split_on_conjunction(q)
        return QueryRoute(type="complex", sub_queries=sub_queries, strategy="hybrid")

    return QueryRoute(type="simple", sub_queries=[q], strategy="dense")


async def query_router(state: AgentState) -> dict:
    """LangGraph node: classify the query and set routing metadata in state."""
    route = classify(state["query"])

    logger.info(
        "query_routed",
        query=state["query"][:80],
        type=route.type,
        strategy=route.strategy,
        sub_query_count=len(route.sub_queries),
    )

    return {
        "query_type": route.type,
        "sub_queries": route.sub_queries,
        "retrieval_strategy": route.strategy,
        "agent_trace": [{"node": "query_router", "query_type": route.type, "strategy": route.strategy}],
    }
