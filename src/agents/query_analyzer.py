"""
Query analyzer: LLM-based decomposition for complex queries (Planning pattern).

For multi-hop questions the retriever needs focused sub-queries, not one
sprawling prompt. Breaking "A and B and C" into ["A", "B", "C"] before
retrieval dramatically improves recall on each dimension.
"""
from __future__ import annotations

import re

import structlog

from src.graph.state import AgentState
from src.utils.llm import DummyLLM, get_llm

logger = structlog.get_logger()

_DECOMPOSE_PROMPT = """\
You are a query decomposition expert.
Break the following complex query into 2-4 specific, focused sub-queries that together \
answer the original question.
Return only a newline-separated list of sub-queries — no numbering, no explanation.

Query: {query}

Sub-queries:"""


async def query_analyzer(state: AgentState) -> dict:
    """
    LangGraph node: decompose complex queries into focused sub-queries.

    Pass-through for simple/ambiguous queries. For complex queries, LLM
    decomposition improves multi-hop retrieval because each sub-query maps
    to a specific document cluster rather than pulling from everywhere.
    """
    query_type = state.get("query_type", "simple")
    query = state["query"]

    if query_type != "complex":
        logger.info("query_analyzed", query_type=query_type, sub_query_count=1, skipped=True)
        return {
            "sub_queries": state.get("sub_queries") or [query],
            "agent_trace": [{"node": "query_analyzer", "skipped": True}],
        }

    llm = get_llm()

    if isinstance(llm, DummyLLM):
        # Deterministic fallback: split on ' and ' or return original query
        parts = [p.strip() for p in re.split(r"\s+and\s+", query, flags=re.IGNORECASE)]
        sub_queries = [p for p in parts if p] or [query]
    else:
        prompt = _DECOMPOSE_PROMPT.format(query=query)
        response = await llm.ainvoke(prompt)
        raw = response.content.strip()
        sub_queries = [line.strip() for line in raw.splitlines() if line.strip()] or [query]

    logger.info(
        "query_analyzed",
        query_type=query_type,
        sub_query_count=len(sub_queries),
    )

    return {
        "sub_queries": sub_queries,
        "agent_trace": [{"node": "query_analyzer", "sub_query_count": len(sub_queries)}],
    }
