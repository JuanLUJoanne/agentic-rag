"""
Query rewriter — Corrective RAG pattern.

When the grader finds retrieved documents are only partially relevant, the
most likely cause is a vocabulary mismatch between the user's phrasing and
the document index. Rewriting to use alternative terminology — synonyms,
expanded acronyms, more technical language — before a second retrieval
attempt often recovers the missing documents.
"""
from __future__ import annotations

import structlog

from src.graph.state import AgentState
from src.utils.llm import DummyLLM, get_llm

logger = structlog.get_logger()

_REWRITE_PROMPT = """\
You are a query optimization expert for a document retrieval system.
Rewrite the query below to improve retrieval recall.
Use alternative terminology, expanded acronyms, and more specific phrasing.
Return only the rewritten query — no explanation.

Original query: {query}

Rewritten query:"""


async def rewrite_query(state: AgentState) -> dict:
    """
    LangGraph node: reformulate the query for a second retrieval attempt.

    Sets should_rewrite_query=True to signal to the grader's routing function
    that we've already tried rewriting — preventing an infinite rewrite→retrieve→
    grade→rewrite loop if retrieval quality remains poor after rewriting.
    """
    query = state["query"]
    llm = get_llm()

    if isinstance(llm, DummyLLM):
        rewritten = query  # No-op in test mode; retriever returns mock docs anyway
    else:
        prompt = _REWRITE_PROMPT.format(query=query)
        response = await llm.ainvoke(prompt)
        rewritten = response.content.strip() or query

    logger.info(
        "query_rewritten",
        original_length=len(query),
        rewritten_length=len(rewritten),
        changed=rewritten != query,
    )

    return {
        "query": rewritten,
        "should_rewrite_query": True,
        "agent_trace": [{"node": "rewrite_query", "rewritten": rewritten[:80]}],
    }
