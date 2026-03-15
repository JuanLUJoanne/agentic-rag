"""
Web search fallback — Tool Use pattern.

When local retrieval returns no relevant documents, the system falls back
to live web search rather than hallucinating from stale or missing knowledge.

Batch 1: stub returns empty results so the workflow completes gracefully.
Batch 3: integrates Tavily search via TAVILY_API_KEY; the node signature
and state contract are frozen here so the workflow graph doesn't change.
"""
from __future__ import annotations

import structlog

from src.graph.state import AgentState

logger = structlog.get_logger()

_STUB_NOTE = (
    "Web search is not yet configured. "
    "Set TAVILY_API_KEY and install tavily-python to enable live search."
)


async def web_search(state: AgentState) -> dict:
    """
    LangGraph node: fetch real-time results when local retrieval is exhausted.

    The stub returns a single explanatory doc so downstream nodes don't
    break on an empty doc list. The generator degrades gracefully on
    low-quality context rather than crashing.
    """
    query = state.get("query", "")

    logger.info("web_search_fallback", query=query[:80], status="stub")

    # Stub doc keeps the pipeline alive; Batch 3 replaces with real results
    stub_doc = {
        "id": "web_stub",
        "content": _STUB_NOTE,
        "source": "web_search_stub",
        "score": 0.0,
    }

    return {
        "retrieved_docs": [stub_doc],
        "retrieval_strategy": "web_search",
        "agent_trace": [{"node": "web_search", "status": "stub"}],
    }
