"""
Retriever node — Tool Use pattern.

Batch 3: replaces static mock docs with ParallelRetriever (BM25 + dense +
GraphRAG via asyncio.gather) merged by Reciprocal Rank Fusion.

EmbeddingCache sits in front of ParallelRetriever: identical queries within
the TTL window skip the full retrieval pipeline entirely, which matters
especially for repeated sub-queries in multi-hop decomposition.
"""
from __future__ import annotations

import structlog

from src.graph.state import AgentState
from src.retrieval.cache import get_default_cache
from src.retrieval.parallel_retriever import ParallelRetriever

logger = structlog.get_logger()

# Module-level singletons — one retriever per process, cache shared globally
_retriever = ParallelRetriever()
_cache = get_default_cache()


async def retrieve(state: AgentState) -> dict:
    """
    LangGraph node: fetch and rank documents for the current query.

    Cache-first: if EmbeddingCache has a fresh result for this exact
    query string, we skip the full parallel retrieval and return
    immediately. This pays off when the same question appears in multiple
    sub-queries after decomposition.
    """
    query = state["query"]
    query_type = state.get("query_type") or "simple"
    sub_queries = state.get("sub_queries") or [query]

    # ── Cache check ────────────────────────────────────────────────────────────
    cached = await _cache.get(query)
    if cached:
        docs = [
            {"id": r.doc_id, "content": r.content, "source": r.source, "score": r.score}
            for r in cached
        ]
        logger.info("retrieval_complete", doc_count=len(docs), strategy="cache", cached=True)
        return {
            "retrieved_docs": docs,
            "retrieval_strategy": "cache",
            "agent_trace": [{"node": "retrieve", "doc_count": len(docs), "strategy": "cache"}],
        }

    # ── Parallel hybrid retrieval ──────────────────────────────────────────────
    results = await _retriever.retrieve(query, query_type=query_type)

    # Store in cache for future identical queries
    await _cache.set(query, results)

    docs = [
        {"id": r.doc_id, "content": r.content, "source": r.source, "score": r.score}
        for r in results
    ]
    strategy = f"parallel_{query_type}"

    logger.info(
        "retrieval_complete",
        doc_count=len(docs),
        strategy=strategy,
        sub_query_count=len(sub_queries),
        cached=False,
    )

    return {
        "retrieved_docs": docs,
        "retrieval_strategy": strategy,
        "agent_trace": [{"node": "retrieve", "doc_count": len(docs), "strategy": strategy}],
    }
