"""
Dense (embedding) retriever — mock implementation.

Interface mirrors the production pgvector retriever so the parallel
retriever and tests stay unchanged when the real implementation lands.

Production path (TODO):
  1. Encode query with sentence-transformers or OpenAI embeddings
  2. Execute: SELECT id, content, 1 - (embedding <=> %s) AS score
              FROM documents ORDER BY score DESC LIMIT %s
  3. Return SearchResult list
"""
from __future__ import annotations

import structlog

from src.retrieval.models import SearchResult

logger = structlog.get_logger()

_MOCK_DENSE_DOCS: list[dict] = [
    {
        "id": "tech_01",  # shared with BM25 — tests RRF boosting
        "content": "LangGraph is a library for building stateful, multi-actor applications with large language models using a graph-based execution model.",
        "score": 0.93,
    },
    {
        "id": "tech_07",  # shared with BM25
        "content": "Dense retrieval encodes queries and documents into continuous vector spaces using bi-encoder models like BERT or E5.",
        "score": 0.88,
    },
    {
        "id": "dense_01",
        "content": "Semantic search finds conceptually related documents even when query and document use different vocabulary.",
        "score": 0.82,
    },
    {
        "id": "dense_02",
        "content": "Bi-encoder models independently encode query and document, enabling pre-computation of document embeddings at index time.",
        "score": 0.79,
    },
    {
        "id": "dense_03",
        "content": "Cross-encoder models jointly encode query and document for higher accuracy but cannot pre-compute document representations.",
        "score": 0.75,
    },
]


class DenseRetriever:
    """
    Mock dense retriever for Batch 3.

    Returns a fixed ranked list so the RRF merger, cache, and workflow
    tests work deterministically without a running pgvector instance.
    Shares doc_id 'tech_01' and 'tech_07' with BM25Retriever so that
    RRF boosting tests can verify cross-source document promotion.
    """

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        TODO: replace with real pgvector cosine similarity search.
        SELECT id, content, 1-(embedding<=>query_vec) score FROM docs
        ORDER BY score DESC LIMIT top_k
        """
        if not query.strip():
            return []
        results = [
            SearchResult(
                doc_id=d["id"],
                content=d["content"],
                score=d["score"],
                source="dense",
            )
            for d in _MOCK_DENSE_DOCS[:top_k]
        ]
        logger.debug("dense_search_complete", doc_count=len(results), mock=True)
        return results
