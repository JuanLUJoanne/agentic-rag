"""
Graph (Neo4j) retriever — mock implementation.

Only dispatched for complex/multi-hop queries where entity relationships
matter. Dense and BM25 retrievers handle most simple queries adequately.

Production path (TODO):
  MATCH (e:Entity {name: $entity})-[:RELATED_TO*1..2]-(related)
  RETURN related.content AS content, related.id AS id, ...
  ORDER BY score DESC LIMIT $top_k
"""
from __future__ import annotations

import structlog

from src.retrieval.models import SearchResult

logger = structlog.get_logger()

_MOCK_GRAPH_DOCS: list[dict] = [
    {
        "id": "graph_01",
        "content": "LangGraph extends LangChain with stateful graph execution. Nodes are agents; edges encode control flow and state transitions.",
        "score": 0.87,
    },
    {
        "id": "graph_02",
        "content": "Knowledge graph traversal enables multi-hop reasoning: starting from a seed entity, following typed edges to discover related facts.",
        "score": 0.81,
    },
    {
        "id": "graph_03",
        "content": "Entity linking connects surface mentions in text to canonical knowledge-graph nodes, enabling structured retrieval over unstructured documents.",
        "score": 0.74,
    },
]


class GraphRetriever:
    """
    Mock graph retriever for Batch 3.

    Returns relationship-enriched context for complex queries.
    Real Neo4j implementation lands in Batch 5 alongside the full
    knowledge-graph indexing pipeline.
    """

    async def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        TODO: replace with real Neo4j Cypher traversal.
        Uses APOC for weighted path scoring and entity disambiguation.
        """
        if not query.strip():
            return []
        results = [
            SearchResult(
                doc_id=d["id"],
                content=d["content"],
                score=d["score"],
                source="graph",
            )
            for d in _MOCK_GRAPH_DOCS[:top_k]
        ]
        logger.debug("graph_search_complete", doc_count=len(results), mock=True)
        return results
