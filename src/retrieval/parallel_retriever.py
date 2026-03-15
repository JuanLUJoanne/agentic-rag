"""
Parallel hybrid retriever — Parallelization pattern.

Runs BM25, dense, and (for complex queries) GraphRAG concurrently via
asyncio.gather, then merges results with Reciprocal Rank Fusion (RRF).

Design choices:
  - Per-retriever timeout (default 5 s) prevents one slow source from
    blocking the whole pipeline; other sources' results are still used.
  - RRF rather than score normalisation: avoids the need to calibrate
    BM25 scores (unbounded) against cosine similarities (0–1).
  - query_type gating: graph retrieval adds latency and is only useful
    for multi-hop queries, so simple lookups skip it.
"""
from __future__ import annotations

import asyncio
import time
from typing import NamedTuple

import structlog

from src.retrieval.bm25_retriever import SAMPLE_DOCS, BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.models import SearchResult

logger = structlog.get_logger()


class _SourceResult(NamedTuple):
    name: str
    results: list[SearchResult]
    latency: float
    error: str | None


def _rrf_merge(
    source_results: list[_SourceResult],
    k: int = 60,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion across all source result lists.

    score(doc) = Σ  1 / (k + rank_i(doc))   for each source i

    k=60 follows the original RRF paper (Cormack et al., 2009).
    Documents that appear in multiple sources receive additive score
    boosts — this is the key cross-source promotion property.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, SearchResult] = {}

    for sr in source_results:
        if sr.error or not sr.results:
            continue
        for rank, result in enumerate(sr.results):
            rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0.0) + 1.0 / (k + rank + 1)
            if result.doc_id not in doc_map:
                doc_map[result.doc_id] = result

    merged = []
    for doc_id in sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True):
        original = doc_map[doc_id]
        merged.append(
            SearchResult(
                doc_id=original.doc_id,
                content=original.content,
                score=round(rrf_scores[doc_id], 6),
                source="rrf_merged",
                metadata={**original.metadata, "original_source": original.source},
            )
        )
    return merged


class ParallelRetriever:
    """
    Concurrent hybrid retriever with RRF fusion and per-source timeout.

    The internal BM25 index is pre-loaded with SAMPLE_DOCS so the
    retriever works in Batch 1-4 without a database.  Batch 5 replaces
    SAMPLE_DOCS with real document ingestion.
    """

    def __init__(self, timeout: float = 5.0) -> None:
        self.bm25 = BM25Retriever()
        self.bm25.index(SAMPLE_DOCS)
        self.dense = DenseRetriever()
        self.graph = GraphRetriever()
        self.timeout = timeout

    async def _timed_search(
        self,
        name: str,
        retriever,
        query: str,
        top_k: int,
    ) -> _SourceResult:
        """Run one retriever with timeout; return a _SourceResult regardless of outcome."""
        start = time.monotonic()
        try:
            results = await asyncio.wait_for(
                retriever.search(query, top_k), timeout=self.timeout
            )
            return _SourceResult(name=name, results=results, latency=time.monotonic() - start, error=None)
        except TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning("retriever_timeout", source=name, timeout=self.timeout)
            return _SourceResult(name=name, results=[], latency=elapsed, error="timeout")
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.warning("retriever_error", source=name, error=str(exc))
            return _SourceResult(name=name, results=[], latency=elapsed, error=str(exc))

    async def retrieve(
        self,
        query: str,
        query_type: str = "simple",
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Run retrievers concurrently, merge via RRF, return top_k results.

        query_type="simple"  → BM25 + dense  (graph adds latency for no gain)
        query_type="complex" → BM25 + dense + graph  (multi-hop needs entities)
        """
        retrievers: list[tuple[str, object, int]] = [
            ("bm25", self.bm25, top_k * 2),   # fetch 2× so RRF has headroom
            ("dense", self.dense, top_k * 2),
        ]
        if query_type == "complex":
            retrievers.append(("graph", self.graph, top_k))

        source_results: list[_SourceResult] = await asyncio.gather(
            *[self._timed_search(name, r, query, k) for name, r, k in retrievers]
        )

        failures = [sr.name for sr in source_results if sr.error]
        if failures:
            logger.warning("retriever_failures", sources=failures)

        merged = _rrf_merge(source_results)[:top_k]

        logger.info(
            "parallel_retrieval_complete",
            total_results=len(merged),
            per_source_counts={sr.name: len(sr.results) for sr in source_results},
            per_source_latency={sr.name: round(sr.latency, 3) for sr in source_results},
            failures=failures,
        )
        return merged
