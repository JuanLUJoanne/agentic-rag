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
  - MMR post-processing (opt-in via MMR_ENABLED=true): applies Maximal
    Marginal Relevance after RRF to reduce redundant results. Uses token
    Jaccard similarity as a document-similarity proxy so it works without
    a live vector store. Enable with MMR_LAMBDA to tune relevance/diversity
    trade-off (default 0.5; higher = more relevance, lower = more diversity).
  - Lost-in-the-Middle reordering (opt-in via LITM_ENABLED=true): reorders
    results so highest-scored docs are at positions 0 and -1 to combat
    LLM attention bias toward the middle of long context.
  - Circuit breakers (per-retriever): automatically opens after repeated
    failures and probes for recovery, preventing cascade failures.
  - Contextual compression (opt-in via COMPRESSION_ENABLED=true): uses an
    LLM to extract only query-relevant sentences from each document.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from typing import NamedTuple

import structlog

from src.retrieval.bm25_retriever import SAMPLE_DOCS, BM25Retriever
from src.retrieval.circuit_breaker import CircuitBreaker, CircuitOpenError, get_circuit_breaker
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


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase word tokens — used for Jaccard similarity."""
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _mmr(
    candidates: list[SearchResult],
    top_k: int,
    lambda_: float = 0.5,
) -> list[SearchResult]:
    """
    Maximal Marginal Relevance reranking.

    Iteratively selects the candidate that best balances:
      score = λ · relevance(doc) − (1−λ) · max_similarity(doc, selected)

    relevance  — normalised RRF score from the candidate list
    similarity — Jaccard overlap of lowercased word tokens

    Works without embeddings; effective for deduplicating BM25/RRF output
    where near-duplicate passages share high token overlap.

    lambda_=1.0 → pure relevance order (identical to no MMR)
    lambda_=0.0 → pure diversity (greedy maximum coverage)
    """
    if not candidates:
        return []

    max_score = max(r.score for r in candidates) or 1.0
    tokens = [_tokenize(r.content) for r in candidates]

    selected_indices: list[int] = []
    remaining = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining:
        best_idx: int | None = None
        best_score = float("-inf")

        for i in remaining:
            relevance = candidates[i].score / max_score
            if not selected_indices:
                mmr_score = relevance
            else:
                max_sim = max(_jaccard(tokens[i], tokens[j]) for j in selected_indices)
                mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


def _lost_in_middle_reorder(results: list[SearchResult]) -> list[SearchResult]:
    """
    Reorder so highest-scored docs are at the ends of the list.
    Position 0 → rank 1, position -1 → rank 2, position 1 → rank 3, etc.
    (alternating outside-in fill)

    LLMs attend poorly to context in the middle of a long list. Placing
    the most relevant documents at the start and end maximises the chance
    that the model reads them.
    """
    if not results:
        return results

    # Results are assumed already sorted by descending score (RRF or MMR order)
    reordered: list[SearchResult | None] = [None] * len(results)
    left, right = 0, len(results) - 1
    fill_left = True

    for doc in results:
        if fill_left:
            reordered[left] = doc
            left += 1
        else:
            reordered[right] = doc
            right -= 1
        fill_left = not fill_left

    return [r for r in reordered if r is not None]


# ── Feature-flag helpers ─────────────────────────────────────────────────────

def _mmr_enabled() -> bool:
    return os.getenv("MMR_ENABLED", "false").lower() in ("1", "true", "yes")


def _mmr_lambda() -> float:
    try:
        return float(os.getenv("MMR_LAMBDA", "0.5"))
    except ValueError:
        return 0.5


def _litm_enabled() -> bool:
    return os.getenv("LITM_ENABLED", "false").lower() in ("1", "true", "yes")


def _compression_enabled() -> bool:
    return os.getenv("COMPRESSION_ENABLED", "false").lower() in ("1", "true", "yes")


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
        # Per-retriever circuit breakers
        self._breakers: dict[str, CircuitBreaker] = {
            name: get_circuit_breaker(name)
            for name in ("bm25", "dense", "graph")
        }

    async def _timed_search(
        self,
        name: str,
        retriever,
        query: str,
        top_k: int,
    ) -> _SourceResult:
        """Run one retriever with timeout and circuit breaker; return a _SourceResult regardless of outcome."""
        start = time.monotonic()
        breaker = self._breakers.get(name)
        try:
            if breaker is not None:
                results = await breaker.call(
                    asyncio.wait_for(retriever.search(query, top_k), timeout=self.timeout)
                )
            else:
                results = await asyncio.wait_for(
                    retriever.search(query, top_k), timeout=self.timeout
                )
            return _SourceResult(name=name, results=results, latency=time.monotonic() - start, error=None)
        except CircuitOpenError:
            logger.warning("circuit_open", source=name)
            return _SourceResult(name=name, results=[], latency=0.0, error="circuit_open")
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

        candidates = _rrf_merge(source_results)

        # Apply MMR if enabled
        if _mmr_enabled():
            lambda_ = _mmr_lambda()
            merged = _mmr(candidates, top_k=top_k, lambda_=lambda_)
            mmr_on = True
        else:
            merged = candidates[:top_k]
            mmr_on = False
            lambda_ = None

        # Apply Lost-in-the-Middle reordering if enabled
        litm_on = _litm_enabled()
        if litm_on:
            merged = _lost_in_middle_reorder(merged)

        # Apply contextual compression if enabled
        if _compression_enabled():
            from src.retrieval.compressor import get_compressor
            merged = await get_compressor().compress_batch(query, merged)

        log_kwargs: dict = dict(
            total_results=len(merged),
            per_source_counts={sr.name: len(sr.results) for sr in source_results},
            per_source_latency={sr.name: round(sr.latency, 3) for sr in source_results},
            failures=failures,
            mmr_enabled=mmr_on,
            litm_enabled=litm_on,
        )
        if mmr_on and lambda_ is not None:
            log_kwargs["mmr_lambda"] = lambda_

        logger.info("parallel_retrieval_complete", **log_kwargs)
        return merged
