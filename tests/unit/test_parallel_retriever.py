"""Unit tests for ParallelRetriever: concurrency, failure handling, RRF, timeouts."""
from __future__ import annotations

import asyncio
import time

import pytest

from src.retrieval.models import SearchResult
from src.retrieval.parallel_retriever import ParallelRetriever, _rrf_merge, _SourceResult

# ── _rrf_merge pure-function tests ─────────────────────────────────────────────


def _sr(doc_id: str, score: float = 0.9, source: str = "bm25") -> SearchResult:
    return SearchResult(doc_id=doc_id, content=f"content of {doc_id}", score=score, source=source)


def _source(name: str, results: list[SearchResult], error=None) -> _SourceResult:
    return _SourceResult(name=name, results=results, latency=0.01, error=error)


def test_rrf_doc_in_multiple_sources_ranks_higher():
    """A document appearing in two sources must outscore one appearing in only one."""
    bm25_results = [_sr("shared", source="bm25"), _sr("bm25_only", source="bm25")]
    dense_results = [_sr("shared", source="dense"), _sr("dense_only", source="dense")]

    merged = _rrf_merge([
        _source("bm25", bm25_results),
        _source("dense", dense_results),
    ])

    assert len(merged) > 0
    assert merged[0].doc_id == "shared"


def test_rrf_empty_sources_returns_empty():
    merged = _rrf_merge([_source("bm25", []), _source("dense", [])])
    assert merged == []


def test_rrf_single_source_preserves_order():
    results = [_sr("a", score=0.9), _sr("b", score=0.5), _sr("c", score=0.3)]
    merged = _rrf_merge([_source("bm25", results)])
    assert [r.doc_id for r in merged] == ["a", "b", "c"]


def test_rrf_failed_source_ignored():
    """A source with error=... should contribute no results."""
    good_results = [_sr("good_doc")]
    merged = _rrf_merge([
        _source("bm25", good_results),
        _source("dense", [], error="timeout"),
    ])
    assert len(merged) == 1
    assert merged[0].doc_id == "good_doc"


def test_rrf_scores_are_positive():
    merged = _rrf_merge([_source("bm25", [_sr("d1"), _sr("d2")])])
    assert all(r.score > 0 for r in merged)


def test_rrf_source_becomes_rrf_merged():
    merged = _rrf_merge([_source("bm25", [_sr("d1")])])
    assert merged[0].source == "rrf_merged"


# ── ParallelRetriever integration tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_returns_results():
    pr = ParallelRetriever()
    results = await pr.retrieve("LangGraph stateful applications", query_type="simple")
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_simple_query_skips_graph_retriever():
    """query_type='simple' must NOT call the graph retriever."""
    pr = ParallelRetriever()
    graph_called = []

    original = pr.graph.search
    async def track_graph(q, top_k=5):
        graph_called.append(True)
        return await original(q, top_k)

    pr.graph.search = track_graph
    await pr.retrieve("simple question", query_type="simple")
    assert len(graph_called) == 0


@pytest.mark.asyncio
async def test_complex_query_includes_graph_retriever():
    """query_type='complex' must call the graph retriever."""
    pr = ParallelRetriever()
    graph_called = []

    original = pr.graph.search
    async def track_graph(q, top_k=5):
        graph_called.append(True)
        return await original(q, top_k)

    pr.graph.search = track_graph
    await pr.retrieve("multi-hop complex query", query_type="complex")
    assert len(graph_called) == 1


@pytest.mark.asyncio
async def test_all_three_sources_run_for_complex():
    """All three retrievers must be called for complex queries."""
    pr = ParallelRetriever()
    called = []

    for name in ("bm25", "dense", "graph"):
        original_retriever = getattr(pr, name)
        original_search = original_retriever.search

        async def make_tracker(orig, n):
            async def _track(q, top_k=10):
                called.append(n)
                return await orig(q, top_k)
            return _track

        original_retriever.search = await make_tracker(original_search, name)

    await pr.retrieve("compare BM25 and dense retrieval", query_type="complex")
    assert "bm25" in called
    assert "dense" in called
    assert "graph" in called


@pytest.mark.asyncio
async def test_one_source_failure_does_not_kill_others():
    """A failing BM25 retriever must not prevent dense results from being returned."""
    pr = ParallelRetriever()

    async def always_fails(q, top_k=10):
        raise RuntimeError("BM25 index unavailable")

    pr.bm25.search = always_fails
    # Should complete without raising
    results = await pr.retrieve("some query", query_type="simple")
    assert isinstance(results, list)
    # Dense results should still come through
    assert len(results) > 0


@pytest.mark.asyncio
async def test_all_sources_fail_returns_empty():
    pr = ParallelRetriever()

    async def always_fails(q, top_k=10):
        raise RuntimeError("down")

    pr.bm25.search = always_fails
    pr.dense.search = always_fails
    results = await pr.retrieve("query", query_type="simple")
    assert results == []


@pytest.mark.asyncio
async def test_timeout_does_not_block_pipeline():
    """A slow retriever must be cancelled at the timeout boundary."""
    pr = ParallelRetriever(timeout=0.05)  # 50 ms timeout

    async def slow_bm25(q, top_k=10):
        await asyncio.sleep(1.0)  # much longer than timeout
        return []

    pr.bm25.search = slow_bm25
    start = time.monotonic()
    results = await pr.retrieve("test", query_type="simple")
    elapsed = time.monotonic() - start

    assert elapsed < 0.5  # pipeline completed well before slow retriever would have
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_top_k_limits_output():
    pr = ParallelRetriever()
    results = await pr.retrieve("retrieval", query_type="complex", top_k=3)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_rrf_boost_visible_in_retrieve():
    """
    Inject both bm25 and dense returning 'tech_01' to confirm RRF
    places the cross-source doc at rank 1.
    """
    pr = ParallelRetriever()

    shared = SearchResult(doc_id="tech_01", content="shared doc", score=0.9, source="bm25")
    unique = SearchResult(doc_id="unique_dense", content="unique", score=0.95, source="dense")

    async def mock_bm25(q, top_k=10):
        return [shared]

    async def mock_dense(q, top_k=10):
        return [
            SearchResult(doc_id="tech_01", content="shared doc", score=0.92, source="dense"),
            unique,
        ]

    pr.bm25.search = mock_bm25
    pr.dense.search = mock_dense

    results = await pr.retrieve("query", query_type="simple", top_k=5)
    assert results[0].doc_id == "tech_01"
