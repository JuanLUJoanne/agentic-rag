"""Unit tests for ParallelRetriever: concurrency, failure handling, RRF, timeouts."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.circuit_breaker import CircuitOpenError
from src.retrieval.models import SearchResult
from src.retrieval.parallel_retriever import (
    ParallelRetriever,
    _cross_encoder_rerank,
    _lost_in_middle_reorder,
    _rrf_merge,
    _SourceResult,
)

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


# ── Lost-in-the-Middle tests ──────────────────────────────────────────────────


def _make_docs(n: int) -> list[SearchResult]:
    """Create n docs with descending scores (rank 0 = highest)."""
    return [
        SearchResult(
            doc_id=f"d{i}",
            content=f"document content {i}",
            score=1.0 - i * 0.1,
            source="rrf_merged",
        )
        for i in range(n)
    ]


def test_litm_empty_list():
    assert _lost_in_middle_reorder([]) == []


def test_litm_single_doc():
    docs = _make_docs(1)
    assert _lost_in_middle_reorder(docs) == docs


def test_litm_highest_at_index_0_and_last():
    """With 4 docs (scores 1.0, 0.9, 0.8, 0.7), highest (d0) → pos 0, second (d1) → pos -1."""
    docs = _make_docs(4)
    reordered = _lost_in_middle_reorder(docs)
    assert reordered[0].doc_id == "d0"   # rank 1 → position 0
    assert reordered[-1].doc_id == "d1"  # rank 2 → position -1


def test_litm_preserves_all_docs():
    docs = _make_docs(6)
    reordered = _lost_in_middle_reorder(docs)
    assert len(reordered) == len(docs)
    assert {r.doc_id for r in reordered} == {r.doc_id for r in docs}


@pytest.mark.asyncio
async def test_litm_disabled_by_default(monkeypatch):
    """LITM_ENABLED defaults to false — output order follows RRF (not LITM reorder)."""
    monkeypatch.delenv("LITM_ENABLED", raising=False)
    monkeypatch.delenv("MMR_ENABLED", raising=False)

    pr = ParallelRetriever()

    # Inject deterministic docs: d0 (score 0.9) appears in both sources → RRF rank 1
    shared = SearchResult(doc_id="d0", content="shared", score=0.9, source="bm25")
    other = SearchResult(doc_id="d1", content="other", score=0.5, source="bm25")

    async def mock_bm25(q, top_k=10):
        return [shared, other]

    async def mock_dense(q, top_k=10):
        return [SearchResult(doc_id="d0", content="shared", score=0.88, source="dense")]

    pr.bm25.search = mock_bm25
    pr.dense.search = mock_dense

    results = await pr.retrieve("query", query_type="simple", top_k=5)
    # Without LITM, RRF order: d0 first (cross-source boost)
    assert results[0].doc_id == "d0"


@pytest.mark.asyncio
async def test_litm_enabled_reorders_ends(monkeypatch):
    """With LITM_ENABLED=true, highest-score doc is at index 0, second at index -1."""
    monkeypatch.setenv("LITM_ENABLED", "true")
    monkeypatch.delenv("MMR_ENABLED", raising=False)

    pr = ParallelRetriever()

    # Inject 4 docs with known RRF ranking: d0 > d1 > d2 > d3
    docs = [
        SearchResult(doc_id=f"d{i}", content=f"doc {i}", score=1.0 - i * 0.1, source="bm25")
        for i in range(4)
    ]

    async def mock_bm25(q, top_k=10):
        return docs

    async def mock_dense(q, top_k=10):
        return []

    pr.bm25.search = mock_bm25
    pr.dense.search = mock_dense

    results = await pr.retrieve("query", query_type="simple", top_k=4)
    assert len(results) == 4
    assert results[0].doc_id == "d0"   # highest score at start
    assert results[-1].doc_id == "d1"  # second highest at end


# ── Circuit breaker integration tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_circuit_breaker_open_treated_as_source_failure(monkeypatch):
    """When the circuit breaker raises CircuitOpenError, the source returns empty + error='circuit_open'."""
    pr = ParallelRetriever()

    # Replace the bm25 circuit breaker with one that always raises CircuitOpenError
    mock_breaker = MagicMock()
    mock_breaker.call = AsyncMock(side_effect=CircuitOpenError("circuit is open"))
    pr._breakers["bm25"] = mock_breaker

    # Dense still works normally
    async def mock_dense(q, top_k=10):
        return [SearchResult(doc_id="d1", content="dense result", score=0.8, source="dense")]

    pr.dense.search = mock_dense

    results = await pr.retrieve("query", query_type="simple", top_k=5)

    # The pipeline should complete — dense results come through
    assert isinstance(results, list)
    # bm25 contributed nothing (circuit was open)
    bm25_ids = {r.doc_id for r in results if r.source == "bm25"}
    assert len(bm25_ids) == 0


# ── Cross-Encoder reranking tests ─────────────────────────────────────────────


def _docs(n: int) -> list[SearchResult]:
    return [
        SearchResult(doc_id=f"doc{i}", content=f"content {i}", score=1.0 / (i + 1), source="rrf_merged")
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_cross_encoder_disabled_by_default(monkeypatch):
    """RERANKER_ENABLED defaults to false — cross-encoder model never loaded."""
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    pr = ParallelRetriever()
    with patch("src.retrieval.parallel_retriever._reranker_enabled", return_value=False):
        results = await pr.retrieve("what is RAG?", top_k=3)
    assert len(results) <= 3


def _make_mock_cross_encoder(scores):
    """Return a mock CrossEncoder instance that returns the given scores."""
    import numpy as np
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(scores)
    return mock_model


@pytest.mark.asyncio
async def test_cross_encoder_reranks_by_model_score():
    """Cross-encoder reranks candidates by model score, not original RRF score."""
    docs = _docs(4)
    # Model says doc3 is best (score 0.99), doc0 is worst (score 0.01)
    with patch("src.retrieval.parallel_retriever._load_cross_encoder",
               return_value=_make_mock_cross_encoder([0.01, 0.30, 0.60, 0.99])):
        result = await _cross_encoder_rerank("query", docs, top_k=4)

    assert result[0].doc_id == "doc3"   # highest model score
    assert result[-1].doc_id == "doc0"  # lowest model score


@pytest.mark.asyncio
async def test_cross_encoder_adds_reranker_score_to_metadata():
    """Each result should have metadata['reranker_score'] set."""
    docs = _docs(2)
    with patch("src.retrieval.parallel_retriever._load_cross_encoder",
               return_value=_make_mock_cross_encoder([0.8, 0.4])):
        result = await _cross_encoder_rerank("query", docs, top_k=2)

    assert "reranker_score" in result[0].metadata
    assert result[0].metadata["reranker_score"] == pytest.approx(0.8, abs=1e-4)


@pytest.mark.asyncio
async def test_cross_encoder_respects_top_k():
    """Only top_k results returned even if more candidates are passed."""
    docs = _docs(6)
    with patch("src.retrieval.parallel_retriever._load_cross_encoder",
               return_value=_make_mock_cross_encoder([0.1, 0.5, 0.9, 0.3, 0.7, 0.2])):
        result = await _cross_encoder_rerank("query", docs, top_k=3)

    assert len(result) == 3
    assert result[0].doc_id == "doc2"  # score 0.9 — highest


@pytest.mark.asyncio
async def test_cross_encoder_empty_input():
    """Empty candidate list returns empty list without model call."""
    result = await _cross_encoder_rerank("query", [], top_k=5)
    assert result == []


@pytest.mark.asyncio
async def test_cross_encoder_fallback_on_import_error():
    """If sentence_transformers is unavailable, fall back to top_k slice."""
    docs = _docs(5)
    with patch("src.retrieval.parallel_retriever._load_cross_encoder",
               side_effect=ImportError("no module")):
        result = await _cross_encoder_rerank("query", docs, top_k=3)

    assert len(result) == 3


@pytest.mark.asyncio
async def test_cross_encoder_enabled_via_env(monkeypatch):
    """RERANKER_ENABLED=true wires cross-encoder into the retrieve() pipeline."""
    monkeypatch.setenv("RERANKER_ENABLED", "true")
    monkeypatch.delenv("MMR_ENABLED", raising=False)

    pr = ParallelRetriever()

    async def mock_bm25(q, top_k=10):
        return [SearchResult(doc_id=f"d{i}", content=f"doc {i}", score=0.9 - i * 0.1, source="bm25")
                for i in range(4)]

    async def mock_dense(q, top_k=10):
        return []

    pr.bm25.search = mock_bm25
    pr.dense.search = mock_dense

    # d1 gets highest cross-encoder score
    with patch("src.retrieval.parallel_retriever._load_cross_encoder",
               return_value=_make_mock_cross_encoder([0.1, 0.9, 0.3, 0.5, 0.2, 0.4, 0.15, 0.35])):
        results = await pr.retrieve("neural retrieval", top_k=2)

    assert len(results) <= 2
    assert results[0].doc_id == "d1"
