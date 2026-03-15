"""Unit tests for BM25Retriever."""
from __future__ import annotations

import pytest

from src.retrieval.bm25_retriever import SAMPLE_DOCS, BM25Retriever
from src.retrieval.models import SearchResult

TECH_DOCS = [
    {"id": "d1", "content": "LangGraph builds stateful multi-actor LLM applications."},
    {"id": "d2", "content": "BM25 is a sparse retrieval algorithm based on term frequency."},
    {"id": "d3", "content": "Dense retrieval encodes text into continuous vector spaces."},
    {"id": "d4", "content": "Reciprocal Rank Fusion merges ranked result lists from multiple sources."},
    {"id": "d5", "content": "Knowledge graphs represent entity relationships as nodes and edges."},
]


@pytest.fixture
def retriever() -> BM25Retriever:
    r = BM25Retriever()
    r.index(TECH_DOCS)
    return r


# ── index() tests ──────────────────────────────────────────────────────────────


def test_index_builds_corpus(retriever):
    assert retriever._bm25 is not None
    assert len(retriever._docs) == len(TECH_DOCS)


def test_unindexed_retriever_returns_empty():
    BM25Retriever()
    # No index() called


@pytest.mark.asyncio
async def test_unindexed_search_returns_empty():
    r = BM25Retriever()
    results = await r.search("LangGraph")
    assert results == []


# ── search() result tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_returns_results(retriever):
    results = await retriever.search("LangGraph stateful applications")
    assert len(results) > 0


@pytest.mark.asyncio
async def test_search_returns_search_result_objects(retriever):
    results = await retriever.search("BM25 retrieval")
    assert all(isinstance(r, SearchResult) for r in results)


@pytest.mark.asyncio
async def test_search_source_is_bm25(retriever):
    results = await retriever.search("dense retrieval vectors")
    assert all(r.source == "bm25" for r in results)


@pytest.mark.asyncio
async def test_search_scores_positive(retriever):
    results = await retriever.search("term frequency sparse")
    assert all(r.score > 0 for r in results)


@pytest.mark.asyncio
async def test_search_results_sorted_descending(retriever):
    results = await retriever.search("graphs nodes edges")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_empty_query_returns_empty(retriever):
    results = await retriever.search("")
    assert results == []


@pytest.mark.asyncio
async def test_whitespace_only_query_returns_empty(retriever):
    results = await retriever.search("   ")
    assert results == []


@pytest.mark.asyncio
async def test_top_k_limits_results(retriever):
    results = await retriever.search("retrieval information", top_k=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_top_k_one(retriever):
    results = await retriever.search("LangGraph", top_k=1)
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_relevant_doc_ranks_first(retriever):
    """doc about BM25 should rank first for BM25 query."""
    results = await retriever.search("BM25 term frequency sparse algorithm")
    assert len(results) > 0
    assert results[0].doc_id == "d2"


@pytest.mark.asyncio
async def test_doc_id_preserved(retriever):
    results = await retriever.search("LangGraph multi-actor")
    assert any(r.doc_id == "d1" for r in results)


# ── SAMPLE_DOCS smoke test ─────────────────────────────────────────────────────


def test_sample_docs_count():
    assert len(SAMPLE_DOCS) == 20


@pytest.mark.asyncio
async def test_sample_docs_searchable():
    r = BM25Retriever()
    r.index(SAMPLE_DOCS)
    results = await r.search("LangGraph stateful applications")
    assert len(results) > 0
    assert results[0].doc_id == "tech_01"
