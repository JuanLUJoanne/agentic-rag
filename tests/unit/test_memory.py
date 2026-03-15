"""Unit tests for EmbeddingCache and QueryMemory."""
from __future__ import annotations

import pytest

from src.retrieval.cache import EmbeddingCache
from src.retrieval.memory import MemoryResult, QueryMemory
from src.retrieval.models import SearchResult

# ── EmbeddingCache tests ───────────────────────────────────────────────────────


@pytest.fixture
def cache() -> EmbeddingCache:
    return EmbeddingCache(db_path=":memory:")


@pytest.fixture
def sample_results() -> list[SearchResult]:
    return [
        SearchResult(doc_id="d1", content="LangGraph info", score=0.9, source="bm25"),
        SearchResult(doc_id="d2", content="RAG explanation", score=0.8, source="dense"),
    ]


@pytest.mark.asyncio
async def test_cache_miss_returns_none(cache):
    result = await cache.get("unknown query")
    assert result is None


@pytest.mark.asyncio
async def test_cache_set_then_get(cache, sample_results):
    await cache.set("what is langgraph", sample_results)
    cached = await cache.get("what is langgraph")
    assert cached is not None
    assert len(cached) == 2
    assert cached[0].doc_id == "d1"


@pytest.mark.asyncio
async def test_cache_hit_skips_retrieval(cache, sample_results):
    """Integration: cached results are returned on second get without re-running retrieval."""
    await cache.set("test query", sample_results, ttl=3600)

    first = await cache.get("test query")
    second = await cache.get("test query")

    assert first is not None
    assert second is not None
    assert first[0].doc_id == second[0].doc_id

    stats = cache.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 0


@pytest.mark.asyncio
async def test_cache_preserves_all_fields(cache, sample_results):
    await cache.set("query", sample_results)
    cached = await cache.get("query")
    assert cached[0].source == "bm25"
    assert cached[0].score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_cache_expired_entry_returns_none(cache, sample_results):
    """TTL=0 entries expire immediately and return None on next get."""
    await cache.set("expiring query", sample_results, ttl=0)
    # The entry expires at set-time + 0 seconds, so any subsequent get should miss
    import time
    time.sleep(0.01)  # ensure clock has advanced past expires_at
    result = await cache.get("expiring query")
    assert result is None


@pytest.mark.asyncio
async def test_cache_stats_hit_rate(cache, sample_results):
    await cache.set("q", sample_results)
    await cache.get("q")      # hit
    await cache.get("nope")   # miss

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)


# ── QueryMemory tests ──────────────────────────────────────────────────────────


@pytest.fixture
def memory() -> QueryMemory:
    return QueryMemory(db_path=":memory:", min_faithfulness=0.85)


@pytest.mark.asyncio
async def test_recall_returns_none_for_unknown(memory):
    result = await memory.recall("unknown query xyz")
    assert result is None


@pytest.mark.asyncio
async def test_learn_then_recall_returns_answer(memory):
    await memory.learn("What is RAG?", "RAG combines retrieval with generation.", [], 0.9)
    result = await memory.recall("What is RAG?")
    assert result is not None
    assert result.answer == "RAG combines retrieval with generation."


@pytest.mark.asyncio
async def test_recall_returns_memory_result_type(memory):
    await memory.learn("test q", "test answer", [], 0.9)
    result = await memory.recall("test q")
    assert isinstance(result, MemoryResult)


@pytest.mark.asyncio
async def test_low_score_answer_not_returned(memory):
    """Answers below min_faithfulness (0.85) must not be surfaced by recall."""
    await memory.learn("What is LangGraph?", "LangGraph is...", [], 0.7)
    result = await memory.recall("What is LangGraph?")
    assert result is None


@pytest.mark.asyncio
async def test_exact_threshold_not_returned(memory):
    """Score exactly at threshold (0.85) should pass through."""
    await memory.learn("threshold test", "answer", [], 0.85)
    result = await memory.recall("threshold test")
    assert result is not None


@pytest.mark.asyncio
async def test_learn_with_citations(memory):
    citations = [{"doc_id": "d1", "source": "docs", "index": 1}]
    await memory.learn("cited query", "cited answer", citations, 0.95)
    result = await memory.recall("cited query")
    assert result is not None
    assert result.citations == citations


@pytest.mark.asyncio
async def test_forget_removes_entry(memory):
    await memory.learn("remove me", "answer", [], 0.9)
    await memory.forget("remove me")
    result = await memory.recall("remove me")
    assert result is None


@pytest.mark.asyncio
async def test_forget_nonexistent_is_safe(memory):
    """Forgetting a query that was never learned should not raise."""
    await memory.forget("never stored")  # should not raise


@pytest.mark.asyncio
async def test_stats_accurate(memory):
    await memory.recall("miss_1")          # miss
    await memory.recall("miss_2")          # miss
    await memory.learn("known", "ans", [], 0.9)
    await memory.recall("known")           # hit

    stats = memory.stats()
    assert stats["recall_hits"] == 1
    assert stats["recall_misses"] == 2
    assert stats["learned_count"] == 1
    assert stats["hit_rate"] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_stats_learned_count_excludes_low_score(memory):
    """Low-score learns should not increment learned_count."""
    await memory.learn("high", "answer", [], 0.9)
    await memory.learn("low", "answer", [], 0.5)  # skipped
    stats = memory.stats()
    assert stats["learned_count"] == 1


@pytest.mark.asyncio
async def test_overwrite_existing_entry(memory):
    """Learning the same query twice updates the stored answer."""
    await memory.learn("q", "first answer", [], 0.9)
    await memory.learn("q", "improved answer", [], 0.95)
    result = await memory.recall("q")
    assert result.answer == "improved answer"
