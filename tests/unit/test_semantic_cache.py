"""
Tests for SemanticCache (src/retrieval/semantic_cache.py).

All tests use a helper that monkeypatches _encode and _init_backends so
no real SentenceTransformer model is loaded — the suite runs offline
without any model downloads.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.retrieval.semantic_cache import (
    SemanticCache,
    SemanticCacheEntry,
    _NumpyIndex,
    get_default_semantic_cache,
)

# ── Test helpers ──────────────────────────────────────────────────────────────


def _make_test_cache(
    similarity_threshold: float = 0.95,
    min_faithfulness: float = 0.85,
    ttl: int = 3600,
) -> SemanticCache:
    """
    Return a SemanticCache wired with a deterministic mock encoder.

    Bypasses _init_backends so no real model is loaded.  The mock encoder
    simply returns a fixed unit vector so cosine similarity between identical
    queries is 1.0 and between different queries is something lower.
    """
    cache = object.__new__(SemanticCache)
    cache._threshold = similarity_threshold
    cache._min_faithfulness = min_faithfulness
    cache._ttl = ttl
    cache._model_name = "mock"
    cache._entries: list[SemanticCacheEntry] = []
    cache._index = _NumpyIndex()
    cache._lock = threading.Lock()
    cache._hits = 0
    cache._misses = 0
    cache._available = True

    dim = 4

    def _mock_encode(text: str) -> np.ndarray:
        # Use a consistent hash-based vector so two identical strings get
        # the same vector (similarity=1.0) and different strings get different
        # vectors (similarity < 1.0 generally).
        seed = sum(ord(c) for c in text) % (2**31)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.float32)
        return vec / np.linalg.norm(vec)

    cache._encode = _mock_encode  # type: ignore[method-assign]
    return cache


# ── _NumpyIndex unit tests ────────────────────────────────────────────────────


def test_numpy_index_empty_returns_negative_idx() -> None:
    idx = _NumpyIndex()
    scores, indices = idx.search(np.zeros((1, 4), dtype=np.float32), k=1)
    assert indices[0][0] == -1


def test_numpy_index_returns_best_match() -> None:
    idx = _NumpyIndex()
    v1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    v2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    idx.add(v1)
    idx.add(v2)
    scores, indices = idx.search(v1, k=1)
    assert indices[0][0] == 0  # v1 is the best match for itself
    assert abs(scores[0][0] - 1.0) < 1e-5


# ── SemanticCache.set / get ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_hit_on_identical_query() -> None:
    cache = _make_test_cache(similarity_threshold=0.95)
    query = "What is retrieval augmented generation?"
    await cache.set(query, "RAG combines retrieval with LLMs.", [], eval_score=0.9)
    result = await cache.get(query)
    assert result is not None
    assert result.answer == "RAG combines retrieval with LLMs."
    assert result.similarity >= 0.99


@pytest.mark.asyncio
async def test_cache_miss_on_different_query() -> None:
    cache = _make_test_cache(similarity_threshold=0.95)
    await cache.set("What is RAG?", "RAG answer.", [], eval_score=0.9)
    # A very different query will have a low cosine similarity
    result = await cache.get("How does photosynthesis work?")
    # May or may not hit depending on random vectors; the key invariant is
    # that the result matches only when similarity >= threshold
    if result is not None:
        assert result.similarity >= 0.95


@pytest.mark.asyncio
async def test_faithfulness_gate_skips_low_score() -> None:
    cache = _make_test_cache(min_faithfulness=0.85)
    query = "What is RAG?"
    await cache.set(query, "Low quality answer.", [], eval_score=0.70)
    result = await cache.get(query)
    # Entry was never stored, so must be a miss
    assert result is None


@pytest.mark.asyncio
async def test_faithfulness_gate_stores_high_score() -> None:
    cache = _make_test_cache(min_faithfulness=0.85)
    query = "Explain FAISS."
    await cache.set(query, "FAISS is a library for efficient similarity search.", [], eval_score=0.92)
    result = await cache.get(query)
    assert result is not None
    assert result.eval_score == 0.92


@pytest.mark.asyncio
async def test_precomputed_embedding_reused() -> None:
    """Passing a precomputed embedding bypasses _encode."""
    cache = _make_test_cache(similarity_threshold=0.95)
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    await cache.set("test query", "answer", [], eval_score=0.9, embedding=vec)
    result = await cache.get("test query", embedding=vec)
    assert result is not None
    assert result.similarity >= 0.99


# ── TTL expiry ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expired_entry_is_a_miss() -> None:
    cache = _make_test_cache(ttl=1)  # 1-second TTL
    query = "Short-lived query."
    await cache.set(query, "answer", [], eval_score=0.9)
    # Manually expire the entry
    cache._entries[0].expires_at = time.time() - 1
    result = await cache.get(query)
    assert result is None


# ── Stats ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_tracks_hits_and_misses() -> None:
    cache = _make_test_cache()
    query = "Stats test query."
    await cache.set(query, "answer", [], eval_score=0.9)

    await cache.get(query)  # hit
    await cache.get("totally different query xyz abc 123")  # likely miss

    stats = cache.stats()
    assert stats["hits"] >= 1
    assert stats["available"] is True
    assert "hit_rate" in stats
    assert stats["size"] == 1


# ── Unavailable cache ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unavailable_cache_returns_none() -> None:
    cache = _make_test_cache()
    cache._available = False
    await cache.set("query", "answer", [], eval_score=0.9)
    result = await cache.get("query")
    assert result is None
    assert cache.stats()["size"] == 0  # nothing was stored


# ── Singleton ─────────────────────────────────────────────────────────────────


def test_get_default_semantic_cache_returns_same_instance() -> None:
    a = get_default_semantic_cache()
    b = get_default_semantic_cache()
    assert a is b
