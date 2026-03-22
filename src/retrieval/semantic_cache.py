"""
Semantic cache — approximate nearest-neighbour answer cache.

Sits between the exact-match QueryMemory and the full RAG pipeline:

    QueryMemory (exact string match)
        │ miss
        ▼
    SemanticCache (cosine similarity ≥ threshold)    ← this module
        │ miss
        ▼
    Full RAG pipeline (retrieve → grade → generate)

Why semantic caching?
    Exact-match caches miss when users rephrase the same intent.
    "What is RAG?" and "Can you explain retrieval augmented generation?"
    carry identical information needs.  Encoding both and comparing their
    L2-normalised embeddings with cosine similarity catches these cases,
    serving the cached answer at a fraction of the retrieval cost.

Index backends (in priority order)
    1. FAISS IndexFlatIP — fast ANN search; used when ``faiss-cpu`` is
       installed (``pip install -e ".[semantic]"``).
    2. _NumpyIndex       — pure-numpy brute-force inner product; zero extra
       deps beyond the existing sentence-transformers install.  Suitable for
       caches up to ~10 k entries; switch to FAISS beyond that.

Faithfulness gate:
    Only answers with eval_score ≥ min_faithfulness are stored, matching the
    quality bar used by QueryMemory (default 0.85).

TTL expiry:
    Each entry carries an expires_at timestamp.  Expired entries are treated
    as misses (lazy eviction, matching EmbeddingCache pattern).

Embedding reuse:
    The public API accepts an optional pre-computed numpy embedding so
    callers can skip re-encoding when the dense retriever has already
    produced the query vector.  The ``semantic_cache_check`` workflow node
    stores the embedding in AgentState so downstream retrieval nodes reuse it.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import structlog

logger = structlog.get_logger()


# ── Pure-numpy fallback index ───────────────────────────────────────────────


class _NumpyIndex:
    """
    Brute-force inner-product index implemented with numpy.

    Exposes the same ``add`` / ``search`` API as ``faiss.IndexFlatIP``
    so the rest of the code never needs to branch on which backend is active.

    Time complexity: O(n·d) per query.  Fine for ≤ 10 k entries on a
    modern CPU; switch to FAISS for larger caches.
    """

    def __init__(self) -> None:
        self._vecs: list[np.ndarray] = []
        self.ntotal: int = 0

    def add(self, matrix: np.ndarray) -> None:
        """Add a single row vector (shape ``(1, dim)``)."""
        self._vecs.append(matrix[0].copy())
        self.ntotal += 1

    def search(
        self, query_matrix: np.ndarray, k: int  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (scores, indices) for the top-1 nearest neighbour.

        ``k`` is accepted for API compatibility but only top-1 is returned;
        the semantic cache only needs the single best match.
        """
        if not self._vecs:
            return np.array([[0.0]], dtype=np.float32), np.array([[-1]])
        mat = np.stack(self._vecs)          # (n, dim)
        sims = mat @ query_matrix[0]        # (n,) — inner product = cosine for L2-normed vecs
        best = int(np.argmax(sims))
        return np.array([[float(sims[best])]], dtype=np.float32), np.array([[best]])


# ── Data containers ─────────────────────────────────────────────────────────


@dataclass
class SemanticCacheEntry:
    """Metadata stored alongside the FAISS/numpy vector for one cached answer."""

    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    eval_score: float = 0.0
    expires_at: float = 0.0  # Unix timestamp; 0 = never expires


@dataclass
class SemanticCacheResult:
    """Return value on a semantic cache hit."""

    query: str          # the stored query (may differ from the lookup query)
    answer: str
    citations: list[dict]
    eval_score: float
    similarity: float   # cosine similarity between lookup and stored query (0–1)


# ── SemanticCache ─────────────────────────────────────────────────────────────


class SemanticCache:
    """
    In-memory semantic answer cache using sentence-transformers + FAISS/numpy.

    Parameters
    ----------
    similarity_threshold:
        Minimum cosine similarity for a cache hit (default 0.95).
        0.95 is deliberately conservative — below this score, queries differ
        enough that surfacing the wrong cached answer would hurt more than help.
    min_faithfulness:
        Only cache answers whose eval_score meets this bar (default 0.85).
        Matches the quality gate used by QueryMemory.
    ttl:
        Entry lifetime in seconds (default 3 600 = 1 h).  Matches
        EmbeddingCache's default TTL for retrieval results.
    model_name:
        SentenceTransformer model for query encoding.  Default is
        ``all-MiniLM-L6-v2`` (80 MB, 384-dim, fast on CPU).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        min_faithfulness: float = 0.85,
        ttl: int = 3600,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._threshold = similarity_threshold
        self._min_faithfulness = min_faithfulness
        self._ttl = ttl
        self._model_name = model_name

        self._entries: list[SemanticCacheEntry] = []
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._model = None
        self._index: _NumpyIndex | object = _NumpyIndex()  # replaced by FAISS if available
        self._available = False
        self._init_backends()

    # ── Backend initialisation ──────────────────────────────────────────────

    def _init_backends(self) -> None:
        """
        Load the sentence-transformer model and choose the best index backend.

        Priority:
          1. FAISS IndexFlatIP (fast, scales to millions of entries)
          2. _NumpyIndex       (always-available, fine for ≤ 10 k entries)

        If sentence-transformers fails to load the model (e.g. network not
        available for download) the cache stays in pass-through mode.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_name)
            dim = model.get_sentence_embedding_dimension()

            # Prefer FAISS for fast search at scale
            try:
                import faiss

                self._index = faiss.IndexFlatIP(dim)
                backend = "faiss"
            except ImportError:
                self._index = _NumpyIndex()
                backend = "numpy"

            self._model = model
            self._available = True
            logger.info(
                "semantic_cache_initialized",
                model=self._model_name,
                dim=dim,
                backend=backend,
                threshold=self._threshold,
            )

        except Exception as exc:
            logger.warning(
                "semantic_cache_unavailable",
                reason=str(exc)[:120],
                fallback="pass_through",
            )

    # ── Encoding ────────────────────────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        """
        Return a L2-normalised float32 embedding for ``text``.

        Defined as an instance method (not a closure) so tests can monkeypatch
        it to inject deterministic embeddings without loading a real model.
        """
        emb: np.ndarray = self._model.encode([text], normalize_embeddings=True)
        return emb[0].astype(np.float32)

    # ── Synchronous inner operations (run inside asyncio.to_thread) ──────────

    def _get_sync(
        self, query: str, embedding: np.ndarray | None
    ) -> SemanticCacheResult | None:
        with self._lock:
            if self._index.ntotal == 0:
                self._misses += 1
                logger.debug("semantic_cache_miss", reason="empty_index", query=query[:60])
                return None

            q_emb: np.ndarray
            if embedding is not None:
                q_emb = np.array(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(q_emb))
                if norm > 0:
                    q_emb = q_emb / norm
            else:
                q_emb = self._encode(query)

            scores, indices = self._index.search(q_emb.reshape(1, -1), k=1)
            similarity = float(scores[0][0])
            idx = int(indices[0][0])

            if idx < 0 or similarity < self._threshold:
                self._misses += 1
                logger.debug(
                    "semantic_cache_miss",
                    reason="below_threshold",
                    similarity=round(similarity, 4),
                    query=query[:60],
                )
                return None

            entry = self._entries[idx]

            # Lazy TTL eviction: expired entry counts as a miss
            if time.time() > entry.expires_at:
                self._misses += 1
                logger.debug("semantic_cache_miss", reason="expired", query=query[:60])
                return None

            self._hits += 1
            logger.info(
                "semantic_cache_hit",
                similarity=round(similarity, 4),
                stored_query=entry.query[:60],
                query=query[:60],
            )
            return SemanticCacheResult(
                query=entry.query,
                answer=entry.answer,
                citations=entry.citations,
                eval_score=entry.eval_score,
                similarity=similarity,
            )

    def _set_sync(
        self,
        query: str,
        answer: str,
        citations: list[dict],
        eval_score: float,
        embedding: np.ndarray | None,
    ) -> None:
        emb: np.ndarray
        if embedding is not None:
            emb = np.array(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
        else:
            emb = self._encode(query)

        expires_at = time.time() + self._ttl

        with self._lock:
            self._index.add(emb.reshape(1, -1))
            self._entries.append(
                SemanticCacheEntry(
                    query=query,
                    answer=answer,
                    citations=citations,
                    eval_score=eval_score,
                    expires_at=expires_at,
                )
            )
        logger.info(
            "semantic_cache_set",
            query=query[:60],
            ttl=self._ttl,
            eval_score=eval_score,
            size=len(self._entries),
        )

    # ── Public async API ────────────────────────────────────────────────────

    async def get(
        self,
        query: str,
        embedding: np.ndarray | None = None,
    ) -> SemanticCacheResult | None:
        """
        Semantically look up ``query`` in the cache.

        Parameters
        ----------
        query:
            Raw query string; encoded internally if ``embedding`` is None.
        embedding:
            Optional pre-computed L2-normalised float32 numpy array.  When
            provided the encoding step is skipped (~10 ms saved on CPU).
            Pass the embedding produced by the dense retriever to avoid
            re-encoding the same query twice in one request.

        Returns
        -------
        ``SemanticCacheResult`` on a hit (similarity ≥ threshold, TTL valid);
        ``None`` on a miss or when the cache is unavailable.
        """
        if not self._available:
            return None
        return await asyncio.to_thread(self._get_sync, query, embedding)

    async def set(
        self,
        query: str,
        answer: str,
        citations: list[dict],
        eval_score: float,
        embedding: np.ndarray | None = None,
    ) -> None:
        """
        Store a query-answer pair if eval_score ≥ min_faithfulness.

        Silently skips low-quality answers — the quality gate ensures only
        trustworthy answers are surfaced for future similar queries.
        """
        if not self._available:
            return
        if eval_score < self._min_faithfulness:
            logger.debug(
                "semantic_cache_set_skipped",
                reason="faithfulness_below_threshold",
                eval_score=eval_score,
                threshold=self._min_faithfulness,
            )
            return
        await asyncio.to_thread(self._set_sync, query, answer, citations, eval_score, embedding)

    def stats(self) -> dict:
        """Return hit/miss counters, hit rate, current size, and availability."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
            "size": len(self._entries),
            "available": self._available,
            "threshold": self._threshold,
        }


# ── Module-level singleton ───────────────────────────────────────────────────

_default_semantic_cache: SemanticCache | None = None


def get_default_semantic_cache() -> SemanticCache:
    """Return the process-wide default SemanticCache instance (lazy init)."""
    global _default_semantic_cache
    if _default_semantic_cache is None:
        _default_semantic_cache = SemanticCache()
    return _default_semantic_cache
