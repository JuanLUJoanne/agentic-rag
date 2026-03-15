"""
Query memory — Memory pattern (answer-level learning).

Stores high-quality answers so identical queries are served instantly
from memory instead of running the full retrieval + generation pipeline.

Only answers that exceeded `min_faithfulness` (default 0.85) are
stored — low-quality answers are not worth caching and might mislead
future users asking the same question.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()


@dataclass
class MemoryResult:
    query: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    eval_score: float = 0.0


class QueryMemory:
    """
    SQLite-backed query-answer store.

    recall() is a strict lookup — only exact query strings match.
    Fuzzy/semantic matching is deferred to a later batch that embeds
    stored queries and finds nearest neighbours at recall time.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        min_faithfulness: float = 0.85,
    ) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._min_faithfulness = min_faithfulness
        self._recall_hits = 0
        self._recall_misses = 0
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_memory (
                    query      TEXT PRIMARY KEY,
                    answer     TEXT NOT NULL,
                    citations  TEXT NOT NULL DEFAULT '[]',
                    eval_score REAL NOT NULL,
                    stored_at  REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    # ── Sync helpers (run inside asyncio.to_thread) ────────────────────────────

    def _recall_sync(self, query: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT answer, citations, eval_score FROM query_memory WHERE query = ?",
                (query,),
            ).fetchone()
        if row is None:
            return None
        answer, citations_json, eval_score = row
        if eval_score < self._min_faithfulness:
            return None  # stored below threshold — don't surface it
        return {"answer": answer, "citations": json.loads(citations_json), "eval_score": eval_score}

    def _learn_sync(
        self, query: str, answer: str, citations: list[dict], eval_score: float
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO query_memory
                (query, answer, citations, eval_score, stored_at) VALUES (?, ?, ?, ?, ?)
                """,
                (query, answer, json.dumps(citations), eval_score, time.time()),
            )
            self._conn.commit()

    def _forget_sync(self, query: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM query_memory WHERE query = ?", (query,))
            self._conn.commit()

    def _count_sync(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM query_memory").fetchone()[0]

    # ── Public async API ───────────────────────────────────────────────────────

    async def recall(self, query: str) -> MemoryResult | None:
        """
        Return a cached MemoryResult if the exact query was previously
        answered with eval_score >= min_faithfulness, else None.
        """
        data = await asyncio.to_thread(self._recall_sync, query)
        if data is not None:
            self._recall_hits += 1
            logger.info("memory_hit", query=query[:80], score=data["eval_score"])
            return MemoryResult(
                query=query,
                answer=data["answer"],
                citations=data["citations"],
                eval_score=data["eval_score"],
            )
        self._recall_misses += 1
        logger.info("memory_miss", query=query[:80])
        return None

    async def learn(
        self,
        query: str,
        answer: str,
        citations: list[dict],
        eval_score: float,
    ) -> None:
        """
        Store a query-answer pair if eval_score meets the faithfulness bar.

        Silently skips low-quality answers — callers should not need to
        pre-filter; this is the quality gate.
        """
        if eval_score < self._min_faithfulness:
            logger.debug(
                "memory_learn_skipped",
                reason="score_below_threshold",
                score=eval_score,
                threshold=self._min_faithfulness,
            )
            return
        await asyncio.to_thread(self._learn_sync, query, answer, citations, eval_score)
        logger.info("memory_learned", query=query[:80], score=eval_score)

    async def forget(self, query: str) -> None:
        """Remove a stored query-answer pair."""
        await asyncio.to_thread(self._forget_sync, query)
        logger.info("memory_forgotten", query=query[:80])

    def stats(self) -> dict:
        learned_count = self._count_sync()
        total = self._recall_hits + self._recall_misses
        return {
            "recall_hits": self._recall_hits,
            "recall_misses": self._recall_misses,
            "hit_rate": self._recall_hits / total if total else 0.0,
            "learned_count": learned_count,
        }


# Module-level default instance used by workflow nodes
_default_memory = QueryMemory(
    db_path=os.getenv("QUERY_MEMORY_DB", ":memory:"),
    min_faithfulness=0.85,
)


def get_default_memory() -> QueryMemory:
    """Return the process-wide default QueryMemory instance."""
    return _default_memory
