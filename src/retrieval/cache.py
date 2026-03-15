"""
Embedding cache — Memory pattern (retrieval results).

Caches retrieved SearchResult lists keyed by sha256(query) in SQLite.
Avoids re-running the full parallel retrieval pipeline for repeated or
near-identical queries within the TTL window.

A single shared sqlite3 connection with a threading.Lock lets us call
the sync SQLite API safely from async code via asyncio.to_thread without
creating per-call in-memory databases.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import sqlite3
import threading
import time

import structlog

from src.retrieval.models import SearchResult

logger = structlog.get_logger()


class EmbeddingCache:
    """
    SQLite-backed cache for retrieval results.

    TTL-based expiry: expired entries are deleted on the next get() for
    that key (lazy eviction), avoiding a background sweep thread.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        # check_same_thread=False + explicit lock allows safe multi-thread use
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    key        TEXT PRIMARY KEY,
                    query      TEXT NOT NULL,
                    results    TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            self._conn.commit()

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def _get_sync(self, key: str) -> str | None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT results, expires_at FROM embedding_cache WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            if row[1] <= now:
                # Lazy eviction of expired entry
                self._conn.execute("DELETE FROM embedding_cache WHERE key = ?", (key,))
                self._conn.commit()
                return None
            return row[0]

    def _set_sync(self, key: str, query: str, serialized: str, expires_at: float) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                (key, query, results, created_at, expires_at) VALUES (?, ?, ?, ?, ?)
                """,
                (key, query, serialized, now, expires_at),
            )
            self._conn.commit()

    async def get(self, query: str) -> list[SearchResult] | None:
        """Return cached results if present and unexpired, else None."""
        key = self._key(query)
        raw = await asyncio.to_thread(self._get_sync, key)
        if raw is not None:
            self._hits += 1
            logger.debug("embedding_cache_hit", query=query[:60])
            return [SearchResult(**r) for r in json.loads(raw)]
        self._misses += 1
        logger.debug("embedding_cache_miss", query=query[:60])
        return None

    async def set(self, query: str, results: list[SearchResult], ttl: int = 3600) -> None:
        """Store results under sha256(query), expiring after ttl seconds."""
        key = self._key(query)
        serialized = json.dumps([dataclasses.asdict(r) for r in results])
        expires_at = time.time() + ttl
        await asyncio.to_thread(self._set_sync, key, query, serialized, expires_at)
        logger.debug("embedding_cache_set", query=query[:60], ttl=ttl)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }


# Module-level default instance — shared by retriever.py nodes.
# Uses :memory: so tests start clean; production sets EMBEDDING_CACHE_DB env var.
_default_cache = EmbeddingCache(db_path=os.getenv("EMBEDDING_CACHE_DB", ":memory:"))


def get_default_cache() -> EmbeddingCache:
    """Return the process-wide default EmbeddingCache instance."""
    return _default_cache
