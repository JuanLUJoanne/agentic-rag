"""
Prompt versioning store backed by SQLite.

Agents call ``store.get("relevance_grader")`` at runtime instead of
hardcoding prompt strings. This enables rollbacks, A/B prompt experiments,
and audit trails without code changes.

Usage example::

    store = get_prompt_store()

    # Save a new version
    v1 = store.save("relevance_grader", "Is the document relevant? Answer yes/no.")
    v2 = store.save("relevance_grader", "Rate relevance 1-5. Document: {doc}")

    # Get latest
    prompt = store.get("relevance_grader")

    # Get a specific version
    old_prompt = store.get("relevance_grader", version=1)

    # Rollback to v1 (creates a new version with v1's template)
    store.rollback("relevance_grader", version=1)
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from typing import TypedDict

import structlog

logger = structlog.get_logger()

_DEFAULT_DB = ":memory:"


class PromptVersion(TypedDict):
    version: int
    template: str
    created_at: float
    description: str


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prompt_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    template    TEXT    NOT NULL,
    description TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL,
    UNIQUE (name, version)
)
"""

_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_prompt_name ON prompt_versions (name)"


class PromptStore:
    """
    Thread-safe, SQLite-backed prompt version store.

    Version numbers are auto-incremented per prompt name (i.e., the second
    save of "foo" is version 2, regardless of other prompt names).
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or os.getenv("PROMPT_STORE_DB", _DEFAULT_DB)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_INDEX_SQL)
            self._conn.commit()

    # ── Write operations ────────────────────────────────────────────────────

    def save(self, name: str, template: str, description: str = "") -> int:
        """
        Save a new version of the named prompt.

        Returns the new version number (1-indexed, auto-incremented per name).
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM prompt_versions WHERE name = ?",
                (name,),
            )
            max_version: int = cur.fetchone()[0]
            new_version = max_version + 1
            self._conn.execute(
                "INSERT INTO prompt_versions (name, version, template, description, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (name, new_version, template, description, time.time()),
            )
            self._conn.commit()
        logger.info("prompt_saved", name=name, version=new_version)
        return new_version

    def rollback(self, name: str, version: int) -> PromptVersion:
        """
        Copy an old version as the new latest version.

        Returns the newly created PromptVersion.
        Raises KeyError if the target version does not exist.
        """
        old = self.get(name, version=version)
        if old is None:
            raise KeyError(f"Prompt '{name}' version {version} not found")
        new_version = self.save(
            name,
            old["template"],
            description=f"Rollback to v{version}: {old['description']}".strip(": "),
        )
        result = self.get(name, version=new_version)
        assert result is not None  # just saved it
        logger.info("prompt_rolled_back", name=name, from_version=version, new_version=new_version)
        return result

    # ── Read operations ─────────────────────────────────────────────────────

    def get(self, name: str, version: int | None = None) -> PromptVersion | None:
        """
        Return a specific version or the latest if version is None.

        Returns None if the prompt name (or version) does not exist.
        """
        with self._lock:
            if version is None:
                cur = self._conn.execute(
                    "SELECT version, template, created_at, description "
                    "FROM prompt_versions WHERE name = ? ORDER BY version DESC LIMIT 1",
                    (name,),
                )
            else:
                cur = self._conn.execute(
                    "SELECT version, template, created_at, description "
                    "FROM prompt_versions WHERE name = ? AND version = ?",
                    (name, version),
                )
            row = cur.fetchone()
        if row is None:
            return None
        return PromptVersion(
            version=row["version"],
            template=row["template"],
            created_at=row["created_at"],
            description=row["description"],
        )

    def list_versions(self, name: str) -> list[PromptVersion]:
        """Return all versions for a prompt name, ordered by version ascending."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT version, template, created_at, description "
                "FROM prompt_versions WHERE name = ? ORDER BY version ASC",
                (name,),
            )
            rows = cur.fetchall()
        return [
            PromptVersion(
                version=r["version"],
                template=r["template"],
                created_at=r["created_at"],
                description=r["description"],
            )
            for r in rows
        ]


# ── Module-level singleton ──────────────────────────────────────────────────

_prompt_store: PromptStore | None = None


def get_prompt_store() -> PromptStore:
    global _prompt_store
    if _prompt_store is None:
        _prompt_store = PromptStore()
    return _prompt_store
