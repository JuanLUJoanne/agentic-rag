"""
Shared data models for the retrieval layer.

SearchResult is the common currency passed between every retriever,
the parallel merger, the cache, and the agent nodes. Keeping it in one
place prevents the field-drift that happens when each retriever defines
its own result type.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    doc_id: str
    content: str
    score: float
    source: str                       # "bm25" | "dense" | "graph" | "rrf_merged" | "cache"
    metadata: dict = field(default_factory=dict)
