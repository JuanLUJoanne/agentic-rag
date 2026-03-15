"""Unit tests for the human-in-the-loop review queue."""
from __future__ import annotations

import pytest

import src.api.human_review as hr_module
from src.api.human_review import (
    approve_item,
    get_pending_items,
    get_stats,
    reject_item,
    submit_for_review,
)


@pytest.fixture(autouse=True)
def reset_store():
    """Isolate each test with a fresh review store."""
    hr_module._reset_store()
    yield
    hr_module._reset_store()


# ── submit_for_review ──────────────────────────────────────────────────────────


def test_submit_creates_pending_item() -> None:
    review_id = submit_for_review("What is LangGraph?", "LangGraph is a...", 0.6)
    pending = get_pending_items()
    assert len(pending) == 1
    item = pending[0]
    assert item.id == review_id
    assert item.query == "What is LangGraph?"
    assert item.status == "pending"
    assert item.confidence == pytest.approx(0.6)


def test_submit_returns_unique_ids() -> None:
    id1 = submit_for_review("q1", "a1", 0.5)
    id2 = submit_for_review("q2", "a2", 0.5)
    assert id1 != id2


def test_submit_stores_reason() -> None:
    submit_for_review("q", "a", 0.4, reason="quality below threshold")
    item = get_pending_items()[0]
    assert item.reason == "quality below threshold"


def test_pending_only_returns_pending_items() -> None:
    """get_pending_items must exclude approved/rejected items."""
    id1 = submit_for_review("q1", "a1", 0.9)
    submit_for_review("q2", "a2", 0.9)
    reject_item(id1, reason="wrong")

    pending = get_pending_items()
    assert len(pending) == 1
    assert pending[0].query == "q2"


# ── approve_item ───────────────────────────────────────────────────────────────


async def test_approve_changes_status(monkeypatch) -> None:
    """Approved item must have status='approved'."""
    # Monkeypatch memory.learn to avoid touching the singleton
    from src.retrieval import memory as mem_module
    from src.retrieval.memory import QueryMemory

    fresh_mem = QueryMemory()
    monkeypatch.setattr(mem_module, "_default_memory", fresh_mem)

    review_id = submit_for_review("What is RAG?", "RAG is...", confidence=0.9)
    item = await approve_item(review_id)
    assert item.status == "approved"


async def test_approve_calls_memory_learn(monkeypatch) -> None:
    """Approve must store the answer in QueryMemory (high-confidence path)."""
    from src.retrieval import memory as mem_module
    from src.retrieval.memory import QueryMemory

    fresh_mem = QueryMemory(min_faithfulness=0.85)
    monkeypatch.setattr(mem_module, "_default_memory", fresh_mem)

    query = "What is vector search?"
    answer = "Vector search finds nearest neighbours..."
    review_id = submit_for_review(query, answer, confidence=0.9)
    await approve_item(review_id)

    result = await fresh_mem.recall(query)
    assert result is not None
    assert result.answer == answer


async def test_approve_not_stored_if_low_confidence(monkeypatch) -> None:
    """If confidence < min_faithfulness, memory.learn should skip storage."""
    from src.retrieval import memory as mem_module
    from src.retrieval.memory import QueryMemory

    fresh_mem = QueryMemory(min_faithfulness=0.85)
    monkeypatch.setattr(mem_module, "_default_memory", fresh_mem)

    review_id = submit_for_review("q", "a", confidence=0.5)
    await approve_item(review_id)

    result = await fresh_mem.recall("q")
    assert result is None  # not stored below threshold


async def test_approve_raises_on_unknown_id() -> None:
    with pytest.raises(KeyError):
        await approve_item("nonexistent-id")


# ── reject_item ────────────────────────────────────────────────────────────────


def test_reject_changes_status() -> None:
    review_id = submit_for_review("q", "a", 0.4)
    item = reject_item(review_id, reason="hallucinated fact")
    assert item.status == "rejected"
    assert item.reason == "hallucinated fact"


def test_reject_raises_on_unknown_id() -> None:
    with pytest.raises(KeyError):
        reject_item("bad-id")


# ── get_stats ──────────────────────────────────────────────────────────────────


async def test_stats_accurate(monkeypatch) -> None:
    """Stats should reflect the correct counts after mixed operations."""
    from src.retrieval import memory as mem_module
    from src.retrieval.memory import QueryMemory

    monkeypatch.setattr(mem_module, "_default_memory", QueryMemory())

    id1 = submit_for_review("q1", "a1", 0.9)
    id2 = submit_for_review("q2", "a2", 0.5)
    submit_for_review("q3", "a3", 0.3)

    await approve_item(id1)
    reject_item(id2, "bad")

    stats = get_stats()
    assert stats["total"] == 3
    assert stats["pending"] == 1
    assert stats["approved"] == 1
    assert stats["rejected"] == 1
    assert stats["approval_rate"] == pytest.approx(0.5)


def test_stats_empty_store() -> None:
    stats = get_stats()
    assert stats["total"] == 0
    assert stats["approval_rate"] == 0.0
