"""
Human-in-the-loop review queue.

When the quality agent scores an answer below the confidence threshold, the
supervisor routes to the human_review node, which calls ``submit_for_review``.
A human reviewer then inspects the item via the REST endpoints and either
approves or rejects it.

Approved items are stored in QueryMemory so future identical queries are
served instantly from cache.  Rejected items receive a reason string that
can feed back into prompt engineering.

This is an in-memory store (a plain list) — production would use a database
table and a WebSocket notification channel, but the interface is identical.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = structlog.get_logger()

# ── Data model ─────────────────────────────────────────────────────────────────


@dataclass
class ReviewItem:
    id: str
    query: str
    answer: str
    confidence: float
    reason: str
    status: str  # "pending" | "approved" | "rejected"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


# ── In-memory store ────────────────────────────────────────────────────────────

_store: list[ReviewItem] = []


def _reset_store() -> None:
    """Clear the store — used in tests to avoid cross-test contamination."""
    _store.clear()


def _find_item(review_id: str) -> ReviewItem:
    for item in _store:
        if item.id == review_id:
            return item
    raise KeyError(f"Review item not found: {review_id!r}")


# ── Public API (callable from workflow nodes) ──────────────────────────────────


def submit_for_review(
    query: str,
    answer: str,
    confidence: float,
    reason: str = "",
) -> str:
    """
    Add an answer to the review queue and return the review_id.

    Called by the human_review workflow node when answer quality falls below
    the confidence threshold.
    """
    review_id = str(uuid.uuid4())
    item = ReviewItem(
        id=review_id,
        query=query,
        answer=answer,
        confidence=confidence,
        reason=reason,
        status="pending",
    )
    _store.append(item)
    logger.info(
        "review_submitted",
        review_id=review_id,
        confidence=confidence,
        query=query[:80],
    )
    return review_id


async def approve_item(review_id: str) -> ReviewItem:
    """
    Approve a review item and store the answer in QueryMemory.

    Uses the human confidence score as the eval_score so that high-confidence
    human judgements meet the memory's min_faithfulness bar (0.85).
    """
    from src.retrieval.memory import get_default_memory

    item = _find_item(review_id)
    item.status = "approved"
    # Human approval → treat confidence as eval_score for memory storage
    await get_default_memory().learn(
        item.query, item.answer, [], item.confidence
    )
    logger.info("review_approved", review_id=review_id, confidence=item.confidence)
    return item


def reject_item(review_id: str, reason: str = "") -> ReviewItem:
    """Mark an item rejected with an optional reason for feedback."""
    item = _find_item(review_id)
    item.status = "rejected"
    item.reason = reason
    logger.info("review_rejected", review_id=review_id, reason=reason[:120])
    return item


def get_pending_items() -> list[ReviewItem]:
    return [i for i in _store if i.status == "pending"]


def get_stats() -> dict:
    total = len(_store)
    pending = sum(1 for i in _store if i.status == "pending")
    approved = sum(1 for i in _store if i.status == "approved")
    rejected = sum(1 for i in _store if i.status == "rejected")
    return {
        "total": total,
        "pending": pending,
        "approved": approved,
        "rejected": rejected,
        "approval_rate": approved / (approved + rejected) if (approved + rejected) else 0.0,
    }


# ── FastAPI router ─────────────────────────────────────────────────────────────

router = APIRouter(prefix="/review", tags=["human-review"])


class RejectRequest(BaseModel):
    reason: str = ""


def _item_to_dict(item: ReviewItem) -> dict:
    return {
        "id": item.id,
        "query": item.query,
        "answer": item.answer,
        "confidence": item.confidence,
        "reason": item.reason,
        "status": item.status,
        "created_at": item.created_at.isoformat(),
    }


@router.get("/pending")
async def list_pending() -> list[dict]:
    """Return all items currently awaiting review."""
    return [_item_to_dict(i) for i in get_pending_items()]


@router.post("/{review_id}/approve")
async def approve_endpoint(review_id: str) -> dict:
    """Approve an item and cache the answer in QueryMemory."""
    try:
        item = await approve_item(review_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _item_to_dict(item)


@router.post("/{review_id}/reject")
async def reject_endpoint(review_id: str, body: RejectRequest) -> dict:
    """Reject an item with an optional reason."""
    try:
        item = reject_item(review_id, reason=body.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _item_to_dict(item)


@router.get("/stats")
async def stats_endpoint() -> dict:
    return get_stats()
