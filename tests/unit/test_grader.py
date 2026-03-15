"""Unit tests for the relevance grader."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.relevance_grader import _aggregate_status, grade_documents

# ── _aggregate_status pure-function tests ─────────────────────────────────────


def test_aggregate_all_relevant():
    assert _aggregate_status([0.9, 0.8, 0.7]) == "all_relevant"


def test_aggregate_none_empty():
    assert _aggregate_status([]) == "none"


def test_aggregate_none_all_low():
    assert _aggregate_status([0.2, 0.3]) == "none"


def test_aggregate_partial():
    assert _aggregate_status([0.9, 0.2]) == "partial"


def test_aggregate_boundary_exactly_half():
    # Score of exactly 0.5 counts as relevant
    assert _aggregate_status([0.5, 0.4]) == "partial"


# ── grade_documents node tests (DummyLLM path) ────────────────────────────────


@pytest.mark.asyncio
async def test_all_docs_relevant_with_dummy_llm(base_state):
    result = await grade_documents(base_state)
    assert result["docs_relevant"] == "all_relevant"
    assert len(result["retrieved_docs"]) == 2
    assert "agent_trace" in result


@pytest.mark.asyncio
async def test_no_docs_returns_none():
    state = {
        "query": "test query",
        "retrieved_docs": [],
        "agent_trace": [],
    }
    result = await grade_documents(state)
    assert result["docs_relevant"] == "none"
    assert result["retrieved_docs"] == []
    assert result["relevance_scores"] == []


@pytest.mark.asyncio
async def test_partial_docs_filtered(base_state):
    """When one doc is irrelevant, status is partial and filtered list is shorter."""

    async def mock_grade(llm, query, doc):
        # Make doc_2 irrelevant
        return (False, 0.1) if doc["id"] == "doc_2" else (True, 0.9)

    with patch("src.agents.relevance_grader._grade_single_doc", side_effect=mock_grade):
        result = await grade_documents(base_state)

    assert result["docs_relevant"] == "partial"
    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["id"] == "doc_1"


@pytest.mark.asyncio
async def test_all_irrelevant_returns_none(base_state):
    """When every doc is irrelevant, the node should return none."""

    async def mock_grade(llm, query, doc):
        return False, 0.1

    with patch("src.agents.relevance_grader._grade_single_doc", side_effect=mock_grade):
        result = await grade_documents(base_state)

    assert result["docs_relevant"] == "none"
    assert result["retrieved_docs"] == []


@pytest.mark.asyncio
async def test_relevance_scores_returned(base_state):
    result = await grade_documents(base_state)
    # DummyLLM returns 0.8 for every doc
    assert len(result["relevance_scores"]) == len(base_state["retrieved_docs"])
    for score in result["relevance_scores"]:
        assert 0.0 <= score <= 1.0
