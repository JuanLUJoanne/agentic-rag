"""Unit tests for the answer generator."""
from __future__ import annotations

import pytest

from src.agents.generator import _extract_citations, generate

# ── _extract_citations pure-function tests ────────────────────────────────────


def test_extract_citations_two_refs():
    docs = [
        {"id": "d1", "source": "source_1"},
        {"id": "d2", "source": "source_2"},
    ]
    answer = "LangGraph is great [Doc 1]. It uses state machines [Doc 2]."
    citations = _extract_citations(answer, docs)
    assert len(citations) == 2
    ids = {c["doc_id"] for c in citations}
    assert "d1" in ids
    assert "d2" in ids


def test_extract_citations_no_refs():
    docs = [{"id": "d1", "source": "s1"}]
    answer = "Some answer with no citation markers."
    assert _extract_citations(answer, docs) == []


def test_extract_citations_out_of_range():
    docs = [{"id": "d1", "source": "s1"}]
    # [Doc 5] references a non-existent document
    answer = "Answer [Doc 5]."
    assert _extract_citations(answer, docs) == []


def test_extract_citations_deduplicates():
    docs = [{"id": "d1", "source": "s1"}]
    answer = "Point one [Doc 1]. Point two [Doc 1]."
    citations = _extract_citations(answer, docs)
    assert len(citations) == 1


def test_extract_citations_preserves_index():
    docs = [{"id": "d1", "source": "s1"}, {"id": "d2", "source": "s2"}]
    answer = "Answer [Doc 2]."
    citations = _extract_citations(answer, docs)
    assert citations[0]["index"] == 2
    assert citations[0]["doc_id"] == "d2"


# ── generate node tests (DummyLLM path) ───────────────────────────────────────


@pytest.mark.asyncio
async def test_generates_answer(base_state):
    result = await generate(base_state)
    assert result["generation"]
    assert "agent_trace" in result


@pytest.mark.asyncio
async def test_dummy_llm_returns_placeholder(base_state):
    """DummyLLM path must produce a placeholder containing the query."""
    result = await generate(base_state)
    assert "Placeholder answer for:" in result["generation"]
    assert base_state["query"] in result["generation"]


@pytest.mark.asyncio
async def test_citations_included(base_state):
    result = await generate(base_state)
    assert "citations" in result
    assert isinstance(result["citations"], list)


@pytest.mark.asyncio
async def test_dummy_llm_attaches_first_two_docs_as_citations(base_state):
    """With DummyLLM, citations reference the first two docs."""
    result = await generate(base_state)
    assert len(result["citations"]) == min(2, len(base_state["retrieved_docs"]))


@pytest.mark.asyncio
async def test_no_docs_returns_placeholder():
    state = {
        "query": "What is RAG?",
        "retrieved_docs": [],
        "agent_trace": [],
    }
    result = await generate(state)
    assert "Placeholder answer for:" in result["generation"]
    assert result["citations"] == []


@pytest.mark.asyncio
async def test_agent_trace_populated(base_state):
    result = await generate(base_state)
    assert len(result["agent_trace"]) == 1
    assert result["agent_trace"][0]["node"] == "generate"
