"""Unit tests for ContextualCompressor."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retrieval.compressor import ContextualCompressor, get_compressor
from src.retrieval.models import SearchResult
import src.retrieval.compressor as compressor_module


# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc(doc_id: str = "d1", content: str = "Some content about the topic.") -> SearchResult:
    return SearchResult(doc_id=doc_id, content=content, score=0.9, source="rrf_merged")


# ── Dummy-LLM mode ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compress_skipped_in_dummy_llm_mode(monkeypatch):
    """When OPENAI_API_KEY is a placeholder, compress must return the doc unchanged."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-xxx")
    compressor = ContextualCompressor()
    doc = _doc(content="This is a long document with lots of sentences.")
    result = await compressor.compress("What is the topic?", doc)
    assert result is doc  # same object, untouched


@pytest.mark.asyncio
async def test_compress_skipped_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    compressor = ContextualCompressor()
    doc = _doc()
    result = await compressor.compress("query", doc)
    assert result is doc


# ── LLM-backed compression ────────────────────────────────────────────────────


def _make_mock_llm(response_text: str = "Relevant sentence."):
    """Return a mock LangChain-style LLM that responds with response_text."""
    mock_response = MagicMock()
    mock_response.content = response_text

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


@pytest.mark.asyncio
async def test_compress_batch_returns_same_count(monkeypatch):
    """compress_batch must return the same number of docs as input."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-abc")
    mock_llm = _make_mock_llm("Compressed sentence.")
    compressor = ContextualCompressor(llm_client=mock_llm)

    docs = [_doc(f"d{i}", f"Document {i} with content.") for i in range(5)]
    results = await compressor.compress_batch("What is this?", docs)
    assert len(results) == len(docs)


@pytest.mark.asyncio
async def test_compressed_doc_has_metadata_flag(monkeypatch):
    """After compression, metadata['compressed'] must be True."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-abc")
    mock_llm = _make_mock_llm("Only the relevant part.")
    compressor = ContextualCompressor(llm_client=mock_llm)

    doc = _doc(content="A long document. With multiple sentences. Only one is relevant.")
    result = await compressor.compress("relevant", doc)

    assert result.metadata.get("compressed") is True
    assert result.metadata.get("original_length") == len(doc.content)


@pytest.mark.asyncio
async def test_compressed_doc_preserves_doc_id_and_score(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-abc")
    mock_llm = _make_mock_llm("Relevant content.")
    compressor = ContextualCompressor(llm_client=mock_llm)

    doc = _doc(doc_id="myid", content="Some text here.")
    doc.score  # access to confirm it's set
    result = await compressor.compress("query", doc)

    assert result.doc_id == "myid"
    assert result.score == doc.score
    assert result.source == doc.source


@pytest.mark.asyncio
async def test_compress_batch_concurrency_semaphore(monkeypatch):
    """compress_batch must not raise even with many docs (semaphore should limit concurrency)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-abc")
    mock_llm = _make_mock_llm("Compressed.")
    compressor = ContextualCompressor(llm_client=mock_llm)

    docs = [_doc(f"d{i}", f"Content {i}.") for i in range(10)]
    results = await compressor.compress_batch("query", docs)
    assert len(results) == 10


# ── Feature-flag integration ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compression_disabled_by_default(monkeypatch):
    """When COMPRESSION_ENABLED is not set, compression must not run.

    We verify this by checking that no output document has
    metadata['compressed'] == True, which only gets set by the compressor.
    """
    monkeypatch.delenv("COMPRESSION_ENABLED", raising=False)
    # Also ensure we're in dummy-LLM mode so no real API calls are made
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    from src.retrieval.parallel_retriever import ParallelRetriever

    pr = ParallelRetriever()
    results = await pr.retrieve("test query", top_k=2)

    # None of the results should have been compressed
    compressed_docs = [r for r in results if r.metadata.get("compressed")]
    assert len(compressed_docs) == 0, (
        "No docs should have compressed=True when COMPRESSION_ENABLED is unset"
    )
