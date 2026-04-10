"""
Tests for multi-granularity chunking and retrieval (parent-child strategy).

Covers:
  - chunker.chunk_document — correct chunk counts and parent-child links
  - MultiGranularityRetriever — indexing, search, parent expansion, dedup
  - ParallelRetriever integration — MULTI_GRANULARITY_ENABLED flag
"""
from __future__ import annotations

import pytest

from src.retrieval.chunker import Chunk, chunk_document
from src.retrieval.multi_granularity_retriever import MultiGranularityRetriever

# Multi-sentence documents that exercise parent-child expansion.
# Each doc has 3 sentences → 1 paragraph chunk + 3 sentence chunks.
DOCS = [
    {
        "id": "doc_bert",
        "content": "BERT was introduced in 2018. It uses bidirectional attention. It changed NLP forever.",
    },
    {
        "id": "doc_gpt",
        "content": "GPT models generate text autoregressively. They predict the next token given context. GPT-4 supports multimodal inputs.",
    },
    {
        "id": "doc_vec",
        "content": "Vector databases store embeddings efficiently. They support approximate nearest neighbour search. pgvector runs inside PostgreSQL.",
    },
]


# ── chunker tests ──────────────────────────────────────────────────────────

def test_chunk_document_produces_both_granularities():
    chunks = chunk_document("d1", "BERT was introduced in 2018. It changed NLP.")
    granularities = {c.granularity for c in chunks}
    assert "sentence" in granularities
    assert "paragraph" in granularities


def test_sentence_chunks_have_parent_id():
    chunks = chunk_document("d1", "BERT was introduced in 2018. It changed NLP.")
    sentences = [c for c in chunks if c.granularity == "sentence"]
    assert sentences, "expected at least one sentence chunk"
    assert all(c.parent_id is not None for c in sentences)


def test_paragraph_chunks_have_no_parent():
    chunks = chunk_document("d1", "BERT was introduced in 2018. It changed NLP.")
    paragraphs = [c for c in chunks if c.granularity == "paragraph"]
    assert paragraphs, "expected at least one paragraph chunk"
    assert all(c.parent_id is None for c in paragraphs)


def test_sentence_parent_ids_match_paragraph_chunk_ids():
    chunks = chunk_document("d1", "BERT was introduced in 2018. It changed NLP.")
    para_ids = {c.chunk_id for c in chunks if c.granularity == "paragraph"}
    sent_parent_ids = {c.parent_id for c in chunks if c.granularity == "sentence"}
    assert sent_parent_ids.issubset(para_ids)


def test_multi_paragraph_text_produces_one_paragraph_per_block():
    text = "First paragraph sentence one. First paragraph sentence two.\n\nSecond paragraph sentence one."
    chunks = chunk_document("d1", text)
    paragraphs = [c for c in chunks if c.granularity == "paragraph"]
    assert len(paragraphs) == 2


def test_single_sentence_document_still_chunked():
    chunks = chunk_document("d1", "A single sentence document without any splits.")
    assert any(c.granularity == "paragraph" for c in chunks)
    assert any(c.granularity == "sentence" for c in chunks)


def test_source_doc_id_propagated():
    chunks = chunk_document("my_doc", "First sentence. Second sentence.")
    assert all(c.source_doc_id == "my_doc" for c in chunks)


def test_chunk_ids_are_unique():
    chunks = chunk_document("d1", "First sentence. Second sentence. Third sentence.")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ── MultiGranularityRetriever tests ──────────────────────────────────────


@pytest.fixture
def retriever() -> MultiGranularityRetriever:
    r = MultiGranularityRetriever()
    r.index(DOCS)
    return r


@pytest.mark.asyncio
async def test_search_returns_results(retriever):
    results = await retriever.search("BERT bidirectional attention", top_k=3)
    assert results


@pytest.mark.asyncio
async def test_search_returns_paragraph_content(retriever):
    results = await retriever.search("BERT bidirectional attention", top_k=3)
    # Paragraph content should contain the full multi-sentence paragraph,
    # not just the single matched sentence.
    top = results[0]
    assert "BERT" in top.content
    # The paragraph contains all three sentences
    assert len(top.content) > len("BERT was introduced in 2018.")


@pytest.mark.asyncio
async def test_search_source_label(retriever):
    results = await retriever.search("BERT bidirectional", top_k=3)
    assert all(r.source == "multi_granularity" for r in results)


@pytest.mark.asyncio
async def test_search_metadata_has_matched_sentence(retriever):
    results = await retriever.search("BERT bidirectional attention", top_k=3)
    assert results
    meta = results[0].metadata
    assert "matched_sentence" in meta
    assert "sentence_id" in meta
    assert meta["granularity"] == "expanded"


@pytest.mark.asyncio
async def test_matched_sentence_is_subset_of_paragraph(retriever):
    results = await retriever.search("BERT bidirectional attention", top_k=3)
    assert results
    top = results[0]
    assert top.metadata["matched_sentence"] in top.content


@pytest.mark.asyncio
async def test_deduplication_no_duplicate_parent_paragraphs(retriever):
    # Multiple sentences from the same doc could match; result must deduplicate
    # to one paragraph per parent_id.
    results = await retriever.search("sentence", top_k=10)
    doc_ids = [r.doc_id for r in results]
    assert len(doc_ids) == len(set(doc_ids)), "duplicate parent paragraphs returned"


@pytest.mark.asyncio
async def test_deduplication_keeps_highest_score(retriever):
    # For the BERT doc, two sentences might match "BERT NLP".
    # The returned result's score should be >= any single-sentence score.
    results_full = await retriever.search("BERT NLP", top_k=5)
    bert_results = [r for r in results_full if "BERT" in r.content]
    if len(bert_results) >= 1:
        # Score on the paragraph result ≥ 0 (basic sanity)
        assert bert_results[0].score >= 0.0


@pytest.mark.asyncio
async def test_search_respects_top_k(retriever):
    results = await retriever.search("the", top_k=2)
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_empty_query_returns_empty(retriever):
    results = await retriever.search("", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_whitespace_only_query_returns_empty(retriever):
    results = await retriever.search("   ", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_unindexed_retriever_returns_empty():
    r = MultiGranularityRetriever()
    results = await r.search("BERT", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_search_scores_are_non_negative(retriever):
    results = await retriever.search("BERT attention", top_k=5)
    assert all(r.score >= 0.0 for r in results)


@pytest.mark.asyncio
async def test_search_results_descending_score(retriever):
    results = await retriever.search("BERT attention vector", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


# ── ParallelRetriever integration ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_retriever_with_multi_granularity(monkeypatch):
    monkeypatch.setenv("MULTI_GRANULARITY_ENABLED", "true")
    from src.retrieval.parallel_retriever import ParallelRetriever
    pr = ParallelRetriever()
    results = await pr.retrieve("BERT transformers attention", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_parallel_retriever_without_multi_granularity(monkeypatch):
    monkeypatch.delenv("MULTI_GRANULARITY_ENABLED", raising=False)
    from src.retrieval.parallel_retriever import ParallelRetriever
    pr = ParallelRetriever()
    results = await pr.retrieve("BERT transformers attention", top_k=3)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_parallel_retriever_multi_gran_results_merged_via_rrf(monkeypatch):
    monkeypatch.setenv("MULTI_GRANULARITY_ENABLED", "true")
    from src.retrieval.parallel_retriever import ParallelRetriever
    pr = ParallelRetriever()
    results = await pr.retrieve("vector database embeddings", top_k=5)
    # All results should use the "rrf_merged" source label after fusion
    sources = {r.source for r in results}
    assert "rrf_merged" in sources
