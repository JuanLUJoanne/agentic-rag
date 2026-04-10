"""
Multi-granularity retriever — parent-child strategy.

Why parent-child instead of independent sentence + paragraph indexes?
  - Searching only the sentence index gives precision (small, focused chunks
    match queries accurately).
  - Expanding hits to their parent paragraph gives the LLM the context it
    needs to reason correctly (a single "It changed NLP." sentence is
    ambiguous without its surrounding paragraph).
  - One sentence BM25 index + one dict lookup is cheaper than maintaining
    two independent BM25 indexes with a separate score-normalisation step.

Query-time flow::

    sentence BM25 search(query, top_k*3)
         │
         ▼
    expand each hit: sentence_id → parent_id → paragraph content (dict lookup)
         │
         ▼
    deduplicate: multiple sentences from the same paragraph → keep highest score
         │
         ▼
    return top_k SearchResult(content=paragraph_text, metadata={matched_sentence})

Score normalisation:
    Raw BM25 sentence scores are used only for deduplication ordering within
    the multi-granularity source.  When this retriever is plugged into
    ParallelRetriever alongside BM25 and Dense, RRF uses only rank positions
    — no cross-source score calibration needed.

Parallel latency:
    MultiGranularityRetriever.search() is an async method with the same
    signature as BM25Retriever.search(), so it slots directly into the
    asyncio.gather() call in ParallelRetriever._timed_search().  The sentence
    BM25 search is ~1 ms for corpora up to ~10k docs; the parent expansion is
    O(top_k) dict lookups.  Net added latency when run in parallel: ~0 ms.
"""
from __future__ import annotations

import structlog

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.chunker import Chunk, chunk_document
from src.retrieval.models import SearchResult

logger = structlog.get_logger()


class MultiGranularityRetriever:
    """
    Indexes sentence chunks; returns parent-paragraph content at query time.

    Drop-in compatible with :class:`BM25Retriever` — implements the same
    ``index(documents)`` / ``async search(query, top_k)`` interface.

    Usage::

        retriever = MultiGranularityRetriever()
        retriever.index([{"id": "d1", "content": "BERT was introduced..."}, ...])
        results = await retriever.search("BERT attention", top_k=5)
        # results[i].content            → full paragraph text (LLM context)
        # results[i].metadata["matched_sentence"] → sentence that matched
        # results[i].metadata["sentence_id"]      → sentence chunk_id
    """

    def __init__(self) -> None:
        self._sentence_index = BM25Retriever()
        # paragraph chunk_id → paragraph text; populated at index time
        self._paragraph_store: dict[str, str] = {}
        self._indexed = False

    def index(self, documents: list[dict]) -> None:
        """
        Chunk *documents* into sentence/paragraph pairs and build the sentence
        BM25 index.  The paragraph store is held in memory for O(1) expansion.

        *documents* must be a list of ``{"id": str, "content": str}`` dicts —
        the same schema as :meth:`BM25Retriever.index`.
        """
        sentence_docs: list[dict] = []

        for doc in documents:
            chunks: list[Chunk] = chunk_document(doc["id"], doc["content"])
            for chunk in chunks:
                if chunk.granularity == "paragraph":
                    self._paragraph_store[chunk.chunk_id] = chunk.content
                else:
                    sentence_docs.append(
                        {
                            "id": chunk.chunk_id,
                            "content": chunk.content,
                            "metadata": {"parent_id": chunk.parent_id},
                        }
                    )

        self._sentence_index.index(sentence_docs)
        self._indexed = True
        logger.debug(
            "multi_granularity_indexed",
            doc_count=len(documents),
            sentence_chunks=len(sentence_docs),
            paragraph_count=len(self._paragraph_store),
        )

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrieve sentences, expand to parent paragraphs, deduplicate.

        Returns at most *top_k* :class:`SearchResult` objects whose
        ``content`` is the parent paragraph text and whose ``score`` is the
        highest BM25 sentence score within that paragraph.
        """
        if not self._indexed or not query.strip():
            return []

        # Fetch 3× candidates so dedup still leaves enough distinct paragraphs
        sentence_hits = await self._sentence_index.search(query, top_k=top_k * 3)

        # Expand each sentence hit to its parent paragraph.
        # For each parent_id keep only the hit with the highest sentence score
        # (that score becomes the paragraph's relevance signal for RRF upstream).
        best_by_parent: dict[str, tuple[float, SearchResult]] = {}

        for hit in sentence_hits:
            parent_id: str | None = hit.metadata.get("parent_id")
            if parent_id is None or parent_id not in self._paragraph_store:
                continue

            existing_score, _ = best_by_parent.get(parent_id, (float("-inf"), None))
            if hit.score > existing_score:
                para_content = self._paragraph_store[parent_id]
                best_by_parent[parent_id] = (
                    hit.score,
                    SearchResult(
                        doc_id=parent_id,
                        content=para_content,
                        score=round(hit.score, 6),
                        source="multi_granularity",
                        metadata={
                            "matched_sentence": hit.content,
                            "sentence_id": hit.doc_id,
                            "granularity": "expanded",
                        },
                    ),
                )

        results = sorted(
            (r for _, r in best_by_parent.values()),
            key=lambda r: r.score,
            reverse=True,
        )
        return results[:top_k]
