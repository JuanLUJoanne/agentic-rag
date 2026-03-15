"""
BM25 sparse retriever backed by rank_bm25.

BM25 excels at exact-keyword and rare-term queries where dense embeddings
underperform. Running it in parallel with dense retrieval and merging via
RRF recovers documents that either method would miss on its own.
"""
from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi

from src.retrieval.models import SearchResult

logger = structlog.get_logger()

# 20 sample documents covering core ML/infra topics.
# ParallelRetriever indexes these at construction time so the system works
# end-to-end without a database connection in Batch 1-3 tests.
SAMPLE_DOCS: list[dict] = [
    {"id": "tech_01", "content": "LangGraph is a library for building stateful, multi-actor applications with large language models using a graph-based execution model."},
    {"id": "tech_02", "content": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model to produce grounded, citation-backed answers."},
    {"id": "tech_03", "content": "Corrective RAG adds a relevance grading step after retrieval. If documents are irrelevant the query is rewritten and retrieval is retried."},
    {"id": "tech_04", "content": "Vector databases store high-dimensional embeddings and support approximate nearest-neighbour search for semantic retrieval."},
    {"id": "tech_05", "content": "pgvector is a PostgreSQL extension that adds vector similarity search. It supports cosine, L2, and inner-product distance functions."},
    {"id": "tech_06", "content": "BM25 is a probabilistic sparse retrieval algorithm that ranks documents by term frequency and inverse document frequency."},
    {"id": "tech_07", "content": "Dense retrieval encodes queries and documents into continuous vector spaces using bi-encoder models like BERT or E5."},
    {"id": "tech_08", "content": "Reciprocal Rank Fusion (RRF) merges ranked lists from multiple retrieval sources by summing 1/(k+rank) scores for each document."},
    {"id": "tech_09", "content": "Knowledge graphs represent entities and their relationships as nodes and edges, enabling multi-hop reasoning queries."},
    {"id": "tech_10", "content": "Neo4j is a graph database that supports Cypher query language for traversing entity relationship networks."},
    {"id": "tech_11", "content": "Transformer models use self-attention mechanisms to build context-aware representations of input sequences."},
    {"id": "tech_12", "content": "BERT (Bidirectional Encoder Representations from Transformers) pre-trains on masked language modelling for rich contextual embeddings."},
    {"id": "tech_13", "content": "GPT models use autoregressive language modelling to generate coherent text token by token."},
    {"id": "tech_14", "content": "Fine-tuning adapts a pre-trained language model to a specific domain by continuing training on domain data."},
    {"id": "tech_15", "content": "LoRA (Low-Rank Adaptation) enables efficient fine-tuning by inserting trainable low-rank matrices into frozen model weights."},
    {"id": "tech_16", "content": "RLHF (Reinforcement Learning from Human Feedback) aligns language model outputs with human preferences using reward modelling."},
    {"id": "tech_17", "content": "LangChain provides composable building blocks for LLM applications: chains, prompts, memory, and agent toolkits."},
    {"id": "tech_18", "content": "Embeddings are dense vector representations of text that capture semantic similarity — similar texts have small cosine distance."},
    {"id": "tech_19", "content": "Self-RAG introduces self-reflection tokens that let the model decide when to retrieve and how to critique its own output."},
    {"id": "tech_20", "content": "Contrastive learning trains embedding models by pulling similar pairs together and pushing dissimilar pairs apart in vector space."},
]


class BM25Retriever:
    """
    BM25 sparse retriever.

    Call index() once with your document corpus before searching.
    The corpus is kept in memory; for large corpora Batch 5 adds an
    on-disk index backed by a serialised BM25Okapi object.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._docs: list[dict] = []

    def index(self, documents: list[dict]) -> None:
        """Build the BM25 index from a list of {'id', 'content', ...} dicts."""
        self._docs = documents
        tokenized = [doc["content"].lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("bm25_indexed", doc_count=len(documents))

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Return up to top_k results sorted by BM25 score descending."""
        if not self._bm25 or not query.strip():
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in ranked:
            if score > 0:
                doc = self._docs[idx]
                results.append(
                    SearchResult(
                        doc_id=doc["id"],
                        content=doc["content"],
                        score=float(score),
                        source="bm25",
                        metadata=doc.get("metadata", {}),
                    )
                )
        return results
