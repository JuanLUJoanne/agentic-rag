"""
Retrieval strategy benchmark — BM25 vs Dense vs RRF.

Evaluates three strategies against eval_data/qa_100.jsonl using
keyword overlap and MRR@5 as proxy relevance signals.

Run:
    source .venv/bin/activate
    python scripts/benchmark_retrieval.py
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.parallel_retriever import ParallelRetriever

# ── Synthetic corpus: 200 short docs covering eval topics ─────────────────────

CORPUS: list[dict] = [
    {"id": f"doc_{i:03d}", "content": c}
    for i, c in enumerate([
        "BM25 is a probabilistic sparse retrieval algorithm ranking documents by term frequency and inverse document frequency.",
        "Dense retrieval encodes queries and documents into continuous vector spaces using neural bi-encoder models.",
        "Reciprocal Rank Fusion merges ranked lists from multiple sources by summing 1/(k+rank) scores for each document.",
        "LangGraph is a library for building stateful multi-actor applications with large language models as directed graphs.",
        "RAG combines a retrieval system with a language model to produce grounded citation-backed answers.",
        "Corrective RAG adds relevance grading after retrieval; irrelevant documents trigger query rewriting and retry.",
        "FAISS is a library for efficient similarity search over dense vectors, supporting exact and approximate indexes.",
        "pgvector is a PostgreSQL extension adding vector similarity search with cosine, L2, and inner-product distances.",
        "LoRA inserts trainable low-rank matrices into frozen pretrained model weights for efficient fine-tuning.",
        "QLoRA combines 4-bit NF4 quantization with LoRA adapters, enabling large model fine-tuning on consumer hardware.",
        "RLHF trains a reward model on human preference comparisons then uses PPO to optimize the LLM policy.",
        "DPO directly optimizes LLMs on preference pairs using binary cross-entropy, eliminating the reward model.",
        "BERT is a bidirectional encoder pretrained on masked language modelling for rich contextual embeddings.",
        "GPT is a decoder-only autoregressive transformer trained to predict the next token, excelling at generation.",
        "Self-attention computes weighted sums of value vectors based on query-key dot products for context-aware representations.",
        "A knowledge graph represents entities and relationships as nodes and edges for multi-hop reasoning queries.",
        "Neo4j is a graph database using Cypher query language for traversing entity relationship networks.",
        "LangChain provides composable building blocks for LLM applications: chains, prompts, retrieval, and memory.",
        "Contrastive learning trains embeddings by pulling similar pairs together and pushing dissimilar pairs apart.",
        "Self-RAG embeds reflection tokens so the model decides when to retrieve and whether its output is grounded.",
        "TF-IDF scores terms by frequency in a document weighted by rarity across the corpus.",
        "BM25 improves TF-IDF with document length normalization and term frequency saturation.",
        "A bi-encoder independently encodes query and document into dense vectors for fast pre-computable retrieval.",
        "A cross-encoder jointly encodes query and document for high accuracy reranking but cannot pre-compute.",
        "Hybrid retrieval combines sparse BM25 and dense semantic search, typically merged with RRF for improved recall.",
        "Semantic search finds conceptually related documents even when query and document vocabulary differ.",
        "An embedding is a dense vector representation capturing semantic meaning; similar texts have high cosine similarity.",
        "Vector databases store high-dimensional embeddings and support approximate nearest-neighbour search.",
        "Chain-of-thought prompting asks LLMs to produce intermediate reasoning steps before the final answer.",
        "Hallucination occurs when an LLM generates plausible-sounding but factually incorrect statements.",
        "Faithfulness measures the fraction of answer claims supported by retrieved context.",
        "RAGAS evaluates RAG pipelines on faithfulness, answer relevancy, context precision, and context recall.",
        "Query decomposition breaks complex questions into simpler sub-queries for independent retrieval.",
        "Approximate nearest neighbour search trades small accuracy loss for dramatically faster vector search.",
        "HNSW builds a hierarchical graph for sub-linear ANN search with tunable recall-speed trade-off.",
        "A semantic cache stores query-answer pairs and serves cached answers for cosine-similar new queries.",
        "TTL-based cache expiry removes stale entries after a configured time window, ensuring freshness.",
        "A faithfulness gate prevents low-quality answers from being cached in the semantic cache.",
        "Rate limiting enforces RPM and TPM constraints using a token-bucket algorithm to prevent API quota errors.",
        "An audit log records queries, answers, cost, and agents used in an append-only JSONL file.",
        "Sentence-BERT fine-tunes BERT with siamese networks and contrastive loss for sentence-level embeddings.",
        "The all-MiniLM-L6-v2 model is a compact sentence-transformers model balancing speed and embedding quality.",
        "Hard negative mining selects BM25 high-overlap irrelevant documents to improve contrastive embedding training.",
        "MultipleNegativesRankingLoss is a contrastive objective treating all other batch items as negatives.",
        "A token bucket accumulates capacity at a fixed rate and allows bursts up to a maximum size.",
        "PII (personally identifiable information) includes emails, phone numbers, credit cards, and government IDs.",
        "Presidio is a Microsoft open-source library for PII detection using NER and pattern matching.",
        "Redaction replaces detected PII with typed placeholders like [EMAIL_1] before sending to an LLM.",
        "CostGuardrail enforces per-request, per-query, anomaly-detection, and total budget spending limits.",
        "Decimal arithmetic avoids floating-point drift when accumulating many small cost values.",
        "A supervisor in a multi-agent system routes tasks to specialist agents based on a skills registry.",
        "An AgentRegistry maps agent names to their declared skills, enabling dynamic supervisor dispatch.",
        "Human-in-the-loop workflows queue low-confidence answers for human review before returning to users.",
        "LangGraph MemorySaver checkpoints state at each node, enabling human-in-the-loop interrupts and replay.",
        "Input sanitization detects prompt injection patterns before a query enters the RAG pipeline.",
        "Prompt injection attempts to override system instructions by embedding adversarial text in user input.",
        "Context precision is the fraction of retrieved documents that are relevant to the question.",
        "Context recall is the fraction of relevant concepts covered by the retrieved document pool.",
        "Answer relevancy measures how well the generated answer addresses the original question.",
        "MRR (Mean Reciprocal Rank) averages 1/rank of the first relevant result across queries.",
        "Recall@5 measures how many relevant documents appear in the top 5 retrieved results.",
        "RRF k=60 follows the Cormack et al. 2009 paper and balances rank equality versus top-rank reward.",
        "IVF (Inverted File Index) partitions the vector space into clusters for faster approximate search.",
        "Product quantization compresses vectors into compact codes, reducing memory for large-scale indexes.",
        "Sparse vectors have most entries equal to zero; dense vectors have all entries non-zero.",
        "Cosine similarity measures the angle between two vectors, ranging from -1 (opposite) to 1 (identical).",
        "Dot product is the unnormalized inner product; equals cosine similarity for L2-normalized vectors.",
        "Temperature controls token sampling randomness; temperature 0 is greedy deterministic decoding.",
        "Tokenization splits text into subword units using BPE, WordPiece, or SentencePiece algorithms.",
        "BPE (Byte-Pair Encoding) iteratively merges the most frequent symbol pairs to build a vocabulary.",
        "WordPiece tokenization is used by BERT; it splits words into subwords to handle out-of-vocabulary terms.",
        "Fine-tuning adapts pretrained model weights to a domain by continuing training on domain data.",
        "Supervised Fine-Tuning (SFT) trains models on input-output demonstrations before preference alignment.",
        "DPO loss is a binary cross-entropy that increases probability of chosen response over rejected.",
        "LoRA rank r controls adapter expressiveness; higher r captures more complex transformations.",
        "lora_alpha scales LoRA updates; effective learning rate for adapters is lora_alpha / r.",
        "Target modules for LoRA are typically query and value projection layers in attention heads.",
        "bitsandbytes provides 4-bit and 8-bit quantization for PyTorch models on NVIDIA GPUs.",
        "accelerate handles distributed training and mixed precision for HuggingFace models.",
        "peft (Parameter-Efficient Fine-Tuning) is the HuggingFace library providing LoRA and other adapters.",
        "trl (Transformer Reinforcement Learning) provides SFTTrainer and DPOTrainer for LLM alignment.",
        "SFTTrainer wraps a causal LLM with a dataset and a loss function for supervised fine-tuning.",
        "DPOTrainer takes a base model and a reference model to compute the implicit reward signal.",
        "A reference model (frozen copy) is needed in DPO to normalize the log-probability ratio.",
        "Retrieval latency is the wall-clock time from query to returned ranked document list.",
        "Index build time is one-time cost; query time matters for production SLAs.",
        "Re-ranking with a cross-encoder improves precision@1 at the cost of higher per-query latency.",
        "A retrieval pipeline typically runs: embed query → ANN search → re-rank → return top-k.",
        "Chroma is an open-source embedding database with a Python API for local and hosted deployments.",
        "Pinecone is a managed vector database supporting real-time upserts and hybrid sparse-dense search.",
        "Weaviate supports hybrid BM25+dense search natively with GraphQL and REST APIs.",
        "Elasticsearch supports both BM25 keyword search and dense vector kNN search in the same index.",
        "OpenSearch is an open-source Elasticsearch fork supporting vector search with HNSW indexes.",
        "Milvus is a cloud-native vector database supporting multiple index types and distributed search.",
        "BERT-based cross-encoders are commonly used as rerankers because they understand query-document interaction.",
        "E5 (EmbEddings from bidirEctional Encoder rEpresentations) is a strong bi-encoder for dense retrieval.",
        "BGE (BAAI General Embedding) models achieve state-of-the-art scores on MTEB retrieval benchmarks.",
        "MTEB (Massive Text Embedding Benchmark) evaluates embedding models across 56 tasks and 112 languages.",
        "Sentence-transformers is a Python library providing pre-trained and fine-tunable sentence embedding models.",
        "An in-batch negative strategy treats all other batch examples as negatives in contrastive training.",
        "Hard negatives improve training efficiency: 1000 hard negatives provide more signal than 10000 easy ones.",
        "Data augmentation for retrieval includes paraphrase generation, back-translation, and LLM-based question synthesis.",
        "Knowledge distillation trains a small student model to mimic a larger teacher's output distribution.",
        "FlashAttention implements memory-efficient exact attention using IO-aware tiling for long sequences.",
        "Grouped Query Attention (GQA) reduces KV-cache memory by sharing keys/values across attention head groups.",
        "Sliding window attention limits attention span to a local window, enabling longer context at lower cost.",
        "RAG can use metadata filters alongside vector similarity to narrow retrieval to relevant document subsets.",
        "Hybrid search in vector databases combines dense ANN results with BM25 scores using RRF or weighted sum.",
        "A vector index must be rebuilt or updated when new documents are added to the corpus.",
        "Incremental indexing adds new document embeddings without rebuilding the entire index from scratch.",
        "Passage retrieval splits long documents into overlapping chunks before embedding for more precise retrieval.",
        "Chunking strategy (chunk size, overlap) significantly affects retrieval quality in RAG systems.",
        "Parent-document retrieval embeds small chunks but returns parent document context for richer generation.",
        "HyDE (Hypothetical Document Embedding) generates a hypothetical answer and embeds it as the query vector.",
        "Query expansion adds synonyms or related terms to improve BM25 sparse retrieval recall.",
        "Multi-query retrieval generates multiple paraphrased versions of a question and merges the result sets.",
        "Step-back prompting asks the LLM to generate a higher-level question before answering the specific one.",
        "Contextual compression retrieves broad documents and then extracts only the relevant passage for generation.",
        "FLARE (Forward-Looking Active REtrieval) proactively retrieves when the model's confidence drops below a threshold.",
        "Iterative RAG alternates between generation and retrieval, using partially generated answers to refine queries.",
        "GraphRAG (Microsoft) builds a community-level knowledge graph to enable global summarization over large corpora.",
        "Agentic RAG uses autonomous agents that decide when to retrieve, rewrite, and validate iteratively.",
        "An LLM judge evaluates generated answers against criteria like correctness, completeness, and faithfulness.",
        "G-Eval uses a chain-of-thought rubric filled by an LLM to produce structured evaluation scores.",
        "ROUGE measures n-gram overlap between generated and reference text for summarization evaluation.",
        "BLEU measures n-gram precision with brevity penalty for machine translation and generation evaluation.",
        "BERTScore uses contextual BERT embeddings to compute precision, recall, and F1 between generated and reference.",
        "A/B testing compares two retrieval or generation strategies on live traffic to measure real-world impact.",
        "Shadow mode runs a new pipeline alongside the production pipeline without serving its outputs.",
        "Feature flags allow gradual rollout of pipeline changes to a fraction of traffic.",
        "Observability for RAG includes latency histograms, per-source retrieval counts, and answer quality scores.",
        "structlog provides structured JSON logging that integrates with log aggregation and alerting systems.",
        "FastAPI provides async HTTP endpoints with automatic OpenAPI spec generation for Python services.",
        "Server-Sent Events (SSE) allow a server to push incremental events to a client over HTTP.",
        "LangGraph astream yields incremental node updates enabling SSE streaming of pipeline progress.",
        "A thread_id in LangGraph's MemorySaver allows isolating conversation state across concurrent requests.",
        "The StateGraph compile step validates edge consistency and produces an executable runnable graph.",
        "Conditional edges in LangGraph route to different nodes based on a function applied to current state.",
        "An operator.add reducer accumulates list values across node updates rather than overwriting the state.",
        "A TypedDict schema for agent state enables type-safe access and LangGraph checkpoint serialization.",
        "SQLite with check_same_thread=False and an explicit threading.Lock allows safe concurrent async access.",
        "asyncio.to_thread runs blocking synchronous code in a thread pool without blocking the event loop.",
        "asyncio.gather runs multiple coroutines concurrently, collecting results when all complete.",
        "asyncio.wait_for wraps a coroutine with a timeout, raising TimeoutError if it does not complete in time.",
        "A singleton pattern with lazy initialization ensures expensive resources are created only once per process.",
        "Environment variable injection allows switching between DummyLLM and real OpenAI without code changes.",
        "pytest-asyncio provides async test support for Python, enabling testing of coroutines with asyncio.run.",
        "Ruff is a fast Python linter and formatter checking for style, imports, and common code issues.",
        "pytest fixtures provide reusable test setup and teardown, injecting dependencies into test functions.",
        "Monkeypatching in tests replaces production objects with deterministic stubs for offline testing.",
        "A graceful degradation pattern tries optional dependencies and falls back silently on ImportError.",
        "Optional extras in pyproject.toml allow users to install only the dependencies they need.",
        "A JSONL (JSON Lines) file stores one JSON object per line for streaming and incremental reads.",
        "Append-only files grow monotonically and can be tailed in real time for live monitoring.",
        "A hash function like SHA-256 produces a fixed-length digest for deterministic cache key generation.",
        "Lazy eviction checks TTL at lookup time rather than scheduling background sweeps.",
        "A token bucket fills at a constant rate and allows bursts up to the bucket capacity.",
        "Exponential backoff retries failed requests with increasing delays to reduce load during failures.",
        "A dead letter queue stores failed messages for later inspection without blocking the main pipeline.",
        "Circuit breaker pattern stops sending requests to a failing downstream service after repeated errors.",
        "Idempotency ensures repeated requests produce the same result, enabling safe retries.",
        "A webhook delivers push notifications to a registered URL when an event occurs.",
        "gRPC provides high-performance RPC over HTTP/2 with Protocol Buffer serialization.",
        "OpenAPI (Swagger) documents REST API endpoints, request/response schemas, and authentication requirements.",
        "Pydantic validates Python data models with type annotations, raising errors for invalid inputs.",
        "Dependency injection passes collaborators as constructor arguments rather than instantiating them internally.",
        "The repository pattern abstracts data storage behind an interface, decoupling business logic from databases.",
        "Event sourcing stores state changes as an immutable log of events rather than mutable records.",
        "CQRS (Command Query Responsibility Segregation) separates read and write models for independent scaling.",
        "A semantic router uses embedding similarity to classify intents and route to appropriate handlers.",
        "Function calling allows LLMs to output structured JSON that triggers external tool invocations.",
        "Tool use in agents passes available tools as JSON schemas in the prompt; the LLM selects and parameterizes them.",
        "ReAct (Reasoning and Acting) interleaves chain-of-thought reasoning with tool-use actions in a loop.",
        "Reflexion uses verbal self-reflection and episodic memory to let agents learn from past failures.",
        "MRKL (Modular Reasoning, Knowledge, and Language) routes sub-problems to specialized expert modules.",
        "An agent trajectory is the sequence of (thought, action, observation) triples from a single episode.",
        "BM25 retriever requires tokenized corpus at index time and produces ranked results in milliseconds.",
        "Dense retriever requires encoded corpus vectors at index time and GPU or CPU vector operations at query time.",
        "Graph retriever traverses entity relationships at query time, scaling with graph degree not corpus size.",
        "A multi-hop query requires chaining at least two retrieval steps or relationship traversals to answer.",
        "Retrieval augmented generation reduces hallucination by grounding generation in retrieved evidence.",
        "An embedding model converts text to a fixed-size float vector; dimensionality ranges from 64 to 4096.",
        "Sentence-level embeddings capture the meaning of an entire sentence better than word-level embeddings.",
        "Pooling strategies (mean, max, CLS token) aggregate token embeddings into a sentence embedding.",
        "Normalization to unit length ensures cosine similarity equals inner product for fast FAISS search.",
        "Batch encoding encodes many texts at once on GPU for higher throughput than per-text encoding.",
        "A recall-precision trade-off: higher k in top-k retrieval improves recall but increases noise for generation.",
        "Chunk overlap in passage retrieval ensures boundary context is not lost between adjacent chunks.",
        "BM25 is sensitive to query length; very short queries may return no results if no terms score above zero.",
        "Dense retrieval handles short queries well because embeddings capture meaning beyond individual terms.",
        "An async RAG pipeline can overlap retrieval and generation for reduced end-to-end latency.",
    ])
]

# ── Helpers ────────────────────────────────────────────────────────────────────

QA_PATH = Path(__file__).parent.parent / "eval_data" / "qa_100.jsonl"
OUT_PATH = Path(__file__).parent.parent / "eval_results" / "benchmark_retrieval.json"
TOP_K = 5


def _load_queries() -> list[dict]:
    return [json.loads(ln) for ln in QA_PATH.read_text().splitlines() if ln.strip()]


def _keyword_overlap(docs_text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = docs_text.lower()
    return sum(1 for kw in keywords if kw.lower() in lower) / len(keywords)


def _mrr(docs, keywords: list[str]) -> float:
    for rank, doc in enumerate(docs, 1):
        if any(kw.lower() in doc.content.lower() for kw in keywords):
            return 1.0 / rank
    return 0.0


async def _bench(name: str, search_fn, queries: list[dict]) -> dict:
    latencies, overlaps, mrrs = [], [], []
    for q in queries:
        t0 = time.monotonic()
        docs = await search_fn(q["question"])
        latencies.append((time.monotonic() - t0) * 1000)
        combined = " ".join(d.content for d in docs)
        overlaps.append(_keyword_overlap(combined, q.get("relevant_keywords", [])))
        mrrs.append(_mrr(docs, q.get("relevant_keywords", [])))
    return {
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "keyword_overlap_at_5": round(sum(overlaps) / len(overlaps), 3),
        "mrr": round(sum(mrrs) / len(mrrs), 3),
    }


async def main() -> None:
    queries = _load_queries()
    print(f"Loaded {len(queries)} queries, corpus size {len(CORPUS)}")

    # BM25
    bm25 = BM25Retriever()
    bm25.index(CORPUS)
    bm25_metrics = await _bench("bm25", lambda q: bm25.search(q, TOP_K), queries)
    print(f"BM25:    {bm25_metrics}")

    # Dense (mock — always returns same 5 fixed docs)
    dense = DenseRetriever()
    dense_metrics = await _bench("dense", lambda q: dense.search(q, TOP_K), queries)
    print(f"Dense:   {dense_metrics}  [mock — fixed docs]")

    # RRF (parallel: BM25 re-indexed + mock Dense + mock Graph)
    parallel = ParallelRetriever()
    parallel.bm25.index(CORPUS)  # override default SAMPLE_DOCS with full corpus
    rrf_metrics = await _bench("rrf", lambda q: parallel.retrieve(q, "simple", TOP_K), queries)
    print(f"RRF:     {rrf_metrics}")

    best_single_mrr = max(bm25_metrics["mrr"], dense_metrics["mrr"])
    rrf_gain = round((rrf_metrics["mrr"] - best_single_mrr) / max(best_single_mrr, 1e-9) * 100, 1)
    rrf_overhead = round(rrf_metrics["avg_latency_ms"] / max(bm25_metrics["avg_latency_ms"], 1e-9), 2)

    report = {
        "benchmark": "retrieval_comparison",
        "dataset": str(QA_PATH.relative_to(QA_PATH.parent.parent)),
        "document_count": len(CORPUS),
        "query_count": len(queries),
        "top_k": TOP_K,
        "results": {
            "bm25_only": bm25_metrics,
            "dense_only": {**dense_metrics, "note": "mock retriever — fixed 5 docs regardless of query"},
            "rrf_merged": rrf_metrics,
        },
        "rrf_parameters": {
            "k": 60,
            "note": "k=60 is standard RRF constant; higher k = more rank equality",
        },
        "conclusion": (
            f"RRF achieves {rrf_gain:+.1f}% MRR vs best single retriever "
            f"at {rrf_overhead:.1f}x BM25 latency overhead"
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
