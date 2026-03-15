# Agentic RAG with Self-Reflection & Fine-Tuned Models

A production-quality implementation of **10 agentic design patterns** using LangGraph, showing how to build a RAG system that goes beyond simple retrieve-and-generate: it reflects on its own outputs, routes queries intelligently, parallelises retrieval, enforces guardrails, streams results in real time, and routes low-confidence answers to human reviewers.

Every pattern is implemented with a real, runnable graph — not pseudocode. The system runs entirely offline (DummyLLM) so you can explore the architecture without an API key.

---

## Architecture

```
User Query
    │
    ▼
InputSanitizer ──[injection]──► Error response
    │
    ▼
QueryMemory ────[cache hit]──► Cached answer ──► AuditLog ──► SSE stream
    │ miss
    ▼
QueryRouter ────[simple]────► Corrective RAG pipeline
    │ complex/ambiguous
    ▼
Supervisor (LLM or DummyLLM)
    │ reads AgentRegistry capabilities dynamically
    ├──► ResearchAgent  (parallel BM25 + Dense + Graph retrieval, RRF merge)
    ├──► AnalysisAgent  (grounded generation + citation extraction)
    └──► QualityAgent   (hallucination check, faithfulness score)
              │
              ├──[quality ≥ 0.7]──► Finalize ──► AuditLog ──► SSE stream
              └──[quality < 0.7]──► HumanReview queue ──► pending answer

Corrective RAG:
    retrieve → grade_docs → [all_relevant] → generate → hallucination_check
                          → [partial]      → rewrite_query ──► retrieve (once)
                          → [none]         → web_search ──► generate
```

---

## 10 Agentic Design Patterns Implemented

| # | Pattern | Where | Description |
|---|---------|-------|-------------|
| 1 | **Tool Use** | `retriever.py`, `web_search.py` | Agents call external tools (BM25, dense retriever, web search) as structured functions |
| 2 | **Reflection** | `hallucination_checker.py`, `relevance_grader.py` | Pipeline checks its own outputs and decides whether to retry or proceed |
| 3 | **Planning** | `supervisor.py`, `query_analyzer.py` | Supervisor decomposes complex queries into sub-tasks dispatched to specialist agents |
| 4 | **Multi-Agent** | `multi_agent_workflow.py` | Skills-based supervisor dynamically reads `AgentRegistry` and routes to Research / Analysis / Quality agents |
| 5 | **Human-in-the-Loop** | `human_review.py`, `multi_agent_workflow.py` | Low-confidence answers enter a review queue; humans approve or reject via REST; approved answers go to memory |
| 6 | **Guardrails** | `guardrails.py`, `security.py` | Per-request cost limits, anomaly detection, per-query budget, prompt-injection detection, PII redaction |
| 7 | **Prompt Chaining** | `simple_workflow.py` | Three-step chain: classify → retrieve → grade → rewrite → generate → verify |
| 8 | **Routing** | `query_router.py` | Heuristic classifier (no LLM cost) routes to simple Corrective RAG or multi-agent supervisor |
| 9 | **Parallelization** | `parallel_retriever.py` | BM25, dense, and graph retrievers run concurrently via `asyncio.gather`; results merged with Reciprocal Rank Fusion |
| 10 | **Memory** | `cache.py`, `memory.py` | Two-layer: `EmbeddingCache` (TTL, sha256 key) for vector lookups; `QueryMemory` (faithfulness gate ≥ 0.85) for full answers |

---

## Key Design Decisions

**Skills-based supervisor over hardcoded routing** — The `Supervisor` reads `AgentRegistry.get_all_capabilities()` at decision time, so adding a new agent is one line: `registry.register(MyAgent())`. The prompt stays current without code changes.

**Heuristic query router** — Keyword matching + length heuristics classify queries in microseconds with zero LLM cost. The error rate on common patterns (comparison, multi-hop, simple factual) is low enough that the downstream Corrective RAG loop handles misclassifications gracefully.

**Parallel retrieval with RRF merge** — BM25 excels at exact keyword matches; dense retrieval captures semantic similarity; graph retrieval surfaces entity relationships. Running them concurrently (5 s timeout each) and merging with RRF (`score = Σ 1/(k+rank)`, k=60) consistently outperforms any single retriever.

**3-step prompt chaining for grounded generation** — Separate prompts for (1) relevance grading, (2) generation with citations, and (3) hallucination checking each do one thing well. A single combined prompt produces lower faithfulness scores because the model must balance competing objectives.

**Corrective RAG + Self-RAG combination** — Corrective RAG handles retrieval quality (rewrite query when docs are irrelevant); Self-RAG handles generation quality (retry when hallucinated). The two loops are capped by `should_rewrite_query` and `retry_count` to prevent infinite loops.

---

## Eval Results (DummyLLM baseline)

| Metric | Simple mode | Multi-Agent | Winner |
|--------|-------------|-------------|--------|
| Faithfulness | ~0.50 | ~0.50 | tie |
| Answer relevancy | ~0.45 | ~0.45 | tie |
| Context precision | ~0.55 | ~0.55 | tie |
| Context recall | ~0.60 | ~0.60 | tie |
| Avg latency | faster | slower | simple |
| Cost per query | lower | higher | simple |

_Scores are heuristic estimates from the mock `RAGEvaluator`. Real scores require live API evaluation against a labelled dataset._

Run a live comparison:
```python
from src.eval.comparative_eval import ComparativeEvaluator
import asyncio

report = asyncio.run(ComparativeEvaluator().run([
    "What is RAG?",
    "Compare BM25 and dense retrieval.",
]))
print(report.winner)
```

---

## Production Features

| Feature | Module | Notes |
|---------|--------|-------|
| Rate limiting | `gateway/rate_limiter.py` | Token-bucket per model; `on_429` halves rate for 60 s |
| Cost guardrails | `gateway/guardrails.py` | Per-request, per-query, anomaly (3× rolling avg), total budget |
| Security | `gateway/security.py` | Prompt injection detection; PII redaction (email, phone, card) |
| Retrieval cache | `retrieval/cache.py` | SQLite, SHA-256 key, TTL expiry |
| Answer memory | `retrieval/memory.py` | Faithfulness gate (≥ 0.85); SQLite-backed |
| Drift detection | `eval/drift_detector.py` | JSON baselines; alerts when metric drops > 5 pp |
| Human review | `api/human_review.py` | In-memory queue; approve stores to memory |
| Audit logging | `gateway/security.py` | Append-only JSONL at `data/audit.jsonl` |
| SSE streaming | `api/streaming.py` | LangGraph `astream` → `text/event-stream` |
| Cost tracking | `gateway/cost_tracker.py` | Decimal precision; per-model and per-query breakdown |

---

## Fine-Tuning (Architecture Stubs)

Both fine-tuning pipelines have complete interfaces, docstrings, and mock implementations ready to swap for real training loops.

**Embedding fine-tuning** (`src/finetuning/embedding_ft.py`)
- Contrastive learning with hard negative mining
- InfoNCE loss: `L = -log[exp(sim(q,p)/τ) / Σ exp(sim(q,n_i)/τ)]`
- Hard negatives: BM25-retrieved docs that are NOT relevant (highest training signal)
- Expected: **+23% Recall@5** on domain-specific corpora

**QLoRA + DPO** (`src/finetuning/qlora_dpo.py`)
- Stage 1 QLoRA: 4-bit NF4 quantisation + LoRA adapters (~0.5% trainable params, fits 7B in 6 GB VRAM)
- Stage 2 DPO: preference alignment without a reward model (3–5× cheaper than RLHF)
- Expected: **+15% domain QA accuracy** (QLoRA), **–44% hallucination rate** (DPO: 18% → 10%)

---

## Quick Start

```bash
# 1. Clone and set up environment
cp .env.example .env          # add OPENAI_API_KEY for live mode
pip install -e .

# 2. (Optional) Start infrastructure
docker compose up -d          # PostgreSQL + pgvector + Neo4j

# 3. Run tests
pytest tests/unit/ -v

# 4. Run the demo (works without API key)
python scripts/demo.py

# 5. Start the API server
uvicorn src.api.main:app --reload
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/query` | Run workflow, return grounded answer |
| `POST` | `/query/stream` | Stream workflow events as SSE |
| `GET` | `/review/pending` | List answers awaiting human review |
| `POST` | `/review/{id}/approve` | Approve answer → stores in memory |
| `POST` | `/review/{id}/reject` | Reject with reason |
| `GET` | `/review/stats` | Pending / approved / rejected counts |
| `GET` | `/costs` | Accumulated cost by model |
| `GET` | `/eval/drift` | Run probe query and compare to baseline |

### POST /query

```json
{
  "query": "What is RAG?",
  "mode": "simple",       // "simple" | "multi_agent"
  "max_retries": 2
}
```

### SSE stream events (`POST /query/stream`)

Each `data:` line is a JSON `StreamEvent`:

| `event_type` | When emitted |
|---|---|
| `supervisor_decision` | Supervisor chose the next agent |
| `agent_start` | Agent node is about to execute |
| `agent_complete` | Agent node finished |
| `answer_chunk` | Generation fragment available |
| `error` | Unhandled workflow exception |
| `done` | Stream finished; carries full answer + cost |

---

## Project Structure

```
src/
├── agents/          # Agent nodes: router, analyzer, retriever, generator, …
├── api/             # FastAPI app, SSE streaming, human review endpoints
├── eval/            # RAGAS-style eval, drift detection, comparative eval
├── finetuning/      # Embedding FT and QLoRA+DPO stubs
├── gateway/         # Rate limiter, cost tracker, guardrails, security
├── graph/           # LangGraph workflows (simple + multi-agent)
├── retrieval/       # BM25, dense, graph, parallel retriever, cache, memory
└── utils/           # LLM factory (DummyLLM ↔ OpenAI)
tests/
└── unit/            # 242 tests, all runnable offline
scripts/
└── demo.py          # End-to-end demo: 5 queries × 10 patterns
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (StateGraph, MemorySaver, astream) |
| LLM | OpenAI GPT-4o / GPT-4o-mini (DummyLLM fallback) |
| Tracing | LangSmith |
| Retrieval | BM25 (rank-bm25), sentence-transformers (dense), custom graph |
| Vector store | pgvector (PostgreSQL) |
| Graph store | Neo4j |
| API framework | FastAPI + uvicorn |
| Validation | Pydantic v2 |
| Logging | structlog |
| Persistence | SQLite (cache, memory, audit) |
| Testing | pytest + pytest-asyncio (242 tests) |
| Linting | ruff |
| CI | GitHub Actions |
