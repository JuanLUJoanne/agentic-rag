# Failure Modes & Recovery

## Parallel Retrieval Failures

| Failure | Trigger | Impact | Recovery |
|---------|---------|--------|----------|
| BM25 timeout | Query takes > 5s (unusual — BM25 is in-memory) | Loses keyword matches for that query | `_timed_search` catches `TimeoutError`; RRF proceeds with Dense + Graph only. MRR degrades on exact-match queries. |
| Dense retriever error | OOM, network failure, or model load error | Loses semantic matches | `_timed_search` catches all exceptions; source logged as failed. RRF proceeds with BM25 + Graph. |
| Graph retriever timeout | Neo4j slow or unreachable (5s timeout) | Loses entity-relationship context | Silently excluded from RRF merge. Only affects `query_type="complex"` queries. |
| All retrievers fail | Full infrastructure outage | `retrieved_docs = []` | `grade_documents` marks `docs_relevant="none"` → `web_search` node fires as last-resort fallback. |
| RRF score tie | Two docs share identical RRF score | Arbitrary ordering of tied documents | Ties broken by insertion order (first source wins). In production, break by document recency to avoid serving stale content. |
| Corpus empty at index time | BM25 indexed before documents loaded | All BM25 scores = 0 → no results | `BM25Retriever.search()` returns `[]` when `_bm25 is None`. Pipeline falls through to Dense only. |

**Real degradation numbers** (from `eval_results/rrf_sensitivity.json`, `eval_results/benchmark_retrieval.json`):

- BM25-only MRR@5: **0.663** (191-doc corpus, 100 queries)
- RRF with mock Dense: **0.659** — nearly identical because mock Dense returns fixed docs
- RRF k=1 vs k≥10: **0.321 vs 0.328 MRR** — k<10 costs 2% MRR due to rank-1 dominance
- In production with real Dense: RRF consistently outperforms single-source by 5–15% MRR (RRF paper, Cormack 2009)

---

## RRF Parameter Failure Modes

| Failure | Impact | Recovery |
|---------|--------|----------|
| `k` too low (< 10) | Rank-1 doc from first source dominates; cross-source promotion nearly eliminated. At k=1, score ratio rank-1:rank-2 = 2:1. | Use k ≥ 40 in production. k=60 is the standard validated choice. |
| `k` too high (> 200) | All ranks treated nearly equally; fusion degrades to a random merge. Cross-source overlap replaces relevance as the ranking signal. | Keep k ≤ 100 for corpora where ranks carry signal. |
| Single source active | RRF with one source = identity function (no fusion benefit) | Acceptable degradation. Log a warning when only one source returns results. |

---

## Supervisor Failures

| Failure | Trigger | Impact | Recovery |
|---------|---------|--------|----------|
| Supervisor misroutes simple query to full pipeline | DummyLLM always runs research → analysis → quality; real LLM may over-route | 2–3 extra agent dispatches, 2× cost | QualityAgent still validates the answer. Output is correct; overhead is cost/latency, not correctness. `benchmark_supervisor.py` shows 30% misroute rate on simple queries. |
| Supervisor loop (re-dispatches same agent) | LLM ignores `agents_called` state | Infinite loop risk | `max_iterations=5` hard cap. Exceeded → `next_agent="done"` with `reasoning="max_iterations_reached"`. |
| Budget exceeded mid-pipeline | `cost_so_far >= budget` | Pipeline terminates early, potentially before quality check | `_supervisor.decide()` returns `done` with `reasoning="budget_exceeded"`. `finalize` uses best available generation. |
| CostGuardrail pre-flight block | Estimated cost would exceed per-request limit | Agent dispatch skipped | Returns `done` with guardrail reason. Current answer (if any) is finalized. |
| Agent raises unhandled exception | Bug in ResearchAgent, AnalysisAgent, QualityAgent | Node raises, LangGraph surfaces error to caller | Wrap agent dispatch nodes in try/except in production; return partial state on exception rather than propagating. |
| LLM returns unparseable JSON | Real LLM outputs narrative instead of JSON | Supervisor cannot route | `_llm_decide` catches `json.JSONDecodeError` and falls back to `_dummy_decide` deterministic sequence. |
| QualityAgent always rejects | Hallucination checker is too strict | Infinite quality retry loop | `max_retries=2` in `route_after_hallucination`; exceeded → `finalize` anyway. Human review path triggered if `answer_quality < 0.7`. |

**Real routing numbers** (from `eval_results/supervisor_analysis.json`):

- Overall routing accuracy: **70%** (35/50 queries) — misroutes are exclusively simple queries over-routed to full pipeline
- Recovery rate: **100%** — all misrouted queries still produce a correct answer
- Misroute cost: **+2 extra agent dispatches** (analysis + quality) for 30% of queries
- Supervisor decision latency (DummyLLM): **0.04 ms** — real LLM adds ~500–1000 ms per decision

### Parallel Dispatch Failures

| Failure | Trigger | Impact | Recovery |
|---------|---------|--------|----------|
| One parallel agent raises exception | Bug in ResearchAgent or AnalysisAgent | `asyncio.gather(return_exceptions=True)` captures the error; the other agent's result is still merged | Failed agent's result logged as error; supervisor can re-dispatch on the next iteration |
| Research subgraph rewrite loop exhausted | All retrieved docs graded as irrelevant after 2 rewrites | Subgraph proceeds to `synthesize` with best-effort documents | `max_rewrites=2` cap prevents infinite loops; synthesis uses whatever docs are available |
| Research subgraph grade_relevance wrong | Grader marks relevant docs as irrelevant (or vice versa) | Unnecessary rewrite (wasted latency) or skipped rewrite (lower quality) | QualityAgent downstream catches low-quality answers; human review path for persistent issues |
| Parallel merge conflict on scalar fields | Both agents return `generation` | `parallel_dispatch_node` uses last-write-wins for scalar fields | Research result takes priority for `retrieved_docs`; analysis result takes priority for `generation` — matching their respective responsibilities |

---

**When NOT to use the supervisor:**

Simple factual queries are better served by the Simple Corrective RAG pipeline:
- `benchmark_workflows.py` shows ~23ms latency and 0 extra agent cost vs 22ms for multi-agent
- The supervisor adds overhead proportional to `max_iterations × LLM_latency` — with a real LLM that is 1–4 extra calls per query

---

## Cost & Budget Failure Modes

| Failure | Impact | Recovery |
|---------|--------|----------|
| LLM cost spike (prompt injection creates huge context) | Budget blown on single query | Per-request limit ($0.05); `CostGuardrail` blocks before the LLM call. |
| Decimal arithmetic overflow | Not realistic — Decimal handles arbitrary precision | N/A — Decimal chosen precisely to avoid float drift on budget comparisons. |
| Budget tracker reset between requests | Cost accumulated in-process; restarts zero the counter | Use persistent storage (DB) for cross-process budget enforcement in production. |
| Rate limit cascade (429 from provider) | All queued queries fail | Token-bucket rate limiter enforces RPM/TPM pre-emptively. On 429, `on_429` callback halves the rate for 60s. |

---

## Cache Failure Modes

| Failure | Impact | Recovery |
|---------|--------|----------|
| SemanticCache FAISS unavailable | Falls back to `_NumpyIndex` (pure numpy) | Transparent degradation; performance drops at > 10K cached entries but correctness unchanged. |
| SemanticCache threshold too low (< 0.85) | Serves stale or off-topic cached answers for loosely similar queries | Raise threshold or add query-type metadata to cache key. Default 0.95 is conservative. |
| EmbeddingCache SQLite lock contention | Concurrent writes block each other | `threading.Lock` serialises writes. High-concurrency production use should switch to a Redis backend. |
| TTL too short | Cache entries expire before they accumulate enough hits | Increase `ttl` (default 3600s). Monitor `hit_rate` from `stats()`. |
| Faithfulness gate too strict (min_faithfulness = 1.0) | Nothing gets cached; cache is permanently empty | Keep `min_faithfulness ≤ 0.90`. Values above 0.95 effectively disable caching in most deployments. |

---

## PII Compliance Failure Modes

| Failure | Impact | Recovery |
|---------|--------|----------|
| Presidio not installed | NER-based detection unavailable | Regex fallback activates automatically. Covers EMAIL, PHONE, CREDIT_CARD, IP, PERSON heuristic, AU IDs. Lower recall on uncommon name formats. |
| Confidence threshold too low | Increases false positives; legitimate content redacted | Raise `confidence_threshold` (default 0.7). Test against allow_list for known safe entities. |
| Redaction map not consulted | De-redaction impossible after response sent | Redaction map is per-request in-memory. For auditability, store the map alongside the audit log entry. |
| OutputScanner misses hallucinated PII | LLM fabricates an email/phone not in context | Cannot detect fabricated PII that passes NER/regex. Strict mode + human review on high-sensitivity queries is the only defence. |
