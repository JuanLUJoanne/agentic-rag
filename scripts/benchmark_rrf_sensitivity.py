"""
RRF k-parameter sensitivity analysis.

Measures how MRR@5 and top-1 change rate vary across k ∈ [1,10,20,40,60,100,200].
Uses _rrf_merge directly from parallel_retriever so we test production code.

Key intuition:
  k=1  → score gap between rank-1 and rank-2 is large (1/1=1.0 vs 1/2=0.5)
          first source's top result dominates
  k=60 → gap is tiny (1/60≈0.0167 vs 1/61≈0.0164) — standard balanced fusion
  k=200→ near-uniform: all ranks score almost the same, fusion ≈ random merge

Run:
    source .venv/bin/activate
    PYTHONPATH=. python scripts/benchmark_rrf_sensitivity.py
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from src.retrieval.bm25_retriever import SAMPLE_DOCS, BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.parallel_retriever import _rrf_merge, _SourceResult

QA_PATH = Path(__file__).parent.parent / "eval_data" / "qa_100.jsonl"
OUT_PATH = Path(__file__).parent.parent / "eval_results" / "rrf_sensitivity.json"

K_VALUES = [1, 10, 20, 40, 60, 100, 200]
TOP_K = 5


def _load_queries() -> list[dict]:
    return [json.loads(ln) for ln in QA_PATH.read_text().splitlines() if ln.strip()]


def _mrr(docs, keywords: list[str]) -> float:
    for rank, doc in enumerate(docs, 1):
        if any(kw.lower() in doc.content.lower() for kw in keywords):
            return 1.0 / rank
    return 0.0


async def main() -> None:
    queries = _load_queries()
    print(f"Loaded {len(queries)} queries, corpus: {len(SAMPLE_DOCS)} docs")

    bm25 = BM25Retriever()
    bm25.index(SAMPLE_DOCS)
    dense = DenseRetriever()

    # ── Collect source results once per query (reused across k values) ─────────
    source_results_per_query: list[tuple[list, list]] = []
    for q in queries:
        bm25_docs = await bm25.search(q["question"], TOP_K * 2)
        dense_docs = await dense.search(q["question"], TOP_K * 2)
        source_results_per_query.append((bm25_docs, dense_docs))

    # ── k=60 baseline: use BM25 top-1 as reference ────────────────────────────
    # We compare each k's top-1 against k=60's top-1 for top-1 change rate
    baseline_top1: list[str] = []
    for bm25_docs, dense_docs in source_results_per_query:
        sources = [
            _SourceResult("bm25", bm25_docs, 0.0, None),
            _SourceResult("dense", dense_docs, 0.0, None),
        ]
        merged = _rrf_merge(sources, k=60)[:TOP_K]
        baseline_top1.append(merged[0].doc_id if merged else "")

    # ── Sweep k values ─────────────────────────────────────────────────────────
    results = []
    for k in K_VALUES:
        t0 = time.monotonic()
        mrrs, top1_changes = [], []

        for i, (bm25_docs, dense_docs) in enumerate(source_results_per_query):
            sources = [
                _SourceResult("bm25", bm25_docs, 0.0, None),
                _SourceResult("dense", dense_docs, 0.0, None),
            ]
            merged = _rrf_merge(sources, k=k)[:TOP_K]
            kws = queries[i].get("relevant_keywords", [])
            mrrs.append(_mrr(merged, kws))
            top1_this = merged[0].doc_id if merged else ""
            top1_changes.append(1 if top1_this != baseline_top1[i] else 0)

        elapsed_ms = (time.monotonic() - t0) * 1000
        avg_mrr = round(sum(mrrs) / len(mrrs), 3)
        change_rate = round(sum(top1_changes) / len(top1_changes), 3)

        note = {
            1: "rank-1 doc dominates; fusion ≈ best-of-top-1 from each source",
            10: "top ranks still heavily weighted; cross-source boosts visible",
            20: "moderate rank-weighting; common sweet-spot for short lists",
            40: "approaching balanced; original Cormack et al. region",
            60: "standard balanced (Cormack et al. 2009)",
            100: "nearly uniform; cross-source overlap matters more than rank",
            200: "near-uniform weighting; fusion ≈ random merge of ranked lists",
        }.get(k, "")

        results.append({
            "k": k,
            "mrr_at_5": avg_mrr,
            "top1_change_rate_vs_k60": change_rate,
            "fusion_latency_ms": round(elapsed_ms, 2),
            "note": note,
        })
        print(f"  k={k:>3d}  MRR={avg_mrr:.3f}  top1_change={change_rate:.3f}  {elapsed_ms:.1f}ms")

    best = max(results, key=lambda r: r["mrr_at_5"])
    mrr_k60 = next(r["mrr_at_5"] for r in results if r["k"] == 60)
    mrr_k1 = next(r["mrr_at_5"] for r in results if r["k"] == 1)
    mrr_k200 = next(r["mrr_at_5"] for r in results if r["k"] == 200)

    report = {
        "benchmark": "rrf_k_sensitivity",
        "corpus_size": len(SAMPLE_DOCS),
        "query_count": len(queries),
        "top_k": TOP_K,
        "k_values_tested": K_VALUES,
        "results": results,
        "best_k": best["k"],
        "analysis": (
            f"k={best['k']} maximises MRR@5 ({best['mrr_at_5']:.3f}). "
            f"k=1 scores {mrr_k1:.3f} — first source's rank-1 doc dominates, "
            f"defeating cross-source promotion. "
            f"k=200 scores {mrr_k200:.3f} — ranks nearly equal so shared-doc boosts "
            f"drive ranking instead of relevance signals. "
            f"k=60 scores {mrr_k60:.3f} — the standard choice balances rank signal "
            f"preservation with cross-source document promotion."
        ),
        "note": (
            "Dense retriever is a mock returning 5 fixed docs regardless of query. "
            "top1_change_rate measures how often RRF at k differs from k=60 baseline. "
            "Production results with real dense retrieval would show larger k-sensitivity."
        ),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
