#!/usr/bin/env python3
"""
Agentic RAG — end-to-end demo.

Runs 5 queries that exercise every major design pattern:

  1. Simple factual        (simple mode, fast path)
  2. Compare X and Y      (complex, multi-agent supervisor)
  3. Repeat query 1       (memory cache hit — zero LLM calls)
  4. Ambiguous query      (routed to clarification / ambiguous path)
  5. Multi-hop question   (parallel retrieval + GraphRAG context)

Usage:
    python scripts/demo.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from src.gateway.cost_tracker import get_default_tracker
from src.graph.multi_agent_workflow import get_initial_supervisor_state
from src.graph.multi_agent_workflow import graph as multi_agent_graph
from src.graph.simple_workflow import get_initial_state
from src.graph.simple_workflow import graph as simple_graph
from src.retrieval.memory import get_default_memory

logger = structlog.get_logger()

# ── Query definitions ──────────────────────────────────────────────────────────

_QUERIES = [
    {
        "id": 1,
        "label": "Simple factual",
        "query": "What is retrieval-augmented generation?",
        "mode": "simple",
        "pattern": "Routing + Corrective RAG",
    },
    {
        "id": 2,
        "label": "Compare X and Y",
        "query": "What is the difference between RAG and fine-tuning for domain adaptation?",
        "mode": "multi_agent",
        "pattern": "Multi-Agent Supervisor + Planning",
    },
    {
        "id": 3,
        "label": "Memory cache hit",
        "query": "What is retrieval-augmented generation?",  # same as Q1
        "mode": "simple",
        "pattern": "Memory (cache hit → zero LLM calls)",
    },
    {
        "id": 4,
        "label": "Ambiguous query",
        "query": "Tell me about it",
        "mode": "simple",
        "pattern": "Routing → ambiguous classification",
    },
    {
        "id": 5,
        "label": "Multi-hop question",
        "query": (
            "How does the corrective RAG rewrite loop interact with "
            "the parallel BM25 and dense retrieval pipeline?"
        ),
        "mode": "multi_agent",
        "pattern": "Parallelization + GraphRAG + Multi-Agent",
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────

_DIV = "─" * 72


def _section(title: str) -> None:
    print(f"\n{_DIV}")
    print(f"  {title}")
    print(_DIV)


def _kv(key: str, value: object) -> None:
    print(f"  {key:<20} {value}")


# ── Per-query runner ───────────────────────────────────────────────────────────


async def _run_query(spec: dict, memory_hit_count: list[int]) -> dict:
    _section(f"Query {spec['id']}: {spec['label']}")
    _kv("Query", spec["query"][:70])
    _kv("Mode", spec["mode"])
    _kv("Pattern", spec["pattern"])

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    t0 = time.monotonic()
    if spec["mode"] == "multi_agent":
        state = get_initial_supervisor_state(spec["query"])
        final = await multi_agent_graph.ainvoke(state, config=config)
    else:
        state = get_initial_state(spec["query"])
        final = await simple_graph.ainvoke(state, config=config)
    elapsed = time.monotonic() - t0

    answer = final.get("final_answer") or "(no answer)"
    agents = final.get("agents_called") or []
    cost = final.get("cost_so_far") or 0.0
    trace = final.get("agent_trace") or []

    # Detect memory hit: first trace entry is memory_check with hit=True
    memory_hit = bool(
        trace and trace[0].get("node") == "memory_check" and trace[0].get("hit")
    )
    if memory_hit:
        memory_hit_count[0] += 1

    print()
    _kv("Agents used", ", ".join(agents) if agents else "none (simple path)")
    _kv("Memory hit", "YES — served from cache" if memory_hit else "no")
    _kv("Cost", f"${cost:.6f}")
    _kv("Latency", f"{elapsed:.2f}s")
    _kv("Trace steps", len(trace))
    print(f"\n  Answer:\n  {answer[:200]}")

    return {"cost": cost, "memory_hit": memory_hit, "latency": elapsed}


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    print("\nAgentic RAG — Batch 6 Demo")
    print("10 design patterns · 5 queries · all offline (DummyLLM)\n")

    # Prime memory: learn Q1's answer after first run so Q3 hits the cache
    memory = get_default_memory()

    total_cost = 0.0
    cache_hits = 0
    memory_hit_count = [0]  # mutable for closure

    results = []
    for spec in _QUERIES:
        r = await _run_query(spec, memory_hit_count)
        results.append(r)
        total_cost += r["cost"]
        if r["memory_hit"]:
            cache_hits += 1

        # After Q1 completes, store its answer in memory so Q3 gets a cache hit
        if spec["id"] == 1:
            from src.graph.simple_workflow import graph as sg
            thread_id = str(uuid.uuid4())
            final_state = await sg.ainvoke(
                get_initial_state(spec["query"]),
                config={"configurable": {"thread_id": thread_id}},
            )
            await memory.learn(
                spec["query"],
                final_state.get("final_answer") or "RAG combines retrieval and generation.",
                [],
                eval_score=0.92,
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("Summary")
    _kv("Queries processed", len(_QUERIES))
    _kv("Memory cache hits", cache_hits)
    _kv("Total cost (DummyLLM)", f"${total_cost:.6f}")
    _kv("Avg latency", f"{sum(r['latency'] for r in results) / len(results):.2f}s")

    tracker = get_default_tracker()
    if tracker.summary_by_model():
        print("\n  Cost by model:")
        for model, info in tracker.summary_by_model().items():
            print(f"    {model:<20} ${info['total_cost']:.6f}  "
                  f"({info['input_tokens']}i / {info['output_tokens']}o tokens)")

    print(f"\n{_DIV}")
    print("  Demo complete. All 10 agentic patterns exercised.")
    print(_DIV)


def _run_finetuning_demo() -> None:
    """Standalone fine-tuning demo — safe to run without GPU."""
    print("\n" + "=" * 60)
    print("FINE-TUNING DEMO")
    print("=" * 60)

    from src.finetuning.embedding_ft import EmbeddingFineTuner
    from src.finetuning.qlora_dpo import QAExample, QLoRADPOTrainer

    # ── Embedding fine-tuning ─────────────────────────────────────────────────
    print("\n--- Embedding Fine-Tuning ---")
    emb_ft = EmbeddingFineTuner()
    queries = [
        "What is retrieval augmented generation?",
        "How does vector search work?",
        "Explain BM25 scoring",
    ]
    corpus = [
        "RAG combines document retrieval with language model generation for grounded answers.",
        "Vector search uses embedding similarity to find semantically related documents.",
        "BM25 is a probabilistic ranking function based on term frequency and document length.",
        "Kubernetes orchestrates containerized applications across clusters.",
        "Python is a high-level programming language used in data science.",
    ]
    relevance = {"0": [0], "1": [1], "2": [2]}

    triples = emb_ft.prepare_data(queries, corpus, relevance)
    print(f"  Prepared {len(triples)} training triples")
    train_result = emb_ft.train(triples, output_dir="/tmp/demo_embedding")
    print(f"  Training: {train_result}")
    eval_result = emb_ft.evaluate(triples[:2])
    print(f"  Eval: recall@5={eval_result['recall@5']:.2f}, MRR@10={eval_result['mrr@10']:.2f}")

    # ── QLoRA + DPO ───────────────────────────────────────────────────────────
    print("\n--- QLoRA + DPO Fine-Tuning ---")
    trainer = QLoRADPOTrainer()
    qa_examples = [
        QAExample(
            context="RAG combines retrieval with generation.",
            question="What is RAG?",
            answer="RAG retrieves relevant documents then generates answers grounded in those documents.",
        ),
        QAExample(
            context="BM25 uses term frequency and inverse document frequency.",
            question="How does BM25 rank documents?",
            answer="BM25 scores documents by term frequency, inverse document frequency, and document length normalization.",
        ),
        QAExample(
            context="DPO aligns models with human preferences without a reward model.",
            question="What is DPO?",
            answer="Direct Preference Optimization trains models on preference pairs, avoiding the complexity of RLHF reward models.",
        ),
    ]

    qlora_result = trainer.train_qlora(qa_examples, output_dir="/tmp/demo_qlora")
    print(f"  QLoRA: {qlora_result}")
    dpo_pairs = trainer.generate_dpo_pairs(qa_examples)
    print(f"  Generated {len(dpo_pairs)} DPO pairs")
    dpo_result = trainer.train_dpo(dpo_pairs, output_dir="/tmp/demo_dpo")
    print(f"  DPO: {dpo_result}")
    eval_result = trainer.evaluate(qa_examples)
    print(f"  Base accuracy:        {eval_result['base_accuracy']:.0%}")
    print(f"  QLoRA accuracy:       {eval_result['qlora_accuracy']:.0%}")
    print(f"  DPO accuracy:         {eval_result['dpo_accuracy']:.0%}")
    print(
        f"  Hallucination: {eval_result['base_hallucination_rate']:.0%} "
        f"→ {eval_result['dpo_hallucination_rate']:.0%}"
    )


if __name__ == "__main__":
    asyncio.run(main())
    _run_finetuning_demo()
