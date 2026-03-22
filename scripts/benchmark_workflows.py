"""
Workflow comparison benchmark — Simple Corrective RAG vs Multi-Agent Supervisor.

Runs both pipelines on eval_data/qa_100.jsonl using DummyLLM (no API key needed)
and measures completion rate, steps, retries, cost, latency, and quality.

Run:
    source .venv/bin/activate
    python scripts/benchmark_workflows.py
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

from src.eval.ragas_eval import RAGEvaluator
from src.graph.multi_agent_workflow import get_initial_supervisor_state
from src.graph.multi_agent_workflow import graph as multi_agent_graph
from src.graph.simple_workflow import get_initial_state
from src.graph.simple_workflow import graph as simple_graph

QA_PATH = Path(__file__).parent.parent / "eval_data" / "qa_100.jsonl"
OUT_PATH = Path(__file__).parent.parent / "eval_results" / "benchmark_workflows.json"

_evaluator = RAGEvaluator()


def _load_queries() -> list[dict]:
    return [json.loads(ln) for ln in QA_PATH.read_text().splitlines() if ln.strip()]


async def _run_simple(query: str, ground_truth: str) -> dict:
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0 = time.monotonic()
    state = await simple_graph.ainvoke(get_initial_state(query), config=cfg)
    latency_ms = (time.monotonic() - t0) * 1000

    answer = state.get("final_answer") or ""
    contexts = [d.get("content", "") for d in state.get("retrieved_docs", [])]
    trace = state.get("agent_trace", [])
    retries = sum(1 for t in trace if t.get("node") in ("rewrite_query", "check_hallucination"))
    eval_result = _evaluator.evaluate_single(query, answer, contexts, ground_truth)

    return {
        "completed": bool(answer),
        "steps": len(trace),
        "retries": retries,
        "cost": state.get("cost_so_far", 0.0),
        "latency_ms": latency_ms,
        "quality": eval_result.faithfulness,
    }


async def _run_multi(query: str, ground_truth: str) -> dict:
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0 = time.monotonic()
    state = await multi_agent_graph.ainvoke(
        get_initial_supervisor_state(query, mode="multi_agent"), config=cfg
    )
    latency_ms = (time.monotonic() - t0) * 1000

    answer = state.get("final_answer") or ""
    contexts = [d.get("content", "") for d in state.get("retrieved_docs", [])]
    trace = state.get("agent_trace", [])
    retries = state.get("iteration_count", 0)
    eval_result = _evaluator.evaluate_single(query, answer, contexts, ground_truth)

    return {
        "completed": bool(answer),
        "steps": len(trace),
        "retries": retries,
        "cost": state.get("cost_so_far", 0.0),
        "latency_ms": latency_ms,
        "quality": eval_result.faithfulness,
    }


def _avg(values: list[float]) -> float:
    return round(sum(values) / max(len(values), 1), 4)


def _aggregate(runs: list[dict]) -> dict:
    return {
        "completion_rate": round(sum(r["completed"] for r in runs) / len(runs), 3),
        "avg_steps": _avg([r["steps"] for r in runs]),
        "avg_retries": _avg([r["retries"] for r in runs]),
        "avg_cost": _avg([r["cost"] for r in runs]),
        "avg_latency_ms": _avg([r["latency_ms"] for r in runs]),
        "avg_quality": _avg([r["quality"] for r in runs]),
    }


async def main() -> None:
    queries = _load_queries()
    print(f"Loaded {len(queries)} queries")

    simple_by_diff: dict[str, list[dict]] = defaultdict(list)
    multi_by_diff: dict[str, list[dict]] = defaultdict(list)
    simple_all, multi_all = [], []

    for i, q in enumerate(queries):
        diff = q.get("difficulty", "simple")
        gt = q.get("expected_answer", "")

        s = await _run_simple(q["question"], gt)
        m = await _run_multi(q["question"], gt)

        simple_all.append(s)
        multi_all.append(m)
        simple_by_diff[diff].append(s)
        multi_by_diff[diff].append(m)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(queries)} queries done")

    overall_simple = _aggregate(simple_all)
    overall_multi = _aggregate(multi_all)

    def _winner(diff: str) -> dict:
        s = _aggregate(simple_by_diff[diff])
        m = _aggregate(multi_by_diff[diff])
        # Quality wins; tie-break on latency
        if m["avg_quality"] > s["avg_quality"] + 0.01:
            winner, reason = "multi_agent", "higher quality score"
        elif s["avg_quality"] > m["avg_quality"] + 0.01:
            winner, reason = "simple_rag", "higher quality at lower cost"
        elif s["avg_latency_ms"] <= m["avg_latency_ms"]:
            winner, reason = "simple_rag", "same quality, lower latency"
        else:
            winner, reason = "multi_agent", "same quality, faster"
        return {"winner": winner, "reason": reason,
                "simple_quality": s["avg_quality"], "multi_quality": m["avg_quality"],
                "simple_latency_ms": s["avg_latency_ms"], "multi_latency_ms": m["avg_latency_ms"]}

    report = {
        "benchmark": "workflow_comparison",
        "dataset": str(QA_PATH.relative_to(QA_PATH.parent.parent)),
        "query_count": len(queries),
        "llm": "DummyLLM (offline)",
        "results": {
            "simple_corrective_rag": overall_simple,
            "multi_agent_supervisor": overall_multi,
        },
        "by_difficulty": {
            diff: _winner(diff)
            for diff in ["simple", "comparison", "multi_hop", "ambiguous"]
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to {OUT_PATH}")
    print(f"\nSimple RAG:   completion={overall_simple['completion_rate']}, "
          f"avg_steps={overall_simple['avg_steps']}, latency={overall_simple['avg_latency_ms']}ms")
    print(f"Multi-Agent:  completion={overall_multi['completion_rate']}, "
          f"avg_steps={overall_multi['avg_steps']}, latency={overall_multi['avg_latency_ms']}ms")


if __name__ == "__main__":
    asyncio.run(main())
