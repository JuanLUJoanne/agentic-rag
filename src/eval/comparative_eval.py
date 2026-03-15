"""
Comparative evaluation: simple_workflow vs multi_agent_workflow.

Runs the same set of queries through both pipeline modes and produces a
side-by-side metric comparison table.  Uses the mock RAGEvaluator so
no API key is required.

The report surfaces which mode wins on each metric (higher is better for
all RAGAS dimensions; lower is better for cost and latency).
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import structlog

from src.eval.ragas_eval import RAGEvalResult, RAGEvaluator

logger = structlog.get_logger()

_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "citation_accuracy",
    "cost",
    "latency_s",
]


@dataclass
class ComparativeReport:
    """Side-by-side comparison of simple vs multi-agent pipeline."""

    queries: list[str]
    simple_results: list[RAGEvalResult]
    multi_agent_results: list[RAGEvalResult]

    # Per-metric averages: {"faithfulness": {"simple": 0.8, "multi_agent": 0.9}}
    per_metric: dict[str, dict[str, float]] = field(default_factory=dict)

    # Which mode wins each metric
    winner: dict[str, str] = field(default_factory=dict)

    simple_avg_cost: float = 0.0
    multi_agent_avg_cost: float = 0.0
    simple_avg_latency: float = 0.0
    multi_agent_avg_latency: float = 0.0


class ComparativeEvaluator:
    """
    Evaluate both pipeline modes on the same queries and compare metrics.

    Uses DummyLLM by default so tests run offline.  Pass real queries and a
    live API key for production benchmarking.
    """

    def __init__(self) -> None:
        self._evaluator = RAGEvaluator()

    async def run(
        self,
        queries: list[str],
        ground_truths: list[str] | None = None,
    ) -> ComparativeReport:
        """
        Run both pipelines on every query and return a ComparativeReport.

        Parameters
        ----------
        queries:       Queries to evaluate.
        ground_truths: Reference answers (same length as queries, or None).
        """
        from src.graph.multi_agent_workflow import (
            get_initial_supervisor_state,
        )
        from src.graph.multi_agent_workflow import graph as multi_agent_graph
        from src.graph.simple_workflow import get_initial_state
        from src.graph.simple_workflow import graph as simple_graph

        if ground_truths is None:
            ground_truths = [""] * len(queries)

        simple_results: list[RAGEvalResult] = []
        multi_results: list[RAGEvalResult] = []
        simple_latencies: list[float] = []
        multi_latencies: list[float] = []

        for query, gt in zip(queries, ground_truths):
            # ── Simple workflow ──────────────────────────────────────────────
            t0 = time.monotonic()
            thread_id = str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": thread_id}}
            simple_state = await simple_graph.ainvoke(
                get_initial_state(query), config=cfg
            )
            simple_latencies.append(time.monotonic() - t0)

            contexts = [
                d.get("content", "") for d in simple_state.get("retrieved_docs", [])
            ]
            result_s = self._evaluator.evaluate_single(
                query,
                simple_state.get("final_answer", ""),
                contexts,
                gt,
            )
            result_s.agent_steps = len(simple_state.get("agent_trace", []))
            result_s.cost_usd = simple_state.get("cost_so_far", 0.0)
            simple_results.append(result_s)

            # ── Multi-agent workflow ─────────────────────────────────────────
            t0 = time.monotonic()
            thread_id = str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": thread_id}}
            multi_state = await multi_agent_graph.ainvoke(
                get_initial_supervisor_state(query, mode="multi_agent"), config=cfg
            )
            multi_latencies.append(time.monotonic() - t0)

            contexts_m = [
                d.get("content", "") for d in multi_state.get("retrieved_docs", [])
            ]
            result_m = self._evaluator.evaluate_single(
                query,
                multi_state.get("final_answer", ""),
                contexts_m,
                gt,
            )
            result_m.agent_steps = multi_state.get("iteration_count", 0)
            result_m.cost_usd = multi_state.get("cost_so_far", 0.0)
            multi_results.append(result_m)

        report = ComparativeReport(
            queries=queries,
            simple_results=simple_results,
            multi_agent_results=multi_results,
            simple_avg_cost=sum(r.cost_usd for r in simple_results) / max(len(simple_results), 1),
            multi_agent_avg_cost=sum(r.cost_usd for r in multi_results) / max(len(multi_results), 1),
            simple_avg_latency=sum(simple_latencies) / max(len(simple_latencies), 1),
            multi_agent_avg_latency=sum(multi_latencies) / max(len(multi_latencies), 1),
        )

        # Build per-metric comparison table
        ragas_dims = [
            "faithfulness", "answer_relevancy",
            "context_precision", "context_recall", "citation_accuracy",
        ]
        for dim in ragas_dims:
            s_avg = sum(getattr(r, dim) for r in simple_results) / max(len(simple_results), 1)
            m_avg = sum(getattr(r, dim) for r in multi_results) / max(len(multi_results), 1)
            report.per_metric[dim] = {"simple": round(s_avg, 3), "multi_agent": round(m_avg, 3)}
            report.winner[dim] = "multi_agent" if m_avg >= s_avg else "simple"

        # Cost and latency (lower is better)
        report.per_metric["cost"] = {
            "simple": round(report.simple_avg_cost, 6),
            "multi_agent": round(report.multi_agent_avg_cost, 6),
        }
        report.winner["cost"] = (
            "simple" if report.simple_avg_cost <= report.multi_agent_avg_cost else "multi_agent"
        )
        report.per_metric["latency_s"] = {
            "simple": round(report.simple_avg_latency, 3),
            "multi_agent": round(report.multi_agent_avg_latency, 3),
        }
        report.winner["latency_s"] = (
            "simple" if report.simple_avg_latency <= report.multi_agent_avg_latency else "multi_agent"
        )

        logger.info(
            "comparative_eval_complete",
            n_queries=len(queries),
            winner_summary={k: v for k, v in report.winner.items()},
            simple_latency=round(report.simple_avg_latency, 3),
            multi_latency=round(report.multi_agent_avg_latency, 3),
        )
        return report
