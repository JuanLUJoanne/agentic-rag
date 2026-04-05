"""
FastAPI application — public HTTP interface for both RAG workflows.

POST /query          run simple or multi-agent workflow, return grounded answer
POST /query/stream   SSE stream of workflow events
GET  /health         liveness probe
GET  /metrics        Prometheus metrics
GET  /costs          cost summary by model
GET  /eval/drift     latest drift report (auto-saved baseline on first call)
GET  /compliance/pii-report   PII detection statistics from the audit log
/review/*            human-in-the-loop review queue (see human_review.py)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, field_validator

from src.api.dedup import get_deduplicator
from src.api.human_review import router as review_router
from src.api.middleware import APIKeyMiddleware, load_api_keys
from src.api.streaming import stream_query
from src.gateway.cost_tracker import get_default_tracker
from src.graph.multi_agent_workflow import get_initial_supervisor_state
from src.graph.multi_agent_workflow import graph as multi_agent_graph
from src.graph.simple_workflow import get_initial_state
from src.graph.simple_workflow import graph as simple_graph
from src.observability.metrics import get_metrics
from src.observability.tracing import get_tracer, setup_tracing

logger = structlog.get_logger()

# ── Concurrency cap — limits in-flight requests so shutdown drains cleanly ──────
_ACTIVE_SEM = asyncio.Semaphore(500)

app = FastAPI(
    title="Agentic RAG",
    version="0.3.0",
    description=(
        "Corrective RAG with self-reflection, multi-agent orchestration, "
        "streaming, human-in-the-loop review, and eval drift detection"
    ),
)

# Authentication middleware
app.add_middleware(APIKeyMiddleware, keys=load_api_keys())

# Mount the review router
app.include_router(review_router)


# ── Lifecycle events ────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup_event() -> None:
    setup_tracing("agentic-rag")
    logger.info("startup_complete")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("shutdown_initiated")


# ── Request / response models ───────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str
    mode: Literal["simple", "multi_agent"] = "simple"
    max_retries: int = 2

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    agent_trace: list[dict[str, Any]]
    cost_so_far: float
    mode: str
    agents_used: list[str]
    iteration_count: int


# ── Endpoints ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Expose Prometheus metrics in text/plain format."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Run the requested workflow and return a grounded answer.

    mode=simple      → single-graph Corrective RAG (fast, lower cost)
    mode=multi_agent → skills-based supervisor with specialist agents
    """
    metrics = get_metrics()
    tracer = get_tracer()
    dedup = get_deduplicator()

    # Deduplication key: sha256(query + mode)
    dedup_key = hashlib.sha256(f"{request.query}:{request.mode}".encode()).hexdigest()

    async with _ACTIVE_SEM:
        metrics.active_requests.inc()
        import time

        start = time.perf_counter()
        status = "success"
        try:
            with tracer.start_as_current_span("rag.workflow") as span:
                thread_id = str(uuid.uuid4())
                span.set_attribute("workflow.mode", request.mode)
                span.set_attribute("workflow.thread_id", thread_id)

                config = {"configurable": {"thread_id": thread_id}}

                logger.info(
                    "api_query_start",
                    query=request.query[:80],
                    mode=request.mode,
                    thread_id=thread_id,
                )

                try:
                    if request.mode == "multi_agent":

                        async def _run_multi():
                            initial_state = get_initial_supervisor_state(
                                request.query,
                                mode="multi_agent",
                                max_retries=request.max_retries,
                            )
                            return await multi_agent_graph.ainvoke(
                                initial_state, config=config
                            )

                        final_state = await dedup.get_or_run(dedup_key, _run_multi)
                    else:

                        async def _run_simple():
                            initial_state = get_initial_state(
                                request.query, max_retries=request.max_retries
                            )
                            return await simple_graph.ainvoke(initial_state, config=config)

                        final_state = await dedup.get_or_run(dedup_key, _run_simple)

                except Exception as exc:
                    status = "error"
                    logger.error("workflow_error", error=str(exc), query=request.query[:80])
                    raise HTTPException(
                        status_code=500, detail=f"Workflow error: {exc}"
                    ) from exc

            logger.info(
                "api_query_complete",
                thread_id=thread_id,
                mode=request.mode,
                has_answer=bool(final_state.get("final_answer")),
            )

            return QueryResponse(
                answer=final_state.get("final_answer") or "",
                citations=final_state.get("citations") or [],
                agent_trace=final_state.get("agent_trace") or [],
                cost_so_far=final_state.get("cost_so_far") or 0.0,
                mode=request.mode,
                agents_used=final_state.get("agents_called") or [],
                iteration_count=final_state.get("iteration_count") or 0,
            )
        finally:
            elapsed = time.perf_counter() - start
            metrics.request_duration_seconds.labels(
                mode=request.mode, status=status
            ).observe(elapsed)
            metrics.active_requests.dec()


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest) -> StreamingResponse:
    """
    Stream workflow events as Server-Sent Events.

    Each event is a JSON line prefixed with ``data: ``.  The stream ends
    with an event of type ``done`` carrying the full answer summary.
    """
    async with _ACTIVE_SEM:
        logger.info(
            "api_stream_start",
            query=request.query[:80],
            mode=request.mode,
        )
        return StreamingResponse(
            stream_query(request.query, mode=request.mode),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


@app.get("/costs")
async def costs_endpoint() -> dict:
    """Return accumulated cost summary broken down by model."""
    tracker = get_default_tracker()
    return {
        "total_cost": float(tracker.total_cost),
        "remaining_budget": float(tracker.remaining_budget),
        "by_model": tracker.summary_by_model(),
    }


@app.get("/eval/drift")
async def eval_drift_endpoint() -> dict:
    """
    Run a quick single-query evaluation and compare against the saved baseline.

    On first call there is no baseline; one is auto-saved from this run.
    """
    from src.eval.drift_detector import DriftDetector
    from src.eval.ragas_eval import RAGEvaluator

    evaluator = RAGEvaluator()
    detector = DriftDetector()

    # Quick smoke test: one representative query
    probe_query = "What is retrieval-augmented generation?"
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    state = await simple_graph.ainvoke(
        get_initial_state(probe_query), config=cfg
    )
    result = evaluator.evaluate_single(
        probe_query,
        state.get("final_answer", ""),
        [d.get("content", "") for d in state.get("retrieved_docs", [])],
    )
    report = detector.detect_drift([result])

    return {
        "baseline_version": report.baseline_version,
        "alert_triggered": report.alert_triggered,
        "degraded_dimensions": report.degraded_dimensions,
        "current_scores": report.current_scores,
        "per_dimension_deltas": report.per_dimension_deltas,
    }


@app.get("/compliance/pii-report")
async def pii_report_endpoint(hours: int = 24) -> dict:
    """
    Return PII detection statistics aggregated from the audit log.

    Reads all entries in ``data/audit.jsonl`` that have an ``event_type``
    field (i.e. PII compliance events) and fall within the requested time
    window.  Returns counts by PII type, by detection layer, and the total
    number of PII instances redacted.

    Query parameters
    ----------------
    hours: int
        How far back to look (default 24 h).
    """
    audit_path = Path("data/audit.jsonl")
    empty: dict = {
        "total_pii_detected": 0,
        "by_type": {},
        "by_layer": {},
        "period_hours": hours,
    }

    if not audit_path.exists():
        return empty

    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    total = 0
    by_type: dict[str, int] = {}
    by_layer: dict[str, int] = {}

    try:
        with audit_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "event_type" not in entry:
                    continue
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts < cutoff:
                    continue
                pii_count = entry.get("pii_count", 0)
                total += pii_count
                for t in entry.get("pii_types", []):
                    by_type[t] = by_type.get(t, 0) + pii_count
                evt = entry["event_type"]
                by_layer[evt] = by_layer.get(evt, 0) + 1
    except OSError as exc:
        logger.warning("pii_report_read_failed", reason=str(exc))
        return empty

    return {
        "total_pii_detected": total,
        "by_type": by_type,
        "by_layer": by_layer,
        "period_hours": hours,
    }


if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
