"""
Prometheus metrics for agentic-rag.

Provides a RequestMetrics singleton with:
  - request_duration_seconds  Histogram   labels: mode, status
  - cache_hits_total          Counter     labels: layer (memory/semantic/embedding)
  - retriever_errors_total    Counter     labels: source
  - llm_tokens_total          Counter     labels: model
  - active_requests           Gauge
"""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, REGISTRY

_instance: RequestMetrics | None = None


class RequestMetrics:
    """Container for all Prometheus collectors used by the API."""

    def __init__(self, registry: CollectorRegistry = REGISTRY) -> None:
        self.request_duration_seconds = Histogram(
            "request_duration_seconds",
            "End-to-end request latency",
            labelnames=["mode", "status"],
            registry=registry,
        )
        self.cache_hits_total = Counter(
            "cache_hits_total",
            "Number of cache hits by caching layer",
            labelnames=["layer"],
            registry=registry,
        )
        self.retriever_errors_total = Counter(
            "retriever_errors_total",
            "Number of retriever errors by source",
            labelnames=["source"],
            registry=registry,
        )
        self.llm_tokens_total = Counter(
            "llm_tokens_total",
            "Total LLM tokens consumed by model",
            labelnames=["model"],
            registry=registry,
        )
        self.active_requests = Gauge(
            "active_requests",
            "Number of requests currently being processed",
            registry=registry,
        )


def get_metrics() -> RequestMetrics:
    """Return the process-wide RequestMetrics singleton."""
    global _instance
    if _instance is None:
        _instance = RequestMetrics()
    return _instance
