"""
OpenTelemetry tracing setup for agentic-rag.

- setup_tracing(service_name) → TracerProvider
  Configures OTLP gRPC exporter to OTEL_EXPORTER_OTLP_ENDPOINT
  (default http://localhost:4317).  Falls back to ConsoleSpanExporter if
  the env var is not set.

- get_tracer() → Tracer
  Returns a module-level tracer named "agentic-rag".
"""
from __future__ import annotations

import os

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = structlog.get_logger()

_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str) -> TracerProvider:
    """
    Configure and register a TracerProvider.

    If OTEL_EXPORTER_OTLP_ENDPOINT is set, export spans via OTLP/gRPC;
    otherwise fall back to ConsoleSpanExporter so the app runs without
    a collector.
    """
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=endpoint)
            logger.info("tracing_otlp_configured", endpoint=endpoint)
        except Exception as exc:  # pragma: no cover
            logger.warning("tracing_otlp_fallback", reason=str(exc))
            exporter = ConsoleSpanExporter()
    else:
        logger.info("tracing_console_exporter", reason="OTEL_EXPORTER_OTLP_ENDPOINT not set")
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Populate the module-level tracer
    global _tracer
    _tracer = trace.get_tracer("agentic-rag")

    return provider


def get_tracer() -> trace.Tracer:
    """Return (or lazily create) the module-level tracer for 'agentic-rag'."""
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("agentic-rag")
    return _tracer
