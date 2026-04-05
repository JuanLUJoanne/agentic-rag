"""
OpenTelemetry tracing setup for agentic-rag.

- setup_tracing(service_name) → TracerProvider (no-op when opentelemetry not installed)
- get_tracer() → Tracer-like object

When the ``opentelemetry-sdk`` optional extra is not installed every call
returns a no-op tracer so the rest of the codebase (simple_workflow,
multi_agent_workflow, api/main) can call::

    with get_tracer().start_as_current_span("node_name"):
        ...

without modification.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import structlog

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Optional import guard
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    _OTEL_AVAILABLE = True
except ModuleNotFoundError:
    _OTEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# No-op shim used when opentelemetry is not installed
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Minimal span-like object so attribute access never raises."""

    def set_attribute(self, key: str, value: object) -> None:  # noqa: ARG002
        pass

    def record_exception(self, exc: BaseException) -> None:  # noqa: ARG002
        pass

    def set_status(self, *args: object, **kwargs: object) -> None:
        pass


class _NoOpTracer:
    """Tracer that produces no spans and incurs no overhead."""

    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: object):  # noqa: ARG002
        yield _NoOpSpan()


# ---------------------------------------------------------------------------
# Module-level tracer cache
# ---------------------------------------------------------------------------
_tracer: object = None  # real Tracer or _NoOpTracer


def setup_tracing(service_name: str) -> object:
    """
    Configure and register a TracerProvider.

    Returns a real ``TracerProvider`` when opentelemetry-sdk is installed;
    otherwise logs a warning and returns ``None``.

    If OTEL_EXPORTER_OTLP_ENDPOINT is set, export spans via OTLP/gRPC;
    otherwise fall back to ConsoleSpanExporter so the app runs without
    a collector.
    """
    if not _OTEL_AVAILABLE:
        logger.warning(
            "tracing_disabled",
            reason="opentelemetry-sdk not installed; install with: pip install 'agentic-rag[observability]'",
        )
        return None

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
    _otel_trace.set_tracer_provider(provider)

    global _tracer
    _tracer = _otel_trace.get_tracer("agentic-rag")

    return provider


def get_tracer() -> object:
    """Return (or lazily create) the module-level tracer for 'agentic-rag'.

    Returns a real OTel ``Tracer`` when opentelemetry-sdk is available,
    otherwise a :class:`_NoOpTracer` so callers need no conditional logic.
    """
    global _tracer
    if _tracer is None:
        if _OTEL_AVAILABLE:
            _tracer = _otel_trace.get_tracer("agentic-rag")
        else:
            _tracer = _NoOpTracer()
    return _tracer
