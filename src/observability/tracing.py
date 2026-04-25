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
    from opentelemetry import context as otel_context
    from opentelemetry import trace as _otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.trace import StatusCode

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

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: object) -> None:
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


def setup_tracing(service_name: str | None = None) -> object:
    """
    Configure and register a TracerProvider.

    Returns a real ``TracerProvider`` when opentelemetry-sdk is installed;
    otherwise logs a warning and returns ``None``.

    Config via environment:
      - OTEL_SERVICE_NAME (fallback: *service_name* arg, then "agentic-rag")
      - OTEL_EXPORTER_OTLP_ENDPOINT → use OTLP/gRPC exporter
      - OTEL_DEV_MODE=true → SimpleSpanProcessor (flush-on-end, good for dev)
    """
    if not _OTEL_AVAILABLE:
        logger.warning(
            "tracing_disabled",
            reason="opentelemetry-sdk not installed; install with: pip install 'agentic-rag[observability]'",
        )
        return None

    svc = os.environ.get("OTEL_SERVICE_NAME") or service_name or "agentic-rag"
    resource = Resource.create({"service.name": svc})
    provider = TracerProvider(resource=resource)

    dev_mode = os.environ.get("OTEL_DEV_MODE", "").lower() in ("1", "true", "yes")

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

    processor = SimpleSpanProcessor(exporter) if dev_mode else BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
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


def get_current_context() -> object | None:
    """Return the current OTel context for propagation across async boundaries.

    Returns ``None`` when opentelemetry is not installed.
    """
    if not _OTEL_AVAILABLE:
        return None
    return otel_context.get_current()


def attach_context(ctx: object) -> object | None:
    """Attach an OTel context (e.g. in a child asyncio task).

    Returns a token to pass to :func:`detach_context`, or ``None`` when
    opentelemetry is not installed.
    """
    if not _OTEL_AVAILABLE or ctx is None:
        return None
    return otel_context.attach(ctx)  # type: ignore[arg-type]


def detach_context(token: object) -> None:
    """Detach a previously attached context."""
    if not _OTEL_AVAILABLE or token is None:
        return
    otel_context.detach(token)  # type: ignore[arg-type]


def set_span_ok(span: object) -> None:
    """Mark a span as OK (no-op when OTel is not installed)."""
    if _OTEL_AVAILABLE and hasattr(span, "set_status"):
        span.set_status(StatusCode.OK)  # type: ignore[attr-defined]


def set_span_error(span: object, exc: BaseException) -> None:
    """Record an exception and mark a span as ERROR."""
    if _OTEL_AVAILABLE and hasattr(span, "set_status"):
        span.record_exception(exc)  # type: ignore[attr-defined]
        span.set_status(StatusCode.ERROR, str(exc))  # type: ignore[attr-defined]
