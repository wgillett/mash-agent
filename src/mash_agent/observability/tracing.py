"""OpenTelemetry tracing: instrument with the API, export spans to a JSONL file.

Instrumentation code calls ``span(...)``. Without ``configure_tracing`` it is a no-op, so the
agents run (and tests pass) with tracing off. The provider is held here rather than installed
globally, so tests and the CLI can each configure their own.
"""

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Span, StatusCode

TRACER_NAME = "mash_agent"
_provider: TracerProvider | None = None


def _attr_value(value: Any) -> Any:
    return value if isinstance(value, (str, bool, int, float)) else str(value)


def span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    ctx = span.get_span_context()
    assert ctx is not None
    start, end = span.start_time or 0, span.end_time or 0
    return {
        "trace_id": f"{ctx.trace_id:032x}",
        "span_id": f"{ctx.span_id:016x}",
        "parent_id": f"{span.parent.span_id:016x}" if span.parent else None,
        "name": span.name,
        "start_ns": start,
        "end_ns": end,
        "duration_ms": round((end - start) / 1e6, 3),
        "status": span.status.status_code.name,
        "status_description": span.status.description,
        "attributes": {k: _attr_value(v) for k, v in (span.attributes or {}).items()},
        "events": [
            {
                "name": e.name,
                "attributes": {k: _attr_value(v) for k, v in (e.attributes or {}).items()},
            }
            for e in span.events
        ],
    }


class JsonlSpanExporter(SpanExporter):
    """Appends one JSON object per finished span."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")  # one file per run

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._path.open("a") as f:
            for s in spans:
                f.write(json.dumps(span_to_dict(s)) + "\n")
        return SpanExportResult.SUCCESS


def configure_tracing(exporter: SpanExporter | None) -> TracerProvider | None:
    """Enable tracing with ``exporter`` (or disable with ``None``). Returns the provider."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
    if exporter is None:
        _provider = None
    else:
        _provider = TracerProvider()
        _provider.add_span_processor(SimpleSpanProcessor(exporter))
    return _provider


def get_tracer() -> trace.Tracer:
    return _provider.get_tracer(TRACER_NAME) if _provider else trace.NoOpTracer()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a child span of the current one; exceptions are recorded and re-raised."""
    with get_tracer().start_as_current_span(name) as sp:
        for key, value in attributes.items():
            if value is not None:
                sp.set_attribute(key, _attr_value(value))
        yield sp


def mark_error(sp: Span, description: str) -> None:
    """Flag a span as failed without an exception (failures we convert into data)."""
    sp.set_status(StatusCode.ERROR, description)
