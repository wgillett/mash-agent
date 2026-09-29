"""Run-scoped context: which pipeline stage is running and where to record usage.

Context variables propagate into asyncio tasks (including LangGraph's parallel node tasks), so
an LLM call made deep inside a specialist is attributed to that specialist without threading
arguments through every function.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mash_agent.observability.meter import UsageMeter

UNATTRIBUTED = "unattributed"

_stage: ContextVar[str] = ContextVar("mash_stage", default=UNATTRIBUTED)
_meter: ContextVar["UsageMeter | None"] = ContextVar("mash_meter", default=None)


def current_stage() -> str:
    return _stage.get()


def current_meter() -> "UsageMeter | None":
    return _meter.get()


@contextmanager
def stage(name: str) -> Iterator[None]:
    token = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(token)


@contextmanager
def metering(meter: "UsageMeter") -> Iterator[None]:
    token = _meter.set(meter)
    try:
        yield
    finally:
        _meter.reset(token)
