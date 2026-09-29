"""Timeout + retry-with-backoff wrapper that turns exceptions into data."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from mash_agent.observability.tracing import span


@dataclass
class Attempted[T]:
    value: T | None
    attempts: int
    latency_s: float
    error: str | None = None
    retry_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


async def run_resilient[T](
    fn: Callable[[], Awaitable[T]],
    *,
    timeout_s: float,
    max_attempts: int,
    backoff_s: float,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Attempted[T]:
    """Call ``fn`` up to ``max_attempts`` times, each bounded by ``timeout_s``.

    Backoff doubles per retry. Never raises (except cancellation): the caller gets the last error
    in ``Attempted.error`` so one failing agent cannot sink the run.
    """
    start = time.monotonic()
    retry_errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            with span("attempt", **{"mash.attempt": attempt, "mash.max_attempts": max_attempts}):
                async with asyncio.timeout(timeout_s):
                    value = await fn()
            return Attempted(value, attempt, time.monotonic() - start, None, retry_errors)
        except Exception as exc:
            message = (
                describe(exc)
                if not isinstance(exc, TimeoutError)
                else f"timeout after {timeout_s}s"
            )
            if attempt == max_attempts:
                return Attempted(None, attempt, time.monotonic() - start, message, retry_errors)
            retry_errors.append(message)
            await sleep(backoff_s * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # pragma: no cover
