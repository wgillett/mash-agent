"""Records every LLM and tool call of a run, including failed ones, and instruments them.

Wrappers here are the single place where calls are attributed to a stage, timed, traced, and
priced, so agent code stays free of bookkeeping.
"""

import json
import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic import BaseModel as _BaseModel  # noqa: F401  (keeps import order stable for ruff)

from mash_agent.agents.llm import Generated, StructuredLLM, StructuredOutputError
from mash_agent.agents.models import Usage
from mash_agent.agents.tools import ToolCaller
from mash_agent.mcp_servers.client import ToolError
from mash_agent.observability.context import current_meter, current_stage
from mash_agent.observability.pricing import Price, cost_usd, load_prices
from mash_agent.observability.tracing import mark_error, span


class LlmCall(BaseModel):
    stage: str
    schema_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_s: float
    ok: bool
    error: str | None = None


class ToolCallRecord(BaseModel):
    stage: str
    tool: str
    latency_s: float
    ok: bool
    error: str | None = None


class Metering(BaseModel):
    llm_calls: list[LlmCall] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class UsageMeter:
    def __init__(self) -> None:
        self.data = Metering()

    def record_llm(self, call: LlmCall) -> None:
        self.data.llm_calls.append(call)

    def record_tool(self, call: ToolCallRecord) -> None:
        self.data.tool_calls.append(call)


class InstrumentedLLM:
    """Wraps a ``StructuredLLM``: per-call span, latency, tokens, cost, stage attribution."""

    def __init__(self, inner: StructuredLLM, prices: dict[str, Price] | None = None) -> None:
        self._inner = inner
        self._prices = prices if prices is not None else load_prices()
        self.model: str = getattr(inner, "model", "unknown")

    async def generate[T: BaseModel](
        self, schema: type[T], *, system: str, user: str
    ) -> Generated[T]:
        stage = current_stage()
        start = time.monotonic()
        usage = Usage()
        error: str | None = None
        with span(
            "llm.generate",
            **{
                "gen_ai.request.model": self.model,
                "mash.stage": stage,
                "mash.schema": schema.__name__,
            },
        ) as sp:
            try:
                out = await self._inner.generate(schema, system=system, user=user)
                usage = out.usage
                return out
            except StructuredOutputError as exc:
                usage, error = exc.usage, f"{type(exc).__name__}: {exc}"
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            except BaseException:  # cancelled, e.g. by a timeout
                error = "cancelled"
                raise
            finally:
                latency = time.monotonic() - start
                sp.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
                sp.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
                sp.set_attribute("gen_ai.usage.cache_read_tokens", usage.cache_read_tokens)
                sp.set_attribute("gen_ai.usage.cache_creation_tokens", usage.cache_creation_tokens)
                cost = cost_usd(usage, self.model, self._prices)
                if cost is not None:
                    sp.set_attribute("mash.cost_usd", cost)
                if error:
                    mark_error(sp, error)
                meter = current_meter()
                if meter is not None:
                    meter.record_llm(
                        LlmCall(
                            stage=stage,
                            schema_name=schema.__name__,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                            cache_read_tokens=usage.cache_read_tokens,
                            cache_creation_tokens=usage.cache_creation_tokens,
                            latency_s=latency,
                            ok=error is None,
                            error=error,
                        )
                    )


def instrument_tools(inner: ToolCaller) -> ToolCaller:
    """Wrap a ``ToolCaller`` with a span, latency and success/failure record per call."""

    async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        stage = current_stage()
        start = time.monotonic()
        error: str | None = None
        with span(
            "tool.call",
            **{"mash.tool": tool, "mash.stage": stage, "mash.args": json.dumps(arguments)[:300]},
        ) as sp:
            try:
                return await inner(tool, arguments)
            except ToolError as exc:
                error = f"ToolError: {exc}"
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            except BaseException:
                error = "cancelled"
                raise
            finally:
                if error:
                    mark_error(sp, error)
                meter = current_meter()
                if meter is not None:
                    meter.record_tool(
                        ToolCallRecord(
                            stage=stage,
                            tool=tool,
                            latency_s=time.monotonic() - start,
                            ok=error is None,
                            error=error,
                        )
                    )

    return call
