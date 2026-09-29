"""Metering, cost, tracing and the run summary."""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel
from rich.console import Console

from mash_agent.agents import literature
from mash_agent.agents.llm import StructuredOutputError
from mash_agent.agents.models import ExtractedFindings, Usage
from mash_agent.agents.tools import ToolCaller, in_process_caller
from mash_agent.graph.state import Decision
from mash_agent.graph.supervisor import SupervisorConfig
from mash_agent.graph.workflow import Workflow
from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed
from mash_agent.observability.context import metering, stage
from mash_agent.observability.meter import InstrumentedLLM, UsageMeter
from mash_agent.observability.pricing import PRICES_ENV_VAR, cost_usd, load_prices
from mash_agent.observability.render import (
    load_spans,
    summary_renderable,
    trace_tree,
)
from mash_agent.observability.tracing import (
    JsonlSpanExporter,
    configure_tracing,
    span_to_dict,
)
from tests.conftest import api_for
from tests.fakes import FunctionLLM
from tests.test_supervisor import QUESTION, no_sleep
from tests.test_workflow import Recorder, handler_with


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter)
    yield exporter
    configure_tracing(None)


def workflow(llm: FunctionLLM, tools: ToolCaller, model: str = "claude-sonnet-5-5") -> Workflow:
    llm.model = model  # type: ignore[attr-defined]
    cfg = SupervisorConfig(backoff_s=0.0)
    return Workflow(llm, tools, cfg, sleep=no_sleep, today=lambda: date(2026, 9, 29))


# ---- pricing ---------------------------------------------------------------------------------


def test_cost_from_default_prices_and_unknown_model() -> None:
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(usage, "claude-sonnet-5-5") == pytest.approx(12.0)  # $2 in + $10 out
    assert cost_usd(Usage(input_tokens=500_000), "claude-opus-5-5") == pytest.approx(2.0)
    assert cost_usd(usage, "some-future-model") is None  # never guess a price


def test_prices_can_be_overridden_from_the_environment() -> None:
    prices = load_prices(
        {PRICES_ENV_VAR: json.dumps({"my-model": [1, 3], "claude-sonnet-5-5": [9, 9]})}
    )
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost_usd(usage, "my-model", prices) == pytest.approx(4.0)
    assert cost_usd(usage, "claude-sonnet-5-5", prices) == pytest.approx(18.0)


# ---- the instrumented LLM --------------------------------------------------------------------


async def test_instrumented_llm_attributes_calls_to_stage_and_counts_failures() -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if user == "unusable":
            raise StructuredOutputError("bad json", Usage(input_tokens=50, output_tokens=7))
        if user == "crash":
            raise RuntimeError("api down")
        return ExtractedFindings(findings=[])

    llm = InstrumentedLLM(FunctionLLM(handler, tokens=(10, 5)))
    meter = UsageMeter()
    with metering(meter), stage("critic"):
        await llm.generate(ExtractedFindings, system="s", user="ok")
        with pytest.raises(StructuredOutputError):
            await llm.generate(ExtractedFindings, system="s", user="unusable")
        with pytest.raises(RuntimeError):
            await llm.generate(ExtractedFindings, system="s", user="crash")

    ok, unusable, crash = meter.data.llm_calls
    assert (ok.stage, ok.ok, ok.input_tokens) == ("critic", True, 10)
    # an answer that had to be retried still cost money, so its tokens are counted
    assert (unusable.ok, unusable.input_tokens, unusable.output_tokens) == (False, 50, 7)
    assert "StructuredOutputError" in (unusable.error or "")
    assert (crash.ok, crash.input_tokens) == (False, 0) and "api down" in (crash.error or "")


async def test_calls_outside_a_run_are_not_recorded_and_do_not_fail() -> None:
    llm = InstrumentedLLM(FunctionLLM(lambda s, sy, u: ExtractedFindings(findings=[])))
    await llm.generate(ExtractedFindings, system="s", user="u")  # no meter, tracing off: fine


# ---- run summary -----------------------------------------------------------------------------


async def test_summary_has_cost_latency_agents_and_tools(all_tools: ToolCaller) -> None:
    result = await workflow(FunctionLLM(handler_with()), all_tools).run(
        QUESTION, Recorder(Decision(approved=True))
    )
    s = result.summary
    assert s is not None and s.run_id == result.run_id and s.model == "claude-sonnet-5-5"
    assert (s.input_tokens, s.output_tokens) == (110, 55)
    assert s.cost_usd == pytest.approx((110 * 2 + 55 * 10) / 1e6)
    assert [r.stage for r in s.stages][0] == "planner" and s.stages[-1].stage == "synthesis"
    assert {a.agent for a in s.agents} == {"literature", "trials", "regulatory"}
    assert {t.tool: t.calls for t in s.tools} == {
        "pubmed_search": 1,
        "trials_search": 1,
        "label_search": 1,
    }
    assert s.wall_time_s > 0 and s.retries == 0 and s.failed_agents == []


async def test_unknown_model_gives_no_cost_instead_of_a_guess(all_tools: ToolCaller) -> None:
    result = await workflow(FunctionLLM(handler_with()), all_tools, model="mystery-model").run(
        QUESTION, Recorder(Decision(approved=True))
    )
    assert result.summary is not None and result.summary.cost_usd is None
    assert all(r.cost_usd is None for r in result.summary.stages)


async def test_failed_attempt_tokens_are_counted_in_the_summary(all_tools: ToolCaller) -> None:
    calls = {"n": 0}

    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is literature.PubMedQuery:
            calls["n"] += 1
            if calls["n"] == 1:
                raise StructuredOutputError("cut off", Usage(input_tokens=100, output_tokens=40))
        return handler_with()(schema, system, user)

    result = await workflow(FunctionLLM(handler), all_tools).run(
        QUESTION, Recorder(Decision(approved=True))
    )
    s = result.summary
    assert s is not None
    lit = next(r for r in s.stages if r.stage == "specialist:literature")
    assert (lit.calls, lit.failed_calls) == (3, 1)  # failed query, retried query, extraction
    assert lit.input_tokens == 100 + 10 + 10 and lit.output_tokens == 40 + 5 + 5
    assert s.retries == 1 and s.failed_llm_calls == 1


# ---- tracing ---------------------------------------------------------------------------------


async def test_trace_covers_the_agent_graph_with_correct_nesting(
    all_tools: ToolCaller, spans: InMemorySpanExporter
) -> None:
    await workflow(FunctionLLM(handler_with()), all_tools).run(
        QUESTION, Recorder(Decision(approved=True))
    )
    finished = [span_to_dict(s) for s in spans.get_finished_spans()]
    names = [s["name"] for s in finished]
    for expected in (
        "mash.run",
        "plan",
        "specialist",
        "attempt",
        "llm.generate",
        "tool.call",
        "critic",
        "critic.source",
        "synthesis",
        "approval",
    ):
        assert expected in names, expected
    assert names.count("specialist") == 3 and names.count("tool.call") == 3

    # one trace; a single root; every other span's parent exists (context reached LangGraph's
    # parallel node tasks, so nothing is orphaned)
    assert len({s["trace_id"] for s in finished}) == 1
    ids = {s["span_id"] for s in finished}
    roots = [s for s in finished if s["parent_id"] is None]
    assert [r["name"] for r in roots] == ["mash.run"]
    assert all(s["parent_id"] in ids for s in finished if s["parent_id"] is not None)

    by_id = {s["span_id"]: s for s in finished}

    def ancestors(s: dict[str, Any]) -> list[str]:
        out = []
        while s["parent_id"]:
            s = by_id[s["parent_id"]]
            out.append(s["name"])
        return out

    for tool_span in (s for s in finished if s["name"] == "tool.call"):
        assert "specialist" in ancestors(tool_span)  # tool calls hang under their specialist
    llm_stages = {s["attributes"]["mash.stage"] for s in finished if s["name"] == "llm.generate"}
    assert llm_stages == {
        "planner",
        "specialist:literature",
        "specialist:trials",
        "specialist:regulatory",
        "critic",
        "synthesis",
    }
    llm_span = next(s for s in finished if s["name"] == "llm.generate")
    assert llm_span["attributes"]["gen_ai.usage.input_tokens"] == 10
    assert llm_span["attributes"]["mash.cost_usd"] > 0


async def test_failures_are_visible_in_the_trace(
    pubmed_transport: httpx.MockTransport,
    trials_transport: httpx.MockTransport,
    spans: InMemorySpanExporter,
) -> None:
    broken = api_for(httpx.MockTransport(lambda r: httpx.Response(503)))
    broken._max_retries = 0  # noqa: SLF001
    tools = in_process_caller(
        pubmed.build_server(api_for(pubmed_transport)),
        clinicaltrials.build_server(api_for(trials_transport)),
        openfda.build_server(broken),
    )
    result = await workflow(FunctionLLM(handler_with()), tools).run(
        QUESTION, Recorder(Decision(approved=True))
    )
    assert result.summary is not None and result.summary.failed_agents == ["regulatory"]
    finished = [span_to_dict(s) for s in spans.get_finished_spans()]
    failed_tools = [s for s in finished if s["name"] == "tool.call" and s["status"] == "ERROR"]
    assert len(failed_tools) == 3  # one per attempt
    reg = next(
        s
        for s in finished
        if s["name"] == "specialist" and s["attributes"]["mash.agent"] == "regulatory"
    )
    assert reg["status"] == "ERROR" and reg["attributes"]["mash.attempts"] == 3
    attempts = [s for s in finished if s["name"] == "attempt" and s["parent_id"] == reg["span_id"]]
    assert [a["status"] for a in attempts] == ["ERROR", "ERROR", "ERROR"]
    assert result.summary.tools and any(t.failed_calls for t in result.summary.tools)


async def test_rejection_is_marked_on_the_approval_span(
    all_tools: ToolCaller, spans: InMemorySpanExporter
) -> None:
    await workflow(FunctionLLM(handler_with()), all_tools).run(
        QUESTION, Recorder(Decision(approved=False, comment="too thin"))
    )
    approval = next(s for s in spans.get_finished_spans() if s.name == "approval")
    assert approval.attributes is not None and approval.attributes["mash.approved"] is False
    assert "too thin" in (approval.status.description or "")


async def test_jsonl_trace_file_and_rendering(all_tools: ToolCaller, tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    configure_tracing(JsonlSpanExporter(path))
    try:
        result = await workflow(FunctionLLM(handler_with()), all_tools).run(
            QUESTION, Recorder(Decision(approved=True))
        )
    finally:
        configure_tracing(None)
    loaded = load_spans(path)
    assert len(loaded) > 20 and all("duration_ms" in s for s in loaded)

    console = Console(width=140, record=True, force_terminal=False)
    console.print(trace_tree(loaded))
    tree_text = console.export_text()
    assert "mash.run" in tree_text and "specialist  " in tree_text and "tool.call" in tree_text
    assert "pubmed_search" in tree_text

    assert result.summary is not None
    console = Console(width=140, record=True, force_terminal=False)
    console.print(summary_renderable(result.summary))
    text = console.export_text()
    for needle in ("Agents", "Stages (LLM)", "Tools", "literature", "specialist:trials", "$"):
        assert needle in text
