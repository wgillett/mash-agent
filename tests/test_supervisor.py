"""Supervisor graph: planning, parallel fan-out, retries, timeouts and failure isolation."""

import asyncio

import httpx
import pytest
from pydantic import BaseModel

from mash_agent.agents import literature, regulatory, trials
from mash_agent.agents.models import ExtractedFindings, RawFinding
from mash_agent.agents.tools import ToolCaller, in_process_caller
from mash_agent.graph.state import Plan, PlannedTask
from mash_agent.graph.supervisor import Supervisor, SupervisorConfig, default_plan
from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed
from tests.conftest import api_for
from tests.fakes import FunctionLLM
from tests.test_specialists import LABEL_QUOTE, NEJM_QUOTE

QUESTION = "State of late-stage MASH therapies and resmetirom label safety"
THREE_TASKS = Plan(
    tasks=[
        PlannedTask(agent="literature", focus="resmetirom evidence"),
        PlannedTask(agent="trials", focus="phase 3 pipeline"),
        PlannedTask(agent="regulatory", focus="resmetirom label safety"),
    ]
)
LABEL_ID = "LABEL:e67ea09f-a840-439c-86c8-f98585f978b2/warnings_and_cautions"


def happy(schema: type[BaseModel], system: str, user: str) -> BaseModel:
    if schema is Plan:
        return THREE_TASKS
    if schema is literature.PubMedQuery:
        return literature.PubMedQuery(query="resmetirom AND MASH")
    if schema is trials.TrialsQuery:
        return trials.TrialsQuery(condition="MASH OR NASH")
    if schema is regulatory.LabelQuery:
        return regulatory.LabelQuery(drugs=["Rezdiffra"])
    assert schema is ExtractedFindings
    if "PubMed abstracts" in system:
        f = RawFinding(claim="lit", source_id="PMID:38324483", evidence=NEJM_QUOTE)
    elif "ClinicalTrials.gov records" in system:
        f = RawFinding(claim="trial", source_id="NCT07701993", evidence="Sponsor: GlaxoSmithKline")
    else:
        f = RawFinding(claim="label", source_id=LABEL_ID, evidence=LABEL_QUOTE)
    return ExtractedFindings(findings=[f])


async def no_sleep(seconds: float) -> None:
    return None


def supervisor(llm: FunctionLLM, tools: ToolCaller, **cfg: float) -> Supervisor:
    config = SupervisorConfig(**{"backoff_s": 0.0, "max_attempts": 3, **cfg})  # type: ignore[arg-type]
    return Supervisor(llm, tools, config, sleep=no_sleep)


async def test_happy_path_merges_all_specialists(all_tools: ToolCaller) -> None:
    result = await supervisor(FunctionLLM(happy), all_tools).run(QUESTION)

    assert result.status == "ok" and result.notes == []
    assert sorted(o.agent for o in result.outcomes) == ["literature", "regulatory", "trials"]
    assert all(o.status == "ok" and o.attempts == 1 for o in result.outcomes)
    assert sorted(f.agent for f in result.findings) == ["literature", "regulatory", "trials"]
    assert all(f.evidence_verified for f in result.findings)
    assert "PMID:38324483" in result.sources and "NCT07701993" in result.sources
    report = result.report
    assert report.failed_agents == [] and report.total_attempts == 3
    # 1 plan call is not attributed to an agent; each specialist made 2 calls of (10 in, 5 out)
    assert (report.usage.input_tokens, report.usage.output_tokens) == (60, 30)


async def test_specialists_really_run_in_parallel(all_tools: ToolCaller) -> None:
    barrier = asyncio.Barrier(3)  # all three must be in flight at once, or the test times out

    async def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema in (literature.PubMedQuery, trials.TrialsQuery, regulatory.LabelQuery):
            await barrier.wait()
        return happy(schema, system, user)

    async with asyncio.timeout(5):
        result = await supervisor(FunctionLLM(handler), all_tools).run(QUESTION)
    assert result.status == "ok"


async def test_one_failing_tool_is_isolated_and_reported(
    pubmed_transport: httpx.MockTransport, trials_transport: httpx.MockTransport
) -> None:
    broken = api_for(httpx.MockTransport(lambda r: httpx.Response(503)))
    broken._max_retries = 0  # noqa: SLF001
    tools = in_process_caller(
        pubmed.build_server(api_for(pubmed_transport)),
        clinicaltrials.build_server(api_for(trials_transport)),
        openfda.build_server(broken),
    )
    result = await supervisor(FunctionLLM(happy), tools).run(QUESTION)

    assert result.status == "partial"
    by_agent = {o.agent: o for o in result.outcomes}
    assert by_agent["regulatory"].status == "failed"
    assert by_agent["regulatory"].attempts == 3 and by_agent["regulatory"].result is None
    assert "ToolError" in (by_agent["regulatory"].error or "")
    assert len(by_agent["regulatory"].retry_errors) == 2
    assert by_agent["literature"].status == by_agent["trials"].status == "ok"
    assert sorted(f.agent for f in result.findings) == ["literature", "trials"]
    assert result.report.failed_agents == ["regulatory"]


async def test_transient_llm_failure_is_retried(all_tools: ToolCaller) -> None:
    failures = {"left": 1}

    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is literature.PubMedQuery and failures["left"]:
            failures["left"] -= 1
            raise ValueError("model output did not match PubMedQuery")
        return happy(schema, system, user)

    result = await supervisor(FunctionLLM(handler), all_tools).run(QUESTION)
    lit = next(o for o in result.outcomes if o.agent == "literature")
    assert result.status == "ok" and lit.status == "ok"
    assert lit.attempts == 2 and lit.retry_errors == [
        "ValueError: model output did not match PubMedQuery"
    ]


async def test_hung_specialist_times_out_without_blocking_others(all_tools: ToolCaller) -> None:
    async def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is trials.TrialsQuery:
            await asyncio.sleep(30)
        return happy(schema, system, user)

    result = await supervisor(
        FunctionLLM(handler), all_tools, specialist_timeout_s=0.05, max_attempts=2
    ).run(QUESTION)
    by_agent = {o.agent: o for o in result.outcomes}
    assert result.status == "partial"
    assert by_agent["trials"].error == "timeout after 0.05s" and by_agent["trials"].attempts == 2
    assert by_agent["literature"].status == by_agent["regulatory"].status == "ok"


async def test_all_specialists_failing_reports_failed(all_tools: ToolCaller) -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is Plan:
            return THREE_TASKS
        raise RuntimeError("model unavailable")

    result = await supervisor(FunctionLLM(handler), all_tools).run(QUESTION)
    assert result.status == "failed" and result.findings == []
    assert result.report.failed_agents == ["literature", "trials", "regulatory"]


@pytest.mark.parametrize("bad_plan", ["raises", "empty", "blank_focus"])
async def test_planner_failure_falls_back_to_default_plan(
    all_tools: ToolCaller, bad_plan: str
) -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is Plan:
            if bad_plan == "raises":
                raise RuntimeError("planner down")
            if bad_plan == "empty":
                return Plan(tasks=[])
            return Plan(tasks=[PlannedTask(agent="trials", focus="  ")])
        return happy(schema, system, user)

    result = await supervisor(FunctionLLM(handler), all_tools).run(QUESTION)
    assert result.plan == default_plan(QUESTION)
    assert len(result.notes) == 1 and "default plan" in result.notes[0]
    assert result.status == "ok" and len(result.outcomes) == 3


async def test_plan_is_capped(all_tools: ToolCaller) -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is Plan:
            return Plan(
                tasks=[PlannedTask(agent="regulatory", focus=f"task {i}") for i in range(9)]
            )
        return happy(schema, system, user)

    result = await supervisor(FunctionLLM(handler), all_tools).run(QUESTION)
    assert len(result.plan.tasks) == 6 and len(result.outcomes) == 6


async def test_two_tasks_for_same_agent_both_run(all_tools: ToolCaller) -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is Plan:
            return Plan(
                tasks=[
                    PlannedTask(agent="literature", focus="efficacy"),
                    PlannedTask(agent="literature", focus="safety"),
                ]
            )
        return happy(schema, system, user)

    result = await supervisor(FunctionLLM(handler), all_tools).run(QUESTION)
    assert sorted(o.task.focus for o in result.outcomes) == ["efficacy", "safety"]
