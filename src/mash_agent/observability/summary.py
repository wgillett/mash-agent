"""Per-run summary: tokens, cost and latency by stage, agent and tool."""

from collections import defaultdict

from pydantic import BaseModel, Field

from mash_agent.agents.models import Usage
from mash_agent.graph.state import AgentOutcome
from mash_agent.observability.meter import LlmCall, Metering
from mash_agent.observability.pricing import Price, cost_usd, load_prices


class StageRow(BaseModel):
    stage: str
    calls: int
    failed_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    latency_s: float = Field(description="Summed LLM call time; parallel calls overlap.")


class AgentRow(BaseModel):
    agent: str
    focus: str
    status: str
    attempts: int
    latency_s: float
    findings: int
    error: str | None = None


class ToolRow(BaseModel):
    tool: str
    calls: int
    failed_calls: int
    latency_s: float


class RunSummary(BaseModel):
    run_id: str
    model: str
    status: str
    agents_status: str
    wall_time_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = Field(description="None if any stage's model has no known price.")
    llm_calls: int
    failed_llm_calls: int
    retries: int
    failed_agents: list[str]
    stages: list[StageRow]
    agents: list[AgentRow]
    tools: list[ToolRow]


def _stage_rank(stage: str) -> tuple[int, str]:
    if stage == "planner":
        return (0, stage)
    if stage.startswith("specialist:"):
        return (1, stage)
    return ({"critic": 2, "synthesis": 3}.get(stage, 4), stage)


def build_summary(
    *,
    run_id: str,
    model: str,
    status: str,
    agents_status: str,
    wall_time_s: float,
    outcomes: list[AgentOutcome],
    metering: Metering,
    prices: dict[str, Price] | None = None,
) -> RunSummary:
    prices = prices if prices is not None else load_prices()
    by_stage: dict[str, list[LlmCall]] = defaultdict(list)
    for call in metering.llm_calls:
        by_stage[call.stage].append(call)

    stages: list[StageRow] = []
    for stage in sorted(by_stage, key=_stage_rank):
        calls = by_stage[stage]
        usage = Usage(
            input_tokens=sum(c.input_tokens for c in calls),
            output_tokens=sum(c.output_tokens for c in calls),
        )
        stages.append(
            StageRow(
                stage=stage,
                calls=len(calls),
                failed_calls=sum(1 for c in calls if not c.ok),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost_usd(usage, model, prices),
                latency_s=sum(c.latency_s for c in calls),
            )
        )

    tool_stats: dict[str, ToolRow] = {}
    for t in metering.tool_calls:
        row = tool_stats.setdefault(
            t.tool, ToolRow(tool=t.tool, calls=0, failed_calls=0, latency_s=0)
        )
        row.calls += 1
        row.failed_calls += 0 if t.ok else 1
        row.latency_s += t.latency_s

    costs = [s.cost_usd for s in stages]
    total_cost = None if any(c is None for c in costs) else sum(c for c in costs if c is not None)
    return RunSummary(
        run_id=run_id,
        model=model,
        status=status,
        agents_status=agents_status,
        wall_time_s=wall_time_s,
        input_tokens=sum(s.input_tokens for s in stages),
        output_tokens=sum(s.output_tokens for s in stages),
        cost_usd=total_cost if stages else 0.0,
        llm_calls=sum(s.calls for s in stages),
        failed_llm_calls=sum(s.failed_calls for s in stages),
        retries=sum(o.attempts - 1 for o in outcomes),
        failed_agents=[o.agent for o in outcomes if o.status == "failed"],
        stages=stages,
        agents=[
            AgentRow(
                agent=o.agent,
                focus=o.task.focus,
                status=o.status,
                attempts=o.attempts,
                latency_s=o.latency_s,
                findings=len(o.result.findings) if o.result else 0,
                error=o.error,
            )
            for o in outcomes
        ],
        tools=list(tool_stats.values()),
    )
