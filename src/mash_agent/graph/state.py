"""Data contracts for the supervisor graph."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from mash_agent.agents.models import Finding, SourceDoc, SpecialistResult, SubTask, Usage

AgentName = Literal["literature", "trials", "regulatory"]
ALL_AGENTS: tuple[AgentName, ...] = ("literature", "trials", "regulatory")


class PlannedTask(BaseModel):
    agent: AgentName = Field(description="Which specialist should handle this sub-task.")
    focus: str = Field(description="What that specialist should find out, in plain language.")


class Plan(BaseModel):
    tasks: list[PlannedTask] = Field(description="Sub-tasks; at most one or two per specialist.")


class AgentOutcome(BaseModel):
    """One specialist run, successful or not. Failures are data, never exceptions."""

    agent: AgentName
    task: SubTask
    status: Literal["ok", "failed"]
    attempts: int
    latency_s: float
    result: SpecialistResult | None = None
    error: str | None = None
    retry_errors: list[str] = Field(
        default_factory=list, description="Errors from failed attempts."
    )


class SpecialistInput(TypedDict):
    task: PlannedTask


class GraphState(TypedDict, total=False):
    question: str
    plan: Plan
    notes: Annotated[list[str], operator.add]
    outcomes: Annotated[list[AgentOutcome], operator.add]


class RunReport(BaseModel):
    """Per-agent operational summary (extended with cost in the observability milestone)."""

    rows: list[AgentOutcome]
    usage: Usage
    total_attempts: int
    failed_agents: list[str]


RunStatus = Literal["ok", "partial", "failed"]


class SupervisorResult(BaseModel):
    question: str
    plan: Plan
    status: RunStatus
    notes: list[str]
    outcomes: list[AgentOutcome]

    @property
    def findings(self) -> list[Finding]:
        return [f for o in self.outcomes if o.result for f in o.result.findings]

    @property
    def sources(self) -> dict[str, SourceDoc]:
        return {s.source_id: s for o in self.outcomes if o.result for s in o.result.sources}

    @property
    def report(self) -> RunReport:
        usage = Usage()
        for o in self.outcomes:
            if o.result:
                usage += o.result.usage
        return RunReport(
            rows=self.outcomes,
            usage=usage,
            total_attempts=sum(o.attempts for o in self.outcomes),
            failed_agents=[o.agent for o in self.outcomes if o.status == "failed"],
        )
