"""Full workflow: plan -> specialists -> critic -> synthesis -> human approval gate."""

import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from mash_agent.agents.critic import CheckedFinding, Critic, CriticReport
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import Finding, SourceDoc, SpecialistResult, SubTask, Usage
from mash_agent.agents.synthesis import Briefing, Bullet, Section, Synthesizer
from mash_agent.agents.tools import ToolCaller
from mash_agent.briefing import render_markdown
from mash_agent.graph.state import (
    AgentOutcome,
    Decision,
    GraphState,
    Plan,
    PlannedTask,
    RunStatus,
    SpecialistInput,
)
from mash_agent.graph.supervisor import Supervisor, SupervisorConfig
from mash_agent.observability.context import metering as metering_context
from mash_agent.observability.context import stage
from mash_agent.observability.meter import InstrumentedLLM, Metering, UsageMeter, instrument_tools
from mash_agent.observability.summary import RunSummary, build_summary
from mash_agent.observability.tracing import mark_error, span

WorkflowStatus = Literal["approved", "rejected", "no_verified_claims"]

# Pydantic models that live in graph state and are therefore (de)serialized by the checkpointer.
# Listed explicitly so LangGraph's msgpack allowlist stays strict; a test fails if one is missing.
CHECKPOINT_TYPES: tuple[tuple[str, str], ...] = tuple(
    (cls.__module__, cls.__name__)
    for cls in (
        Plan,
        PlannedTask,
        AgentOutcome,
        SubTask,
        SpecialistResult,
        SourceDoc,
        Finding,
        Usage,
        CriticReport,
        CheckedFinding,
        Briefing,
        Section,
        Bullet,
        Decision,
    )
)


class ApprovalRequest(BaseModel):
    """What the human sees at the gate."""

    question: str
    markdown: str
    verified_claims: int
    excluded_claims: int
    limitations: list[str]


Decide = Callable[[ApprovalRequest], Awaitable[Decision]]


class WorkflowResult(BaseModel):
    question: str
    status: WorkflowStatus
    agents_status: RunStatus
    plan: Plan
    outcomes: list[AgentOutcome]
    critic: CriticReport | None = None
    briefing: Briefing | None = None
    markdown: str | None = None
    limitations: list[str] = Field(default_factory=list)
    decision: Decision | None = None
    notes: list[str] = Field(default_factory=list)
    run_id: str = ""
    model: str = "unknown"
    wall_time_s: float = 0.0
    metering: Metering = Field(default_factory=Metering)
    summary: RunSummary | None = None

    @property
    def total_usage(self) -> Usage:
        """All LLM tokens of the run, including calls that failed or were retried."""
        return Usage(
            input_tokens=sum(c.input_tokens for c in self.metering.llm_calls),
            output_tokens=sum(c.output_tokens for c in self.metering.llm_calls),
        )


SCOPE_LABELS = {
    "literature": "PubMed query",
    "trials": "ClinicalTrials.gov (Phase 2/3) condition query",
    "regulatory": "openFDA label lookup for",
}


def build_scope(outcomes: list[AgentOutcome]) -> list[str]:
    """What each specialist actually searched, so readers can see the briefing's boundaries."""
    items: list[str] = []
    for o in outcomes:
        if o.result:
            r = o.result
            items.append(
                f"{SCOPE_LABELS[o.agent]}: {'; '.join(r.queries)} "
                f"({len(r.sources)} sources retrieved, {len(r.findings)} claims proposed)"
            )
    return items


def build_limitations(
    outcomes: list[AgentOutcome], critic: CriticReport, notes: list[str]
) -> list[str]:
    items: list[str] = []
    for o in outcomes:
        if o.status == "failed":
            items.append(
                f"The {o.agent} specialist failed after {o.attempts} attempts ({o.error}); "
                "that area may be missing."
            )
    dropped = sum(len(o.result.dropped) for o in outcomes if o.result)
    if dropped:
        items.append(
            f"Specialists dropped {dropped} proposed finding(s) (unknown source or per-agent cap)."
        )
    total = len(critic.checked)
    if critic.count("unsupported"):
        items.append(
            f"The critic excluded {critic.count('unsupported')} of {total} claims as not "
            "supported by their cited source."
        )
    if critic.count("unchecked"):
        items.append(
            f"{critic.count('unchecked')} of {total} claims could not be checked and were excluded."
        )
    items.extend(notes)
    return items


class Workflow:
    def __init__(
        self,
        llm: StructuredLLM,
        tools: ToolCaller,
        config: SupervisorConfig | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        today: Callable[[], date] = date.today,
        progress: Callable[[str], None] | None = None,
        extract_addendum: str | None = None,
    ) -> None:
        cfg = config or SupervisorConfig()
        self._progress = progress or (lambda message: None)
        instrumented = InstrumentedLLM(llm)
        self.model = instrumented.model
        llm = instrumented
        tools = instrument_tools(tools)
        self._supervisor = Supervisor(
            llm, tools, cfg, sleep=sleep, progress=progress, extract_addendum=extract_addendum
        )
        self._critic = Critic(
            llm,
            timeout_s=cfg.specialist_timeout_s,
            max_attempts=cfg.max_attempts,
            backoff_s=cfg.backoff_s,
            max_parallel=cfg.critic_max_parallel,
            sleep=sleep,
        )
        self._synth = Synthesizer(
            llm,
            timeout_s=cfg.specialist_timeout_s,
            max_attempts=cfg.max_attempts,
            backoff_s=cfg.backoff_s,
            sleep=sleep,
        )
        self._cfg = cfg
        self._today = today
        self.graph = self._build()

    # ---- nodes -------------------------------------------------------------------------------

    async def _critic_node(self, state: GraphState) -> dict[str, Any]:
        outcomes = state["outcomes"]
        findings = [f for o in outcomes if o.result for f in o.result.findings]
        sources = {s.source_id: s for o in outcomes if o.result for s in o.result.sources}
        n_sources = len({f.source_id for f in findings})
        self._progress(f"critic checking {len(findings)} claims across {n_sources} sources")
        with stage("critic"), span("critic", **{"mash.claims": len(findings)}) as sp:
            report = await self._critic.check(findings, sources)
            sp.set_attribute("mash.supported", len(report.passed))
        return {"critic": report, "notes": report.notes}

    @staticmethod
    def _after_critic(state: GraphState) -> str:
        return "synthesize" if state["critic"].passed else END

    async def _synthesize_node(self, state: GraphState) -> dict[str, Any]:
        critic = state["critic"]
        outcomes = state["outcomes"]
        self._progress(f"writing briefing from {len(critic.passed)} verified claims")
        with stage("synthesis"), span("synthesis", **{"mash.claims": len(critic.passed)}):
            briefing = await self._synth.synthesize(critic.passed)
        limitations = build_limitations(
            outcomes, critic, [*state.get("notes", []), *briefing.notes]
        )
        # critic notes are already in state notes; drop duplicates while keeping order
        limitations = list(dict.fromkeys(limitations))
        sources = {s.source_id: s for o in outcomes if o.result for s in o.result.sources}
        markdown = render_markdown(
            question=state["question"],
            briefing=briefing,
            sources=sources,
            limitations=limitations,
            generated_on=self._today(),
            scope=build_scope(outcomes),
        )
        return {
            "briefing": briefing,
            "markdown": markdown,
            "limitations": limitations,
        }

    @staticmethod
    def _approval_node(state: GraphState) -> dict[str, Any]:
        critic = state["critic"]
        request = ApprovalRequest(
            question=state["question"],
            markdown=state["markdown"],
            verified_claims=len(critic.passed),
            excluded_claims=len(critic.checked) - len(critic.passed),
            limitations=state["limitations"],
        )
        answer = interrupt(request.model_dump())  # pauses the graph until resumed
        return {"decision": Decision.model_validate(answer)}

    def _build(self) -> Any:
        sup = self._supervisor
        graph = StateGraph(GraphState)
        graph.add_node("plan", sup.plan_node)
        graph.add_node("specialist", sup.specialist_node, input_schema=SpecialistInput)
        graph.add_node("critic", self._critic_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("approval", self._approval_node)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", sup.fan_out, ["specialist"])
        graph.add_edge("specialist", "critic")
        graph.add_conditional_edges("critic", self._after_critic, ["synthesize", END])
        graph.add_edge("synthesize", "approval")
        graph.add_edge("approval", END)
        serde = JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_TYPES))
        return graph.compile(checkpointer=MemorySaver(serde=serde))

    # ---- entry point -------------------------------------------------------------------------

    async def run(self, question: str, decide: Decide) -> WorkflowResult:
        run_id = uuid.uuid4().hex[:12]
        config: Any = {
            "configurable": {"thread_id": run_id},
            "max_concurrency": self._cfg.max_parallel,
        }
        meter = UsageMeter()
        start = time.monotonic()
        with (
            metering_context(meter),
            span("mash.run", **{"mash.run_id": run_id, "mash.model": self.model}),
        ):
            state = await self.graph.ainvoke(
                {"question": question, "notes": [], "outcomes": []}, config=config
            )
            while "__interrupt__" in state:
                request = ApprovalRequest.model_validate(state["__interrupt__"][0].value)
                self._progress("waiting for your approval")
                with span("approval", **{"mash.claims": request.verified_claims}) as sp:
                    decision = await decide(request)
                    sp.set_attribute("mash.approved", decision.approved)
                    if not decision.approved:
                        mark_error(sp, f"rejected: {decision.comment or 'no reason given'}")
                state = await self.graph.ainvoke(
                    Command(resume=decision.model_dump()), config=config
                )
        result = self._result(question, state)
        wall = time.monotonic() - start
        summary = build_summary(
            run_id=run_id,
            model=self.model,
            status=result.status,
            agents_status=result.agents_status,
            wall_time_s=wall,
            outcomes=result.outcomes,
            metering=meter.data,
        )
        return result.model_copy(
            update={
                "run_id": run_id,
                "model": self.model,
                "wall_time_s": wall,
                "metering": meter.data,
                "summary": summary,
            }
        )

    @staticmethod
    def _result(question: str, state: dict[str, Any]) -> WorkflowResult:
        outcomes: list[AgentOutcome] = state["outcomes"]
        failed = [o for o in outcomes if o.status == "failed"]
        agents_status: RunStatus = (
            "failed" if len(failed) == len(outcomes) else "partial" if failed else "ok"
        )
        decision: Decision | None = state.get("decision")
        status: WorkflowStatus = (
            "no_verified_claims"
            if decision is None
            else "approved"
            if decision.approved
            else "rejected"
        )
        critic: CriticReport | None = state.get("critic")
        return WorkflowResult(
            question=question,
            status=status,
            agents_status=agents_status,
            plan=state["plan"],
            outcomes=outcomes,
            critic=critic,
            briefing=state.get("briefing"),
            markdown=state.get("markdown"),
            limitations=state.get("limitations")
            or (build_limitations(outcomes, critic, state["notes"]) if critic else []),
            decision=decision,
            notes=state["notes"],
        )
