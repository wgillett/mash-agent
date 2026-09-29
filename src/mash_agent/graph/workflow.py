"""Full workflow: plan -> specialists -> critic -> synthesis -> human approval gate."""

import uuid
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from mash_agent.agents.critic import Critic, CriticReport
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import Usage
from mash_agent.agents.synthesis import Briefing, Synthesizer
from mash_agent.agents.tools import ToolCaller
from mash_agent.briefing import render_markdown
from mash_agent.graph.state import (
    AgentOutcome,
    Decision,
    GraphState,
    Plan,
    RunStatus,
    SpecialistInput,
    StageUsage,
)
from mash_agent.graph.supervisor import Supervisor, SupervisorConfig

WorkflowStatus = Literal["approved", "rejected", "no_verified_claims"]


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
    stage_usage: list[StageUsage] = Field(default_factory=list)

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for s in self.stage_usage:
            total += s.usage
        return total


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
    ) -> None:
        cfg = config or SupervisorConfig()
        self._supervisor = Supervisor(llm, tools, cfg, sleep=sleep)
        self._critic = Critic(
            llm,
            timeout_s=cfg.specialist_timeout_s,
            max_attempts=cfg.max_attempts,
            backoff_s=cfg.backoff_s,
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
        report = await self._critic.check(findings, sources)
        return {
            "critic": report,
            "notes": report.notes,
            "stage_usage": [StageUsage(stage="critic", usage=report.usage)],
        }

    @staticmethod
    def _after_critic(state: GraphState) -> str:
        return "synthesize" if state["critic"].passed else END

    async def _synthesize_node(self, state: GraphState) -> dict[str, Any]:
        critic = state["critic"]
        outcomes = state["outcomes"]
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
        )
        return {
            "briefing": briefing,
            "markdown": markdown,
            "limitations": limitations,
            "stage_usage": [StageUsage(stage="synthesis", usage=briefing.usage)],
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
        return graph.compile(checkpointer=MemorySaver())

    # ---- entry point -------------------------------------------------------------------------

    async def run(self, question: str, decide: Decide) -> WorkflowResult:
        config: Any = {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "max_concurrency": self._cfg.max_parallel,
        }
        state = await self.graph.ainvoke(
            {"question": question, "notes": [], "outcomes": [], "stage_usage": []}, config=config
        )
        while "__interrupt__" in state:
            request = ApprovalRequest.model_validate(state["__interrupt__"][0].value)
            decision = await decide(request)
            state = await self.graph.ainvoke(Command(resume=decision.model_dump()), config=config)
        return self._result(question, state)

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
            stage_usage=state["stage_usage"],
        )
