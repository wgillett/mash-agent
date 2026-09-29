"""Supervisor: plan -> parallel specialists (isolated failures) -> collect."""

from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from mash_agent.agents import literature, prompts, regulatory, trials
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import SubTask
from mash_agent.agents.specialist import Specialist
from mash_agent.agents.tools import ToolCaller
from mash_agent.graph.resilience import run_resilient
from mash_agent.graph.state import (
    ALL_AGENTS,
    AgentName,
    AgentOutcome,
    GraphState,
    Plan,
    PlannedTask,
    RunStatus,
    SpecialistInput,
    StageUsage,
    SupervisorResult,
)

MAX_TASKS = 6


@dataclass(frozen=True)
class SupervisorConfig:
    specialist_timeout_s: float = 120.0
    planner_timeout_s: float = 60.0
    max_attempts: int = 3
    backoff_s: float = 1.0
    max_parallel: int = 3


def default_plan(question: str) -> Plan:
    """Used when planning fails: ask every specialist the whole question."""
    return Plan(tasks=[PlannedTask(agent=a, focus=question) for a in ALL_AGENTS])


class Supervisor:
    def __init__(
        self,
        llm: StructuredLLM,
        tools: ToolCaller,
        config: SupervisorConfig | None = None,
        sleep: Any = None,
    ) -> None:
        self._llm = llm
        self._config = config or SupervisorConfig()
        self._sleep_kwargs: dict[str, Any] = {"sleep": sleep} if sleep else {}
        self._specialists: dict[AgentName, Specialist[Any]] = {
            "literature": literature.build(llm, tools),
            "trials": trials.build(llm, tools),
            "regulatory": regulatory.build(llm, tools),
        }
        self.graph = self._build()

    # ---- graph nodes -------------------------------------------------------------------------

    async def plan_node(self, state: GraphState) -> dict[str, Any]:
        question = state["question"]
        cfg = self._config
        attempt = await run_resilient(
            lambda: self._llm.generate(Plan, system=prompts.PLANNER, user=question),
            timeout_s=cfg.planner_timeout_s,
            max_attempts=cfg.max_attempts,
            backoff_s=cfg.backoff_s,
            **self._sleep_kwargs,
        )
        if attempt.value is None:
            note = f"planner failed ({attempt.error}); using default plan"
            return {"plan": default_plan(question), "notes": [note]}
        usage = [StageUsage(stage="planner", usage=attempt.value.usage)]
        tasks = [t for t in attempt.value.value.tasks if t.focus.strip()][:MAX_TASKS]
        if not tasks:
            return {
                "plan": default_plan(question),
                "notes": ["planner returned no tasks; used default plan"],
                "stage_usage": usage,
            }
        return {"plan": Plan(tasks=tasks), "stage_usage": usage}

    def fan_out(self, state: GraphState) -> list[Send]:
        return [Send("specialist", {"task": t}) for t in state["plan"].tasks]

    async def specialist_node(self, state: SpecialistInput) -> dict[str, Any]:
        task = state["task"]
        subtask = SubTask(focus=task.focus)
        specialist = self._specialists[task.agent]
        cfg = self._config
        attempt = await run_resilient(
            lambda: specialist.run(subtask),
            timeout_s=cfg.specialist_timeout_s,
            max_attempts=cfg.max_attempts,
            backoff_s=cfg.backoff_s,
            **self._sleep_kwargs,
        )
        outcome = AgentOutcome(
            agent=task.agent,
            task=subtask,
            status="ok" if attempt.ok else "failed",
            attempts=attempt.attempts,
            latency_s=attempt.latency_s,
            result=attempt.value,
            error=attempt.error,
            retry_errors=attempt.retry_errors,
        )
        usage = (
            [StageUsage(stage=f"specialist:{task.agent}", usage=outcome.result.usage)]
            if outcome.result
            else []  # tokens spent on failed attempts are not captured
        )
        return {"outcomes": [outcome], "stage_usage": usage}

    def _build(self) -> Any:
        graph = StateGraph(GraphState)
        graph.add_node("plan", self.plan_node)
        graph.add_node("specialist", self.specialist_node, input_schema=SpecialistInput)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self.fan_out, ["specialist"])
        graph.add_edge("specialist", END)
        return graph.compile()

    # ---- entry point -------------------------------------------------------------------------

    async def run(self, question: str) -> SupervisorResult:
        state = await self.graph.ainvoke(
            {"question": question, "notes": [], "outcomes": []},
            config={"max_concurrency": self._config.max_parallel},
        )
        outcomes: list[AgentOutcome] = state["outcomes"]
        failed = [o for o in outcomes if o.status == "failed"]
        status: RunStatus = (
            "failed" if len(failed) == len(outcomes) else "partial" if failed else "ok"
        )
        return SupervisorResult(
            question=question,
            plan=state["plan"],
            status=status,
            notes=state["notes"],
            outcomes=outcomes,
        )
