"""Run the eval question set end to end and score it."""

import asyncio
import json
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import Usage
from mash_agent.agents.prompts import DEFAULT_EXTRACT_VARIANT, EXTRACT_VARIANTS
from mash_agent.agents.tools import ToolCaller
from mash_agent.evals.canary import CanaryReport, Miss, MutationStats, run_canary
from mash_agent.evals.judge import Judge, JudgedClaim
from mash_agent.evals.metrics import (
    ClaimRecord,
    Rate,
    RunMetrics,
    build_claim_records,
    failed_run,
    run_metrics,
)
from mash_agent.evals.questions import Question
from mash_agent.evals.report import (
    CanarySummary,
    EvalConfig,
    EvalReport,
    aggregate,
    render_markdown,
)
from mash_agent.graph.state import Decision
from mash_agent.graph.supervisor import SupervisorConfig
from mash_agent.graph.workflow import ApprovalRequest, Workflow, WorkflowResult
from mash_agent.observability.context import metering, stage
from mash_agent.observability.meter import InstrumentedLLM, UsageMeter
from mash_agent.observability.pricing import cost_usd


async def _auto_approve(request: ApprovalRequest) -> Decision:
    return Decision(approved=True, comment="eval")


class EvalArtifacts(BaseModel):
    """Full per-question results kept alongside the report (for files and spot checks)."""

    results: dict[str, WorkflowResult] = Field(default_factory=dict)


async def run_eval(
    questions: list[Question],
    *,
    llm: StructuredLLM,
    judge_llm: StructuredLLM,
    tools: ToolCaller,
    variant: str = DEFAULT_EXTRACT_VARIANT,
    canary: bool = True,
    parallel: int = 2,
    config: SupervisorConfig | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> tuple[EvalReport, EvalArtifacts]:
    if variant not in EXTRACT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {sorted(EXTRACT_VARIANTS)}")
    started_at = now()  # taken before any work, so "started" really is the start
    say = on_progress or (lambda message: None)
    system_model = getattr(llm, "model", "unknown")
    judge = InstrumentedLLM(judge_llm)
    judge_model = judge.model
    semaphore = asyncio.Semaphore(max(1, parallel))
    artifacts = EvalArtifacts()
    notes: list[str] = []
    judge_usage = Usage()
    canaries: list[CanaryReport] = []
    canary_usage = Usage()

    outputs: dict[str, tuple[RunMetrics, list[ClaimRecord]]] = {}

    async def evaluate(q: Question) -> None:
        nonlocal judge_usage, canary_usage
        async with semaphore:
            say(f"running {q.id}")
            try:
                result = await Workflow(
                    llm,
                    tools,
                    config,
                    sleep=sleep,
                    extract_addendum=EXTRACT_VARIANTS[variant],
                ).run(q.question, _auto_approve)
            except Exception as exc:  # a crash outside the graph's own isolation
                outputs[q.id] = (failed_run(q, f"{type(exc).__name__}: {exc}"), [])
                say(f"{q.id}: crashed ({type(exc).__name__})")
                return
            artifacts.results[q.id] = result

            judged: list[JudgedClaim] = []
            if result.critic and result.critic.checked:
                findings = [c.finding for c in result.critic.checked]
                sources = {
                    s.source_id: s for o in result.outcomes if o.result for s in o.result.sources
                }
                meter = UsageMeter()
                try:
                    with metering(meter), stage("judge"):
                        judged, judge_notes = await Judge(judge, sleep=sleep).judge(
                            findings, sources
                        )
                    notes.extend(f"{q.id}: {n}" for n in judge_notes)
                except Exception as exc:
                    notes.append(f"{q.id}: judging crashed ({type(exc).__name__}: {exc})")
                for call in meter.data.llm_calls:
                    judge_usage += Usage(
                        input_tokens=call.input_tokens,
                        output_tokens=call.output_tokens,
                        cache_read_tokens=call.cache_read_tokens,
                        cache_creation_tokens=call.cache_creation_tokens,
                    )
                if canary:
                    try:
                        report = await run_canary(result, llm, sleep=sleep)
                        canaries.append(report)
                        canary_usage += report.usage
                    except Exception as exc:
                        notes.append(f"{q.id}: canary crashed ({type(exc).__name__}: {exc})")

            outputs[q.id] = (
                run_metrics(q, result),
                build_claim_records(q.id, result, judged),
            )
            say(f"{q.id}: {result.status}")

    await asyncio.gather(*(evaluate(q) for q in questions))

    runs = [outputs[q.id][0] for q in questions]
    claims = [c for q in questions for c in outputs[q.id][1]]
    canary_summary = _combine_canaries(canaries) if canary and canaries else None
    if canary_summary is not None:
        canary_cost = cost_usd(canary_usage, system_model)
        notes.append(
            "canary calls are billed to the system model"
            + (f" (~${canary_cost:.2f})" if canary_cost is not None else "")
            + " and are not included in the system cost above"
        )
    report = EvalReport(
        config=EvalConfig(
            variant=variant,
            system_model=system_model,
            judge_model=judge_model,
            canary=canary,
            parallel=parallel,
            question_ids=[q.id for q in questions],
            started_at=started_at.isoformat(timespec="seconds"),
        ),
        aggregate=aggregate(runs, claims),
        runs=runs,
        claims=claims,
        canary=canary_summary,
        judge_cost_usd=cost_usd(judge_usage, judge_model),
        judge_tokens=(judge_usage.input_tokens, judge_usage.output_tokens),
        notes=notes,
    )
    return report, artifacts


def _combine_canaries(reports: list[CanaryReport]) -> CanarySummary:
    stats: dict[str, MutationStats] = defaultdict(MutationStats)
    misses: list[Miss] = []
    for r in reports:
        for name, s in r.stats.items():
            t = stats[name]
            t.n += s.n
            t.caught += s.caught
            t.unchecked += s.unchecked
            t.missed += s.missed
        misses.extend(r.misses)
    total = sum(s.n for s in stats.values())
    return CanarySummary(
        stats=dict(stats),
        miss_rate=Rate(num=sum(s.missed for s in stats.values()), den=total),
        misses=misses,
    )


def spotcheck_sample(
    report: EvalReport,
    artifacts: EvalArtifacts,
    n: int,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Claims for a person to label by hand. Disagreements and exclusions come first (they are
    the informative ones), then a random draw of the rest; deterministic for a given seed."""
    rng = random.Random(seed)
    interesting = [c for c in report.claims if c.critic != "supported" or c.judge != "supported"]
    boring = [c for c in report.claims if c not in interesting]
    rng.shuffle(interesting)
    rng.shuffle(boring)
    picked = (interesting + boring)[:n]
    rows: list[dict[str, object]] = []
    for c in picked:
        result = artifacts.results.get(c.question_id)
        source_text = ""
        if result:
            for o in result.outcomes:
                if o.result:
                    for s in o.result.sources:
                        if s.source_id == c.source_id:
                            source_text = s.text
        rows.append(
            {
                "question_id": c.question_id,
                "source_id": c.source_id,
                "claim": c.claim,
                "source_text": source_text,
                "critic": c.critic,
                "critic_reason": c.critic_reason,
                "judge": c.judge,
                "judge_reason": c.judge_reason,
                "human_label": None,  # fill in: supported | partial | unsupported
            }
        )
    return rows


def write_artifacts(
    report: EvalReport, artifacts: EvalArtifacts, out_dir: Path, spotcheck_n: int = 25
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    paths = {
        "report": out_dir / "report.json",
        "summary": out_dir / "summary.md",
        "spotcheck": out_dir / "spotcheck.jsonl",
    }
    paths["report"].write_text(report.model_dump_json(indent=2))
    paths["summary"].write_text(render_markdown(report))
    for qid, result in artifacts.results.items():
        (out_dir / "runs" / f"{qid}.json").write_text(result.model_dump_json(indent=2))
    rows = spotcheck_sample(report, artifacts, spotcheck_n)
    paths["spotcheck"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    return paths
