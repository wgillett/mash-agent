"""The eval runner end to end, with scripted models and recorded API responses."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from mash_agent.agents.critic import ClaimVerdict, SourceVerdicts
from mash_agent.agents.models import ExtractedFindings
from mash_agent.agents.tools import ToolCaller
from mash_agent.evals import runner as runner_module
from mash_agent.evals.judge import JudgedSource, JudgedVerdict
from mash_agent.evals.questions import Question, load_questions, select
from mash_agent.evals.report import EvalReport, compare
from mash_agent.evals.runner import run_eval, spotcheck_sample, write_artifacts
from mash_agent.graph.state import Plan, PlannedTask
from mash_agent.graph.supervisor import SupervisorConfig
from mash_agent.graph.workflow import Workflow
from tests.fakes import FunctionLLM
from tests.test_supervisor import no_sleep
from tests.test_workflow import handler_with

Handler = Callable[[type[BaseModel], str, str], BaseModel]

QUESTIONS = [
    Question(
        id="label",
        category="regulatory",
        question="What is in the label?",
        coverage_terms=["Summary", "absent-term"],
    ),
    Question(
        id="pipeline",
        category="trials",
        question="What is in the pipeline?",
        coverage_terms=["Summary"],
    ),
    Question(
        id="off-topic",
        category="safety",
        question="What is the capital of France?",
        expect_no_claims=True,
    ),
]


def system_handler(critic_passes_bad: bool = False, bad: bool = True) -> Handler:
    """Plans around the question text; off-topic gets no findings; optional BAD literature claim."""
    base = handler_with(bad_lit_claim=bad, critic_rejects_all=False)

    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is Plan:
            return Plan(
                tasks=[
                    PlannedTask(agent="literature", focus=user),
                    PlannedTask(agent="trials", focus=user),
                    PlannedTask(agent="regulatory", focus=user),
                ]
            )
        if schema is ExtractedFindings and "France" in user:
            return ExtractedFindings(findings=[])
        if schema is SourceVerdicts and critic_passes_bad:  # a critic that misses the bad claim
            n = len(re.findall(r"^\d+\. ", user, flags=re.M))
            return SourceVerdicts(
                verdicts=[
                    ClaimVerdict(claim_number=i, verdict="supported", reason="ok")
                    for i in range(1, n + 1)
                ]
            )
        return base(schema, system, user)

    return handler


def judge_handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
    assert schema is JudgedSource
    claims = re.findall(r"^(\d+)\. (.*)$", user, flags=re.M)
    return JudgedSource(
        verdicts=[
            JudgedVerdict(
                claim_number=int(n), label="unsupported" if "BAD" in t else "supported", reason="j"
            )
            for n, t in claims
        ]
    )


def llms(handler: Handler | None = None) -> tuple[FunctionLLM, FunctionLLM]:
    system = FunctionLLM(handler or system_handler())
    system.model = "claude-sonnet-5-5"  # type: ignore[attr-defined]
    judge = FunctionLLM(judge_handler)
    judge.model = "claude-opus-5-5"  # type: ignore[attr-defined]
    return system, judge


async def evaluate(all_tools: ToolCaller, handler: Handler | None = None, **kw: Any):  # type: ignore[no-untyped-def]
    system, judge = llms(handler)
    kw.setdefault("canary", False)
    return await run_eval(
        QUESTIONS,
        llm=system,
        judge_llm=judge,
        tools=all_tools,
        config=SupervisorConfig(backoff_s=0.0),
        sleep=no_sleep,
        now=lambda: datetime(2026, 9, 29, tzinfo=UTC),
        **kw,
    )


# ---- the real question set -------------------------------------------------------------------


def test_the_committed_question_set_is_valid() -> None:
    qs = load_questions(Path("evals/questions.yaml"))
    assert 10 <= len(qs) <= 20
    assert len({q.id for q in qs}) == len(qs) and all(q.question.strip() for q in qs)
    assert sum(q.expect_no_claims for q in qs) == 1
    assert {"regulatory", "trials", "literature", "safety"} <= {q.category for q in qs}


def test_select_filters_limits_and_rejects_unknown_ids() -> None:
    assert [q.id for q in select(QUESTIONS, ["pipeline"], None)] == ["pipeline"]
    assert [q.id for q in select(QUESTIONS, None, 2)] == ["label", "pipeline"]
    with pytest.raises(ValueError, match="unknown question id"):
        select(QUESTIONS, ["nope"], None)


# ---- scoring ---------------------------------------------------------------------------------


async def test_critic_catches_what_the_judge_flags(all_tools: ToolCaller) -> None:
    report, artifacts = await evaluate(all_tools)
    a, c = report.aggregate, report.aggregate.claims
    assert a.runs == 3 and a.completed.num == 3
    # 2 on-topic questions x 4 proposed claims (3 good + 1 BAD); the off-topic one proposes none
    assert c.proposed == 8 and c.judged == 8
    assert (c.unsupported_before.num, c.unsupported_before.den) == (2, 8)
    assert (c.unsupported_after.num, c.unsupported_after.den) == (0, 6)
    assert (c.unsupported_caught, c.unsupported_missed) == (2, 0)
    assert c.critic_recall.value == 1.0 and c.critic_false_reject.num == 0
    assert c.citation_accuracy_strict.value == 1.0
    assert (a.integrity_ok.num, a.integrity_ok.den) == (2, 2)
    assert (a.expectations.num, a.expectations.den) == (1, 1)  # off-topic verified nothing
    assert (a.coverage.num, a.coverage.den) == (2, 3)  # 3 terms in briefings; one is absent
    assert set(artifacts.results) == {"label", "pipeline", "off-topic"}
    assert report.judge_cost_usd is not None and report.judge_cost_usd > 0
    assert report.config.system_model == "claude-sonnet-5-5"


async def test_a_critic_miss_shows_up_as_residual_error_after_the_critic(
    all_tools: ToolCaller,
) -> None:
    report, _ = await evaluate(all_tools, system_handler(critic_passes_bad=True))
    c = report.aggregate.claims
    assert (c.unsupported_before.num, c.unsupported_before.den) == (2, 8)
    assert (c.unsupported_after.num, c.unsupported_after.den) == (2, 8)  # nothing was excluded
    assert (c.unsupported_caught, c.unsupported_missed) == (0, 2)
    assert c.critic_recall.value == 0.0
    assert c.citation_accuracy_strict.value == 0.75
    # structure is unaffected: the guarantee is about critic-passed claims, not judge agreement
    assert report.aggregate.integrity_ok.value == 1.0


async def test_variant_text_reaches_the_extraction_prompt_and_is_recorded(
    all_tools: ToolCaller,
) -> None:
    system, judge = llms()
    for variant, expect in (("baseline", False), ("source-terms", True)):
        system.calls.clear()
        report, _ = await run_eval(
            QUESTIONS[:1],
            llm=system,
            judge_llm=judge,
            tools=all_tools,
            variant=variant,
            canary=False,
            config=SupervisorConfig(backoff_s=0.0),
            sleep=no_sleep,
        )
        prompts = [s for name, s in system.calls if name == "ExtractedFindings"]
        assert prompts and all(("Use the source's own terminology" in p) == expect for p in prompts)
        assert report.config.variant == variant


async def test_unknown_variant_is_rejected(all_tools: ToolCaller) -> None:
    system, judge = llms()
    with pytest.raises(ValueError, match="unknown variant"):
        await run_eval(QUESTIONS, llm=system, judge_llm=judge, tools=all_tools, variant="nope")


async def test_canary_is_aggregated_when_enabled(all_tools: ToolCaller) -> None:
    def critic_rejects_everything_corrupted(
        schema: type[BaseModel], system: str, user: str
    ) -> BaseModel:
        return system_handler()(schema, system, user)

    report, _ = await evaluate(all_tools, critic_rejects_everything_corrupted, canary=True)
    assert report.canary is not None
    assert report.canary.miss_rate.den > 0 and set(report.canary.stats) >= {"add_overclaim"}
    assert any("canary calls are billed" in n for n in report.notes)


# ---- resilience ------------------------------------------------------------------------------


async def test_a_crashed_question_is_reported_not_hidden(
    all_tools: ToolCaller, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Flaky(Workflow):
        async def run(self, question: str, decide: Any) -> Any:
            if "pipeline" in question:
                raise RuntimeError("boom")
            return await super().run(question, decide)

    monkeypatch.setattr(runner_module, "Workflow", Flaky)
    report, artifacts = await evaluate(all_tools)
    by_id = {r.question_id: r for r in report.runs}
    assert by_id["pipeline"].status == "crashed" and by_id["pipeline"].error == "RuntimeError: boom"
    assert by_id["label"].status == "approved"
    assert report.aggregate.completed.num == 2 and report.aggregate.completed.den == 3
    assert "pipeline" not in artifacts.results


async def test_judge_failure_leaves_claims_unjudged_instead_of_guessing(
    all_tools: ToolCaller,
) -> None:
    system = FunctionLLM(system_handler())
    system.model = "claude-sonnet-5-5"  # type: ignore[attr-defined]

    def broken(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        raise RuntimeError("judge down")

    report, _ = await run_eval(
        QUESTIONS[:1],
        llm=system,
        judge_llm=FunctionLLM(broken),
        tools=all_tools,
        canary=False,
        config=SupervisorConfig(backoff_s=0.0),
        sleep=no_sleep,
    )
    c = report.aggregate.claims
    assert (c.proposed, c.judged, c.unjudged) == (4, 0, 4)
    assert c.unsupported_before.value is None  # no data, not 0%
    assert any("judge failed" in n for n in report.notes)


# ---- artifacts and comparison ----------------------------------------------------------------


async def test_artifacts_round_trip_and_spotcheck_prioritises_disagreements(
    all_tools: ToolCaller, tmp_path: Path
) -> None:
    report, artifacts = await evaluate(all_tools)
    paths = write_artifacts(report, artifacts, tmp_path, spotcheck_n=5)
    loaded = EvalReport.model_validate_json(paths["report"].read_text())
    assert loaded.aggregate.claims.proposed == 8 and loaded.config.variant == "baseline"
    assert "Unsupported before the critic" in paths["summary"].read_text()
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [
        "label.json",
        "off-topic.json",
        "pipeline.json",
    ]
    rows = [json.loads(line) for line in paths["spotcheck"].read_text().splitlines()]
    assert len(rows) == 5 and all(r["human_label"] is None and r["source_text"] for r in rows)
    assert [r["critic"] for r in rows[:2]] == ["unsupported", "unsupported"]  # BAD claims first
    assert spotcheck_sample(report, artifacts, 5) == rows  # deterministic


async def test_compare_uses_an_interval_for_the_difference_not_ci_overlap(
    all_tools: ToolCaller,
) -> None:
    from mash_agent.evals.metrics import Rate
    from mash_agent.evals.report import newcombe_difference

    base, _ = await evaluate(all_tools)

    def row(text: str, name: str) -> str:
        return next(line for line in text.splitlines() if line.startswith(f"| {name}"))

    def variant_with(num: int, den: int, ref: tuple[int, int]) -> tuple[EvalReport, EvalReport]:
        a, b = base.model_copy(deep=True), base.model_copy(deep=True)
        a.aggregate.claims.exclusion_rate = Rate(num=ref[0], den=ref[1])
        b.aggregate.claims.exclusion_rate = Rate(num=num, den=den)
        return a, b

    name = "claims excluded by critic"

    # identical reports: no difference
    same = compare(base, base)
    assert "not distinguishable from noise" in same and "different question sets" not in same

    # a large shift is unlikely to be chance
    a, b = variant_with(40, 200, (2, 200))
    assert "unlikely to be chance" in row(compare(a, b), name)

    # one extra claim out of the same 200 is noise
    a, b = variant_with(3, 200, (2, 200))
    assert "not distinguishable from noise" in row(compare(a, b), name)

    # The reason for not using CI overlap: 40/200 vs 22/200 have overlapping Wilson intervals
    # (about 15-26% and 7.4-16%) yet the difference is unlikely to be chance.
    r0, r1 = Rate(num=40, den=200), Rate(num=22, den=200)
    assert r1.ci_high is not None and r0.ci_low is not None and r1.ci_high > r0.ci_low
    diff = newcombe_difference(r0, r1)
    assert diff is not None and diff[2] < 0  # upper bound below zero
    a, b = variant_with(22, 200, (40, 200))
    assert "unlikely to be chance" in row(compare(a, b), name)

    # borderline: halving 19/259 to 9/259 does not clearly exclude zero
    diff = newcombe_difference(Rate(num=19, den=259), Rate(num=9, den=259))
    assert diff is not None and diff[1] < -0.05 and diff[2] > 0

    assert newcombe_difference(Rate(num=0, den=0), Rate(num=1, den=2)) is None
    other = base.model_copy(deep=True)
    other.config.question_ids = ["x"]
    assert "different question sets" in compare(base, other)


async def test_variant_names_include_the_commentary_variant() -> None:
    from mash_agent.agents.prompts import EXTRACT_VARIANTS

    assert set(EXTRACT_VARIANTS) == {"baseline", "source-terms", "quote-anchored", "no-commentary"}
    assert EXTRACT_VARIANTS["baseline"] == ""
    assert "positive statement" in EXTRACT_VARIANTS["no-commentary"]
