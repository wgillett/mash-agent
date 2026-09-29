"""Eval metrics with hand-computed expectations."""

import pytest

from mash_agent.agents.models import Finding
from mash_agent.agents.synthesis import Briefing, Bullet, Section
from mash_agent.evals.judge import Graded, JudgedClaim
from mash_agent.evals.metrics import (
    ClaimRecord,
    CriticOutcome,
    Rate,
    advice_hits,
    build_claim_records,
    claim_metrics,
    coverage,
    failed_run,
    integrity,
    run_metrics,
    wilson,
)
from mash_agent.evals.questions import Question
from mash_agent.graph.state import Decision
from mash_agent.graph.workflow import WorkflowResult
from tests.fakes import FunctionLLM
from tests.test_supervisor import QUESTION, no_sleep  # noqa: F401
from tests.test_workflow import Recorder, handler_with, workflow


def rec(
    critic: CriticOutcome, judge: Graded, quote: bool = True, qid: str = "q", agent: str = "trials"
) -> ClaimRecord:
    return ClaimRecord(
        question_id=qid,
        agent=agent,
        source_id="S",
        claim=f"{critic}/{judge}/{id(object())}",
        evidence_verified=quote,
        critic=critic,
        critic_reason="r",
        judge=judge,
        judge_reason="j",
        in_briefing=critic == "supported",
    )


# ---- rates -----------------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    assert wilson(0, 0) == (0.0, 0.0)
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.7225, abs=1e-3)


def test_rate_serializes_value_and_interval_and_handles_empty() -> None:
    r = Rate(num=3, den=10)
    data = r.model_dump()
    assert data["value"] == 0.3 and data["ci_low"] < 0.3 < data["ci_high"]
    assert "3/10" in r.text() and "30.0%" in r.text()
    empty = Rate(num=0, den=0)
    assert empty.value is None and empty.text() == "n/a (0)"


# ---- claim metrics ---------------------------------------------------------------------------


def test_claim_metrics_hand_computed() -> None:
    records = (
        [rec("supported", "supported")] * 5  # good claims the critic kept
        + [rec("supported", "partial")]  # kept, but only partly supported
        + [rec("supported", "unsupported")]  # judge-unsupported that the critic MISSED
        + [rec("unsupported", "unsupported")] * 2  # judge-unsupported the critic CAUGHT
        + [rec("unsupported", "supported")]  # a good claim the critic wrongly excluded
        + [rec("supported", "unjudged", quote=False)]  # could not be judged: excluded from rates
    )
    m = claim_metrics(records)
    assert (m.proposed, m.judged, m.unjudged) == (11, 10, 1)
    assert (m.unsupported_before.num, m.unsupported_before.den) == (3, 10)
    assert (m.not_fully_supported_before.num, m.not_fully_supported_before.den) == (4, 10)
    # reaches the briefing = critic-passed and judged: 5 + 1 + 1 = 7
    assert m.passed == 7
    assert (m.unsupported_after.num, m.unsupported_after.den) == (1, 7)
    assert (m.citation_accuracy_strict.num, m.citation_accuracy_strict.den) == (5, 7)
    assert (m.citation_accuracy_lenient.num, m.citation_accuracy_lenient.den) == (6, 7)
    assert (m.critic_recall.num, m.critic_recall.den) == (2, 3)
    assert (m.unsupported_caught, m.unsupported_missed) == (2, 1)
    assert (m.critic_false_reject.num, m.critic_false_reject.den) == (1, 6)
    assert (m.exclusion_rate.num, m.exclusion_rate.den) == (3, 11)
    assert (m.quote_verified.num, m.quote_verified.den) == (10, 11)


def test_the_critic_lowers_the_unsupported_rate() -> None:
    m = claim_metrics([rec("unsupported", "unsupported")] * 3 + [rec("supported", "supported")] * 7)
    assert m.unsupported_before.value == 0.3 and m.unsupported_after.value == 0.0
    assert m.critic_recall.value == 1.0


def test_metrics_with_no_records_do_not_divide_by_zero() -> None:
    m = claim_metrics([])
    assert m.proposed == 0 and m.unsupported_before.value is None and m.critic_recall.value is None


# ---- joining critic and judge ----------------------------------------------------------------


async def _result(all_tools, bad: bool = False) -> WorkflowResult:  # type: ignore[no-untyped-def]
    return await workflow(FunctionLLM(handler_with(bad_lit_claim=bad)), all_tools).run(
        QUESTION, Recorder(Decision(approved=True))
    )


async def test_records_join_critic_and_judge_by_source_and_claim(all_tools) -> None:  # type: ignore[no-untyped-def]
    result = await _result(all_tools, bad=True)
    assert result.critic is not None
    judged = [
        JudgedClaim(
            finding=c.finding,
            label="unsupported" if "BAD" in c.finding.claim else "supported",
            reason="j",
        )
        for c in result.critic.checked
    ]
    records = build_claim_records("q1", result, judged)
    assert len(records) == 4
    bad = next(r for r in records if "BAD" in r.claim)
    assert (bad.critic, bad.judge, bad.in_briefing) == ("unsupported", "unsupported", False)
    assert all(r.in_briefing for r in records if "BAD" not in r.claim)
    m = claim_metrics(records)
    assert (m.unsupported_caught, m.unsupported_missed) == (1, 0)


async def test_unmatched_claims_are_unjudged_not_guessed(all_tools) -> None:  # type: ignore[no-untyped-def]
    result = await _result(all_tools)
    records = build_claim_records("q1", result, [])
    assert {r.judge for r in records} == {"unjudged"}


# ---- deterministic checks --------------------------------------------------------------------


async def test_integrity_holds_for_a_normal_run_and_detects_violations(all_tools) -> None:  # type: ignore[no-untyped-def]
    result = await _result(all_tools, bad=True)
    ok = integrity(result)
    assert ok.has_briefing and ok.ok and ok.bullets == 3

    assert result.briefing is not None and result.critic is not None
    excluded: Finding = next(c.finding for c in result.critic.checked if c.outcome != "supported")
    ghost = excluded.model_copy(update={"source_id": "PMID:404", "claim": "never retrieved"})
    tampered = result.briefing.model_copy(
        update={
            "sections": [
                Section(
                    heading="x",
                    bullets=[
                        Bullet(text="uses an excluded claim", claims=[excluded]),
                        Bullet(text="cites an unretrieved source", claims=[ghost]),
                        Bullet(text="cites nothing", claims=[]),
                    ],
                )
            ]
        }
    )
    bad = integrity(result.model_copy(update={"briefing": tampered}))
    assert not bad.ok
    assert (bad.bullets, bad.bullets_without_citation) == (3, 1)
    assert bad.cited_but_not_retrieved == 1
    assert bad.not_critic_passed == 2  # the excluded claim and the ghost

    assert not integrity(result.model_copy(update={"briefing": None})).has_briefing
    assert Briefing(sections=[]).sections == []


def test_advice_detection_ignores_the_disclaimer() -> None:
    from mash_agent.briefing import DISCLAIMER

    assert advice_hits(DISCLAIMER) == []
    assert advice_hits("You should start treatment. We recommend semaglutide.") == [
        r"\byou should\b",
        r"\bwe recommend\b",
    ]
    assert advice_hits(None) == []


def test_coverage_is_case_insensitive_and_reports_misses() -> None:
    q = Question(
        id="q", category="c", question="?", coverage_terms=["Hepatotoxicity", "statin", "zzz"]
    )
    found, missing = coverage(q, "The label warns of HEPATOTOXICITY and interacts with Statins.")
    assert found == ["Hepatotoxicity", "statin"] and missing == ["zzz"]
    assert coverage(q, None) == ([], ["Hepatotoxicity", "statin", "zzz"])


async def test_run_metrics_and_expectations(all_tools) -> None:  # type: ignore[no-untyped-def]
    result = await _result(all_tools)
    q = Question(id="q1", category="regulatory", question="?", coverage_terms=["Summary", "nope"])
    m = run_metrics(q, result)
    assert (m.status, m.proposed, m.passed) == ("approved", 3, 3)
    assert m.integrity.ok and m.coverage_found == ["Summary"] and m.coverage_missing == ["nope"]
    assert m.input_tokens == 110 and m.expectation_met is None

    off = Question(id="q2", category="safety", question="?", expect_no_claims=True)
    assert run_metrics(off, result).expectation_met is False  # verified claims on an off-topic q

    crashed = failed_run(q, "RuntimeError: boom")
    assert crashed.status == "crashed" and crashed.error == "RuntimeError: boom"
    assert crashed.coverage_missing == ["Summary", "nope"]
