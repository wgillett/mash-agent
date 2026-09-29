"""Eval metrics: pure functions over claim records and workflow results (no LLM, no network)."""

import math
import re
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from mash_agent.evals.judge import Graded, JudgedClaim
from mash_agent.evals.questions import Question
from mash_agent.graph.workflow import WorkflowResult

CriticOutcome = Literal["supported", "unsupported", "unchecked"]

# Phrases that would turn a factual briefing into advice. Matched case-insensitively.
ADVICE_PATTERNS = (
    r"\byou should\b",
    r"\bwe recommend\b",
    r"\bi recommend\b",
    r"\bconsult (?:your|a) (?:doctor|physician|clinician)\b",
    r"\bit is recommended that you\b",
    r"\bthe best (?:drug|treatment|option) for you\b",
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class Rate(BaseModel):
    """A proportion with its counts, so a reader can see how much data is behind it."""

    num: int
    den: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def value(self) -> float | None:
        return self.num / self.den if self.den else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ci_low(self) -> float | None:
        return wilson(self.num, self.den)[0] if self.den else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ci_high(self) -> float | None:
        return wilson(self.num, self.den)[1] if self.den else None

    def text(self) -> str:
        if not self.den:
            return "n/a (0)"
        lo, hi = wilson(self.num, self.den)
        return f"{self.num / self.den:.1%} ({self.num}/{self.den}; 95% CI {lo:.1%}-{hi:.1%})"


class ClaimRecord(BaseModel):
    """One proposed claim with the critic's and the independent judge's verdicts."""

    question_id: str
    agent: str
    source_id: str
    claim: str
    evidence_verified: bool
    critic: CriticOutcome
    critic_reason: str
    judge: Graded
    judge_reason: str
    in_briefing: bool


def build_claim_records(
    question_id: str, result: WorkflowResult, judged: list[JudgedClaim]
) -> list[ClaimRecord]:
    """Join the critic's verdicts with the judge's grades by (source, claim)."""
    if result.critic is None:
        return []
    grades = {(j.finding.source_id, j.finding.claim): j for j in judged}
    in_briefing: set[tuple[str, str]] = set()
    if result.briefing:
        for section in result.briefing.sections:
            for bullet in section.bullets:
                in_briefing.update((c.source_id, c.claim) for c in bullet.claims)
    records: list[ClaimRecord] = []
    for c in result.critic.checked:
        key = (c.finding.source_id, c.finding.claim)
        j = grades.get(key)
        records.append(
            ClaimRecord(
                question_id=question_id,
                agent=c.finding.agent,
                source_id=c.finding.source_id,
                claim=c.finding.claim,
                evidence_verified=c.finding.evidence_verified,
                critic=c.outcome,
                critic_reason=c.reason,
                judge=j.label if j else "unjudged",
                judge_reason=j.reason if j else "not judged",
                in_briefing=key in in_briefing,
            )
        )
    return records


class ClaimMetrics(BaseModel):
    proposed: int
    judged: int
    unjudged: int
    quote_verified: Rate = Field(
        description="Proposed claims whose quote is verbatim in the source."
    )
    # Before the critic: everything the specialists proposed.
    unsupported_before: Rate = Field(description="Judge: unsupported / judged, before the critic.")
    not_fully_supported_before: Rate = Field(
        description="Judge: partial or unsupported, before the critic."
    )
    # After the critic: what actually reaches the briefing.
    passed: int
    unsupported_after: Rate = Field(description="Judge: unsupported among critic-passed claims.")
    citation_accuracy_strict: Rate = Field(
        description="Passed claims the judge grades fully supported."
    )
    citation_accuracy_lenient: Rate = Field(
        description="Passed claims graded supported or partial."
    )
    # The critic against the judge's unsupported claims.
    critic_recall: Rate = Field(description="Judge-unsupported claims the critic excluded.")
    critic_false_reject: Rate = Field(description="Judge-supported claims the critic excluded.")
    exclusion_rate: Rate = Field(description="Proposed claims the critic excluded, for any reason.")
    not_fully_supported_after: Rate = Field(
        default_factory=lambda: Rate(num=0, den=0),
        description="Judge: partial or unsupported among critic-passed claims (residual error).",
    )
    critic_recall_partial: Rate = Field(
        default_factory=lambda: Rate(num=0, den=0),
        description="Judge partial-or-unsupported claims the critic excluded.",
    )
    unsupported_caught: int
    unsupported_missed: int


def claim_metrics(records: list[ClaimRecord]) -> ClaimMetrics:
    judged = [r for r in records if r.judge != "unjudged"]
    passed = [r for r in judged if r.critic == "supported"]
    unsupported = [r for r in judged if r.judge == "unsupported"]
    caught = [r for r in unsupported if r.critic != "supported"]
    supported = [r for r in judged if r.judge == "supported"]
    not_full = [r for r in judged if r.judge in ("partial", "unsupported")]
    return ClaimMetrics(
        proposed=len(records),
        judged=len(judged),
        unjudged=len(records) - len(judged),
        quote_verified=Rate(num=sum(r.evidence_verified for r in records), den=len(records)),
        unsupported_before=Rate(num=len(unsupported), den=len(judged)),
        not_fully_supported_before=Rate(
            num=sum(r.judge in ("partial", "unsupported") for r in judged), den=len(judged)
        ),
        passed=len(passed),
        unsupported_after=Rate(num=sum(r.judge == "unsupported" for r in passed), den=len(passed)),
        citation_accuracy_strict=Rate(
            num=sum(r.judge == "supported" for r in passed), den=len(passed)
        ),
        citation_accuracy_lenient=Rate(
            num=sum(r.judge in ("supported", "partial") for r in passed), den=len(passed)
        ),
        critic_recall=Rate(num=len(caught), den=len(unsupported)),
        critic_false_reject=Rate(
            num=sum(r.critic != "supported" for r in supported), den=len(supported)
        ),
        exclusion_rate=Rate(num=sum(r.critic != "supported" for r in records), den=len(records)),
        not_fully_supported_after=Rate(
            num=sum(r.judge != "supported" for r in passed), den=len(passed)
        ),
        critic_recall_partial=Rate(
            num=sum(r.critic != "supported" for r in not_full), den=len(not_full)
        ),
        unsupported_caught=len(caught),
        unsupported_missed=len(unsupported) - len(caught),
    )


# ---- deterministic checks on a run's final output ---------------------------------------------


class IntegrityChecks(BaseModel):
    """Structural guarantees of the briefing; all should hold on every run."""

    has_briefing: bool
    bullets: int = 0
    bullets_without_citation: int = 0
    cited_but_not_retrieved: int = 0
    not_critic_passed: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.bullets_without_citation == 0
            and self.cited_but_not_retrieved == 0
            and self.not_critic_passed == 0
        )


def integrity(result: WorkflowResult) -> IntegrityChecks:
    if result.briefing is None or result.critic is None:
        return IntegrityChecks(has_briefing=False)
    retrieved = {s.source_id for o in result.outcomes if o.result for s in o.result.sources}
    passed = {(f.source_id, f.claim) for f in result.critic.passed}
    checks = IntegrityChecks(has_briefing=True)
    for section in result.briefing.sections:
        for bullet in section.bullets:
            checks.bullets += 1
            if not bullet.claims:
                checks.bullets_without_citation += 1
            for c in bullet.claims:
                if c.source_id not in retrieved:
                    checks.cited_but_not_retrieved += 1
                if (c.source_id, c.claim) not in passed:
                    checks.not_critic_passed += 1
    return checks


def advice_hits(markdown: str | None) -> list[str]:
    """Advice-like phrases, ignoring the fixed disclaimer's 'not medical advice'."""
    if not markdown:
        return []
    return [p for p in ADVICE_PATTERNS if re.search(p, markdown, flags=re.IGNORECASE)]


def coverage(question: Question, markdown: str | None) -> tuple[list[str], list[str]]:
    """(terms found, terms missing) for a question's coverage terms; case-insensitive."""
    text = (markdown or "").lower()
    found = [t for t in question.coverage_terms if t.lower() in text]
    return found, [t for t in question.coverage_terms if t.lower() not in text]


class RunMetrics(BaseModel):
    question_id: str
    category: str
    status: str
    agents_status: str
    latency_s: float
    cost_usd: float | None
    input_tokens: int
    output_tokens: int
    llm_calls: int
    failed_llm_calls: int
    retries: int
    failed_agents: list[str]
    proposed: int
    passed: int
    integrity: IntegrityChecks
    advice_phrases: list[str]
    coverage_found: list[str]
    coverage_missing: list[str]
    expectation_met: bool | None = Field(
        description="For expect_no_claims questions: True if nothing was verified. None otherwise."
    )
    error: str | None = None


def run_metrics(question: Question, result: WorkflowResult) -> RunMetrics:
    s = result.summary
    assert s is not None
    found, missing = coverage(question, result.markdown)
    return RunMetrics(
        question_id=question.id,
        category=question.category,
        status=result.status,
        agents_status=result.agents_status,
        latency_s=s.wall_time_s,
        cost_usd=s.cost_usd,
        input_tokens=s.input_tokens,
        output_tokens=s.output_tokens,
        llm_calls=s.llm_calls,
        failed_llm_calls=s.failed_llm_calls,
        retries=s.retries,
        failed_agents=s.failed_agents,
        proposed=len(result.critic.checked) if result.critic else 0,
        passed=len(result.critic.passed) if result.critic else 0,
        integrity=integrity(result),
        advice_phrases=advice_hits(result.markdown),
        coverage_found=found,
        coverage_missing=missing,
        expectation_met=(result.status == "no_verified_claims")
        if question.expect_no_claims
        else None,
    )


def failed_run(question: Question, error: str) -> RunMetrics:
    """A run that crashed outside the graph's own isolation (kept in the report, never hidden)."""
    return RunMetrics(
        question_id=question.id,
        category=question.category,
        status="crashed",
        agents_status="failed",
        latency_s=0.0,
        cost_usd=None,
        input_tokens=0,
        output_tokens=0,
        llm_calls=0,
        failed_llm_calls=0,
        retries=0,
        failed_agents=[],
        proposed=0,
        passed=0,
        integrity=IntegrityChecks(has_briefing=False),
        advice_phrases=[],
        coverage_found=[],
        coverage_missing=list(question.coverage_terms),
        expectation_met=None,
        error=error,
    )
