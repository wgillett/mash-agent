"""Eval report: machine-readable model, short human-readable summary, and run comparison."""

import statistics
from collections import defaultdict

from pydantic import BaseModel, Field

from mash_agent.evals.canary import Miss, MutationStats
from mash_agent.evals.metrics import ClaimMetrics, ClaimRecord, Rate, RunMetrics, claim_metrics


class EvalConfig(BaseModel):
    variant: str
    system_model: str
    judge_model: str
    canary: bool
    parallel: int
    question_ids: list[str]
    started_at: str


class CanarySummary(BaseModel):
    """Critic miss rate on deliberately corrupted claims (known ground truth)."""

    stats: dict[str, MutationStats]
    miss_rate: Rate
    misses: list[Miss] = Field(default_factory=list)


class Aggregate(BaseModel):
    runs: int
    completed: Rate = Field(
        description="Runs that produced a briefing or a clean 'nothing verified'."
    )
    claims: ClaimMetrics
    by_agent_unsupported_before: dict[str, Rate]
    by_agent_exclusion: dict[str, Rate]
    integrity_ok: Rate = Field(description="Briefings whose structural guarantees all hold.")
    advice_violations: int
    coverage: Rate = Field(description="Coverage terms found in briefings.")
    expectations: Rate = Field(description="expect_no_claims questions handled as expected.")
    failed_agent_runs: int
    system_cost_usd: float | None
    mean_cost_usd: float | None
    latency_mean_s: float
    latency_p50_s: float
    latency_max_s: float
    retries: int
    failed_llm_calls: int


class EvalReport(BaseModel):
    config: EvalConfig
    aggregate: Aggregate
    runs: list[RunMetrics]
    claims: list[ClaimRecord]
    canary: CanarySummary | None = None
    judge_cost_usd: float | None = None
    judge_tokens: tuple[int, int] = (0, 0)
    notes: list[str] = Field(default_factory=list)


def aggregate(runs: list[RunMetrics], claims: list[ClaimRecord]) -> Aggregate:
    with_briefing = [r for r in runs if r.integrity.has_briefing]
    expecting = [r for r in runs if r.expectation_met is not None]
    terms_found = sum(len(r.coverage_found) for r in runs if r.integrity.has_briefing)
    terms_total = sum(
        len(r.coverage_found) + len(r.coverage_missing) for r in runs if r.integrity.has_briefing
    )
    by_agent: dict[str, list[ClaimRecord]] = defaultdict(list)
    for c in claims:
        by_agent[c.agent].append(c)
    costs = [r.cost_usd for r in runs if r.status != "crashed"]
    known = [c for c in costs if c is not None]
    latencies = [r.latency_s for r in runs if r.status != "crashed"] or [0.0]
    return Aggregate(
        runs=len(runs),
        completed=Rate(
            num=sum(r.status in ("approved", "no_verified_claims") for r in runs), den=len(runs)
        ),
        claims=claim_metrics(claims),
        by_agent_unsupported_before={
            a: claim_metrics(cs).unsupported_before for a, cs in sorted(by_agent.items())
        },
        by_agent_exclusion={
            a: claim_metrics(cs).exclusion_rate for a, cs in sorted(by_agent.items())
        },
        integrity_ok=Rate(num=sum(r.integrity.ok for r in with_briefing), den=len(with_briefing)),
        advice_violations=sum(len(r.advice_phrases) for r in runs),
        coverage=Rate(num=terms_found, den=terms_total),
        expectations=Rate(num=sum(bool(r.expectation_met) for r in expecting), den=len(expecting)),
        failed_agent_runs=sum(bool(r.failed_agents) for r in runs),
        system_cost_usd=sum(known) if known and len(known) == len(costs) else None,
        mean_cost_usd=(sum(known) / len(known)) if known and len(known) == len(costs) else None,
        latency_mean_s=statistics.fmean(latencies),
        latency_p50_s=statistics.median(latencies),
        latency_max_s=max(latencies),
        retries=sum(r.retries for r in runs),
        failed_llm_calls=sum(r.failed_llm_calls for r in runs),
    )


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.2f}"


def render_markdown(report: EvalReport) -> str:
    a, c, cfg = report.aggregate, report.aggregate.claims, report.config
    lines = [
        f"# Eval summary: variant `{cfg.variant}`",
        "",
        f"- Started {cfg.started_at}; system model `{cfg.system_model}`, "
        f"judge model `{cfg.judge_model}`; "
        f"{a.runs} question(s).",
        f"- Runs completed: {a.completed.text()}",
        "",
        "## Claims (independent judge, before vs after the critic)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Claims proposed by specialists | {c.proposed} "
        f"({c.judged} judged, {c.unjudged} unjudged) |",
        f"| Quotes verbatim in the source | {c.quote_verified.text()} |",
        f"| **Unsupported before the critic** | {c.unsupported_before.text()} |",
        f"| Not fully supported before the critic | {c.not_fully_supported_before.text()} |",
        f"| Claims that reached the briefing | {c.passed} |",
        f"| **Unsupported after the critic** | {c.unsupported_after.text()} |",
        "| **Citation accuracy, strict** (fully supported) | "
        f"{c.citation_accuracy_strict.text()} |",
        "| Citation accuracy, lenient (supported or partial) | "
        f"{c.citation_accuracy_lenient.text()} |",
        f"| Unsupported claims caught by the critic | {c.unsupported_caught} of "
        f"{c.unsupported_caught + c.unsupported_missed} |",
        f"| Critic recall | {c.critic_recall.text()} |",
        f"| Critic false rejects (good claims excluded) | {c.critic_false_reject.text()} |",
        f"| Claims excluded by the critic (any reason) | {c.exclusion_rate.text()} |",
    ]
    if report.canary:
        lines.append(
            f"| Critic miss rate on corrupted claims (canary) | {report.canary.miss_rate.text()} |"
        )
    lines += [
        "",
        "By specialist, unsupported before the critic: "
        + "; ".join(f"{k} {v.text()}" for k, v in a.by_agent_unsupported_before.items()),
        "",
    ]
    lines += [
        "## Structure, safety and coverage",
        "",
        f"- Briefings with all structural guarantees intact (cited, retrieved, critic-passed): "
        f"{a.integrity_ok.text()}",
        f"- Advice-like phrases found in briefings: {a.advice_violations}",
        f"- Topic coverage terms found: {a.coverage.text()}",
        f"- Out-of-scope questions handled as expected (nothing verified): {a.expectations.text()}",
        "",
        "## Cost and latency (system only)",
        "",
        f"- Total {_usd(a.system_cost_usd)}, mean per run {_usd(a.mean_cost_usd)}",
        f"- Latency mean {a.latency_mean_s:.1f}s, median {a.latency_p50_s:.1f}s, "
        f"max {a.latency_max_s:.1f}s",
        f"- Runs with a failed specialist: {a.failed_agent_runs}; retries {a.retries}; "
        f"failed LLM calls {a.failed_llm_calls}",
        f"- Eval overhead (judge): {_usd(report.judge_cost_usd)}",
        "",
        "## By question",
        "",
        "| Question | Status | Proposed | Passed | Cost | Latency | Missing coverage terms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report.runs:
        lines.append(
            f"| {r.question_id} | {r.status} | {r.proposed} | {r.passed} | {_usd(r.cost_usd)} | "
            f"{r.latency_s:.0f}s | {', '.join(r.coverage_missing) or '-'} |"
        )
    if report.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in report.notes]
    lines += [
        "",
        "## How to read this",
        "",
        "- The judge is an LLM (a different model and prompt from the critic), so it is an "
        "independent check, not ground truth. Sample `spotcheck.jsonl` and label a few by hand.",
        "- Rates come with counts and 95% confidence intervals. With a few hundred claims and a "
        "small error rate, differences of a few claims between runs are noise.",
        "- Cost is an estimate from list prices.",
        "",
    ]
    return "\n".join(lines)


def _rate_rows(report: EvalReport) -> list[tuple[str, Rate]]:
    a, c = report.aggregate, report.aggregate.claims
    rows = [
        ("unsupported before critic", c.unsupported_before),
        ("unsupported after critic", c.unsupported_after),
        ("citation accuracy (strict)", c.citation_accuracy_strict),
        ("citation accuracy (lenient)", c.citation_accuracy_lenient),
        ("critic recall", c.critic_recall),
        ("critic false rejects", c.critic_false_reject),
        ("claims excluded by critic", c.exclusion_rate),
        ("quotes verbatim", c.quote_verified),
        ("topic coverage", a.coverage),
        ("integrity ok", a.integrity_ok),
    ]
    if report.canary:
        rows.append(("canary miss rate", report.canary.miss_rate))
    return rows


def compare(base: EvalReport, other: EvalReport) -> str:
    """Side-by-side of two reports. A difference is 'within noise' if the 95% CIs overlap."""
    lines = [
        f"# Comparison: `{base.config.variant}` vs `{other.config.variant}`",
        "",
        "| Metric | " + base.config.variant + " | " + other.config.variant + " | Difference |",
        "|---|---|---|---|",
    ]
    for (name, r0), (_, r1) in zip(_rate_rows(base), _rate_rows(other), strict=False):
        if r0.value is None or r1.value is None:
            verdict = "n/a"
        else:
            overlap = not (
                r0.ci_high is not None and r1.ci_low is not None and r0.ci_high < r1.ci_low
            ) and not (r1.ci_high is not None and r0.ci_low is not None and r1.ci_high < r0.ci_low)
            delta = r1.value - r0.value
            verdict = f"{delta:+.1%} ({'within noise' if overlap else 'CIs do not overlap'})"
        lines.append(f"| {name} | {r0.text()} | {r1.text()} | {verdict} |")
    a0, a1 = base.aggregate, other.aggregate
    lines += [
        f"| claims proposed | {a0.claims.proposed} | {a1.claims.proposed} | "
        f"{a1.claims.proposed - a0.claims.proposed:+d} |",
        f"| claims reaching briefing | {a0.claims.passed} | {a1.claims.passed} | "
        f"{a1.claims.passed - a0.claims.passed:+d} |",
        f"| system cost | {_usd(a0.system_cost_usd)} | {_usd(a1.system_cost_usd)} | "
        + (
            f"{a1.system_cost_usd - a0.system_cost_usd:+.2f}"
            if a0.system_cost_usd is not None and a1.system_cost_usd is not None
            else "n/a"
        )
        + " |",
        f"| mean latency | {a0.latency_mean_s:.1f}s | {a1.latency_mean_s:.1f}s | "
        f"{a1.latency_mean_s - a0.latency_mean_s:+.1f}s |",
        "",
        "Only runs of the same questions are comparable. Model output varies run to run, so a "
        "single comparison can mislead; repeat runs before acting on small differences.",
        "",
    ]
    if base.config.question_ids != other.config.question_ids:
        lines.insert(2, "**Warning: the two runs used different question sets.**\n")
    return "\n".join(lines)
