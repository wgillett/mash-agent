import re

import pytest
from pydantic import BaseModel

from mash_agent.agents.critic import ClaimVerdict, Critic, SourceVerdicts
from mash_agent.agents.models import Finding, SourceDoc
from tests.fakes import FunctionLLM


def finding(claim: str, source_id: str = "PMID:1", quote: str = "SECRET-QUOTE") -> Finding:
    return Finding(
        agent="literature", claim=claim, source_id=source_id, evidence=quote, evidence_verified=True
    )


def source(source_id: str, text: str) -> SourceDoc:
    return SourceDoc(source_id=source_id, source_type="pubmed", title=f"T {source_id}", text=text)


async def no_sleep(seconds: float) -> None:
    return None


def critic(llm: FunctionLLM, **kw: object) -> Critic:
    return Critic(llm, backoff_s=0.0, sleep=no_sleep, **kw)  # type: ignore[arg-type]


def verdicts_by_marker(schema: type[BaseModel], system: str, user: str) -> BaseModel:
    """Marks a claim unsupported iff its text contains 'BAD'."""
    assert schema is SourceVerdicts
    claims = re.findall(r"^(\d+)\. (.*)$", user, flags=re.M)
    return SourceVerdicts(
        verdicts=[
            ClaimVerdict(
                claim_number=int(n),
                verdict="unsupported" if "BAD" in text else "supported",
                reason="r",
            )
            for n, text in claims
        ]
    )


async def test_one_call_per_source_and_verdicts_mapped() -> None:
    llm = FunctionLLM(verdicts_by_marker)
    findings = [finding("good a"), finding("BAD b"), finding("good c", "PMID:2")]
    sources = {
        "PMID:1": source("PMID:1", "abstract one"),
        "PMID:2": source("PMID:2", "abstract two"),
    }
    report = await critic(llm).check(findings, sources)

    assert len(llm.calls) == 2  # batched by source
    assert [(c.finding.claim, c.outcome) for c in report.checked] == [
        ("good a", "supported"),
        ("BAD b", "unsupported"),
        ("good c", "supported"),
    ]
    assert [f.claim for f in report.passed] == ["good a", "good c"]
    assert report.usage.input_tokens == 20  # 2 calls x 10 tokens from the fake


async def test_critic_sees_full_source_and_claims_but_not_the_extractors_quote() -> None:
    seen: list[str] = []

    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        seen.append(user)
        return verdicts_by_marker(schema, system, user)

    await critic(FunctionLLM(handler)).check(
        [finding("claim x")], {"PMID:1": source("PMID:1", "FULL SOURCE TEXT here")}
    )
    assert "FULL SOURCE TEXT here" in seen[0] and "1. claim x" in seen[0]
    assert "SECRET-QUOTE" not in seen[0]  # independent judgement, no anchoring on the quote


async def test_missing_verdict_fails_closed() -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        return SourceVerdicts(
            verdicts=[ClaimVerdict(claim_number=1, verdict="supported", reason="r")]
        )

    report = await critic(FunctionLLM(handler)).check(
        [finding("one"), finding("two")], {"PMID:1": source("PMID:1", "t")}
    )
    assert [c.outcome for c in report.checked] == ["supported", "unchecked"]
    assert [f.claim for f in report.passed] == ["one"]


async def test_out_of_range_and_duplicate_verdicts_are_ignored_or_first_wins() -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        return SourceVerdicts(
            verdicts=[
                ClaimVerdict(claim_number=1, verdict="unsupported", reason="first"),
                ClaimVerdict(claim_number=1, verdict="supported", reason="second"),
                ClaimVerdict(claim_number=9, verdict="supported", reason="bogus"),
            ]
        )

    report = await critic(FunctionLLM(handler)).check(
        [finding("one")], {"PMID:1": source("PMID:1", "t")}
    )
    assert [(c.outcome, c.reason) for c in report.checked] == [("unsupported", "first")]


async def test_critic_failure_isolated_to_its_source_and_excludes_claims() -> None:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if "PMID:2" in user:
            raise RuntimeError("model down")
        return verdicts_by_marker(schema, system, user)

    report = await critic(FunctionLLM(handler)).check(
        [finding("ok claim"), finding("other", "PMID:2")],
        {"PMID:1": source("PMID:1", "t"), "PMID:2": source("PMID:2", "t")},
    )
    assert [c.outcome for c in report.checked] == ["supported", "unchecked"]
    assert report.count("unchecked") == 1 and len(report.notes) == 1
    assert "PMID:2" in report.notes[0] and "excluded" in report.notes[0]


async def test_claim_citing_unretrieved_source_is_unchecked() -> None:
    llm = FunctionLLM(verdicts_by_marker)
    report = await critic(llm).check([finding("ghost", "PMID:404")], {})
    assert report.checked[0].outcome == "unchecked" and llm.calls == []


@pytest.mark.parametrize("n", [0])
async def test_no_findings_no_calls(n: int) -> None:
    llm = FunctionLLM(verdicts_by_marker)
    report = await critic(llm).check([], {})
    assert report.checked == [] and llm.calls == []
