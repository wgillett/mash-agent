"""Critic: checks every finding against the full text of its cited source; fails closed."""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, Field

from mash_agent.agents import prompts
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import Finding, SourceDoc, Usage
from mash_agent.graph.resilience import run_resilient
from mash_agent.observability.tracing import span

Verdict = Literal["supported", "unsupported"]
Outcome = Literal["supported", "unsupported", "unchecked"]


class ClaimVerdict(BaseModel):
    claim_number: int = Field(description="The claim's number from the list.")
    verdict: Verdict
    reason: str = Field(description="One sentence explaining the verdict.")


class SourceVerdicts(BaseModel):
    verdicts: list[ClaimVerdict]


class CheckedFinding(BaseModel):
    finding: Finding
    outcome: Outcome
    reason: str


class CriticReport(BaseModel):
    checked: list[CheckedFinding]
    usage: Usage = Field(default_factory=Usage)
    notes: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> list[Finding]:
        return [c.finding for c in self.checked if c.outcome == "supported"]

    def count(self, outcome: Outcome) -> int:
        return sum(1 for c in self.checked if c.outcome == outcome)


def render_check_prompt(source: SourceDoc, claims: list[Finding]) -> str:
    numbered = "\n".join(f"{i}. {f.claim}" for i, f in enumerate(claims, start=1))
    return (
        f"<source id={source.source_id!r}>\n{source.title}\n{source.text}\n</source>\n\n"
        f"Claims:\n{numbered}"
    )


class Critic:
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        timeout_s: float = 120.0,
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        max_parallel: int = 4,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._llm = llm
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._sleep_kwargs: dict[str, Callable[[float], Awaitable[None]]] = (
            {"sleep": sleep} if sleep else {}
        )

    async def check(self, findings: list[Finding], sources: dict[str, SourceDoc]) -> CriticReport:
        """One LLM call per distinct source. Anything the critic cannot check is ``unchecked``
        and is excluded from ``passed`` (fail closed)."""
        groups: dict[str, list[Finding]] = defaultdict(list)
        for f in findings:
            groups[f.source_id].append(f)

        async def guarded(
            sid: str, fs: list[Finding]
        ) -> tuple[list[CheckedFinding], Usage, str | None]:
            # Wait for a slot *before* the span/timeout start, so neither counts queueing time.
            async with self._semaphore:
                return await self._check_source(sid, fs, sources.get(sid))

        results = await asyncio.gather(*(guarded(sid, fs) for sid, fs in groups.items()))
        report = CriticReport(checked=[])
        for checked, usage, note in results:
            report.checked.extend(checked)
            report.usage += usage
            if note:
                report.notes.append(note)
        return report

    async def _check_source(
        self, source_id: str, claims: list[Finding], source: SourceDoc | None
    ) -> tuple[list[CheckedFinding], Usage, str | None]:
        with span(
            "critic.source", **{"mash.source_id": source_id, "mash.claims": len(claims)}
        ) as sp:
            checked, usage, note = await self._check_source_inner(source_id, claims, source)
            for outcome in ("supported", "unsupported", "unchecked"):
                sp.set_attribute(f"mash.{outcome}", sum(1 for c in checked if c.outcome == outcome))
            return checked, usage, note

    async def _check_source_inner(
        self, source_id: str, claims: list[Finding], source: SourceDoc | None
    ) -> tuple[list[CheckedFinding], Usage, str | None]:
        def all_as(outcome: Outcome, reason: str) -> list[CheckedFinding]:
            return [CheckedFinding(finding=f, outcome=outcome, reason=reason) for f in claims]

        if source is None:
            return all_as("unchecked", "cited source was never retrieved"), Usage(), None

        async def call() -> tuple[SourceVerdicts, Usage]:
            out = await self._llm.generate(
                SourceVerdicts, system=prompts.CRITIC, user=render_check_prompt(source, claims)
            )
            return out.value, out.usage

        attempt = await run_resilient(
            call,
            timeout_s=self._timeout_s,
            max_attempts=self._max_attempts,
            backoff_s=self._backoff_s,
            **self._sleep_kwargs,
        )
        if attempt.value is None:
            note = f"critic failed for {source_id} ({attempt.error}); its claims were excluded"
            return all_as("unchecked", f"critic failed: {attempt.error}"), Usage(), note

        verdicts, usage = attempt.value
        by_number: dict[int, ClaimVerdict] = {}
        for cv in verdicts.verdicts:
            by_number.setdefault(cv.claim_number, cv)  # first verdict wins on duplicates
        checked: list[CheckedFinding] = []
        for i, f in enumerate(claims, start=1):
            v = by_number.get(i)
            if v is None:
                checked.append(
                    CheckedFinding(finding=f, outcome="unchecked", reason="no verdict returned")
                )
            else:
                checked.append(CheckedFinding(finding=f, outcome=v.verdict, reason=v.reason))
        return checked, usage, None
