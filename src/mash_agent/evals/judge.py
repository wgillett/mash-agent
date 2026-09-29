"""Independent judge: grades every proposed claim against its source, before the critic."""

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel, Field

from mash_agent.agents import prompts
from mash_agent.agents.critic import render_check_prompt
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import Finding, SourceDoc
from mash_agent.graph.resilience import run_resilient

JudgeLabel = Literal["supported", "partial", "unsupported"]
Graded = Literal["supported", "partial", "unsupported", "unjudged"]


class JudgedVerdict(BaseModel):
    claim_number: int = Field(description="The claim's number from the list.")
    label: JudgeLabel
    reason: str = Field(description="One sentence explaining the grade.")


class JudgedSource(BaseModel):
    verdicts: list[JudgedVerdict]


class JudgedClaim(BaseModel):
    finding: Finding
    label: Graded
    reason: str


class Judge:
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        timeout_s: float = 180.0,
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

    async def judge(
        self, findings: list[Finding], sources: dict[str, SourceDoc]
    ) -> tuple[list[JudgedClaim], list[str]]:
        """Grade each finding; one call per source. Ungradable claims are ``unjudged`` and are
        excluded from rates rather than counted as either right or wrong."""
        groups: dict[str, list[Finding]] = defaultdict(list)
        for f in findings:
            groups[f.source_id].append(f)

        async def one(sid: str, fs: list[Finding]) -> tuple[list[JudgedClaim], str | None]:
            async with self._semaphore:
                return await self._judge_source(sid, fs, sources.get(sid))

        results = await asyncio.gather(*(one(sid, fs) for sid, fs in groups.items()))
        judged = [j for js, _ in results for j in js]
        notes = [n for _, n in results if n]
        return judged, notes

    async def _judge_source(
        self, source_id: str, claims: list[Finding], source: SourceDoc | None
    ) -> tuple[list[JudgedClaim], str | None]:
        def all_unjudged(reason: str) -> list[JudgedClaim]:
            return [JudgedClaim(finding=f, label="unjudged", reason=reason) for f in claims]

        if source is None:
            return all_unjudged("cited source was never retrieved"), None

        async def call() -> JudgedSource:
            out = await self._llm.generate(
                JudgedSource, system=prompts.JUDGE, user=render_check_prompt(source, claims)
            )
            return out.value

        attempt = await run_resilient(
            call,
            timeout_s=self._timeout_s,
            max_attempts=self._max_attempts,
            backoff_s=self._backoff_s,
            **self._sleep_kwargs,
        )
        if attempt.value is None:
            return (
                all_unjudged(f"judge failed: {attempt.error}"),
                f"judge failed for {source_id} ({attempt.error})",
            )
        by_number: dict[int, JudgedVerdict] = {}
        for jv in attempt.value.verdicts:
            by_number.setdefault(jv.claim_number, jv)
        out: list[JudgedClaim] = []
        for i, f in enumerate(claims, start=1):
            v = by_number.get(i)
            out.append(
                JudgedClaim(finding=f, label=v.label, reason=v.reason)
                if v
                else JudgedClaim(finding=f, label="unjudged", reason="no grade returned")
            )
        return out, None
