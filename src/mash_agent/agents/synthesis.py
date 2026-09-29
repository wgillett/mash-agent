"""Synthesis: organise critic-approved claims into sections; provenance is structural.

The LLM only groups and words bullets. Citations are never taken from its text: each bullet
lists claim numbers, which are resolved to verified findings in code. A bullet with no valid
claim number is dropped, and any verified claim left uncited is appended so nothing is lost.
"""

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from mash_agent.agents import prompts
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import Finding, Usage
from mash_agent.graph.resilience import run_resilient

AGENT_HEADINGS = {
    "regulatory": "FDA label information",
    "trials": "Clinical trials",
    "literature": "Published literature",
}


class BulletDraft(BaseModel):
    text: str = Field(description="One bullet, built only from the cited claims.")
    claim_numbers: list[int] = Field(description="Numbers of the claims this bullet relies on.")


class SectionDraft(BaseModel):
    heading: str
    bullets: list[BulletDraft]


class BriefingDraft(BaseModel):
    sections: list[SectionDraft]


class Bullet(BaseModel):
    text: str
    claims: list[Finding]

    @property
    def source_ids(self) -> list[str]:
        return list(dict.fromkeys(f.source_id for f in self.claims))


class Section(BaseModel):
    heading: str
    bullets: list[Bullet]


class Briefing(BaseModel):
    sections: list[Section]
    usage: Usage = Field(default_factory=Usage)
    notes: list[str] = Field(default_factory=list)

    @property
    def cited_source_ids(self) -> list[str]:
        ids = (sid for s in self.sections for b in s.bullets for sid in b.source_ids)
        return list(dict.fromkeys(ids))


def _fallback_sections(verified: list[Finding]) -> list[Section]:
    sections: list[Section] = []
    for agent, heading in AGENT_HEADINGS.items():
        claims = [f for f in verified if f.agent == agent]
        if claims:
            sections.append(
                Section(heading=heading, bullets=[Bullet(text=f.claim, claims=[f]) for f in claims])
            )
    return sections


def render_claims(verified: list[Finding]) -> str:
    return "\n".join(f"{i}. ({f.source_id}) {f.claim}" for i, f in enumerate(verified, start=1))


class Synthesizer:
    def __init__(
        self,
        llm: StructuredLLM,
        *,
        timeout_s: float = 120.0,
        max_attempts: int = 3,
        backoff_s: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._llm = llm
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        self._sleep_kwargs: dict[str, Callable[[float], Awaitable[None]]] = (
            {"sleep": sleep} if sleep else {}
        )

    async def synthesize(self, verified: list[Finding]) -> Briefing:
        async def call() -> tuple[BriefingDraft, Usage]:
            out = await self._llm.generate(
                BriefingDraft, system=prompts.SYNTHESIS, user=render_claims(verified)
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
            note = f"synthesis failed ({attempt.error}); claims listed without grouping"
            return Briefing(sections=_fallback_sections(verified), notes=[note])

        draft, usage = attempt.value
        notes: list[str] = []
        cited: set[int] = set()
        sections: list[Section] = []
        for sd in draft.sections:
            bullets: list[Bullet] = []
            for bd in sd.bullets:
                numbers = list(
                    dict.fromkeys(n for n in bd.claim_numbers if 1 <= n <= len(verified))
                )
                if not numbers or not bd.text.strip():
                    notes.append(f"dropped bullet without a valid claim citation: {bd.text[:60]!r}")
                    continue
                cited.update(numbers)
                bullets.append(
                    Bullet(text=bd.text.strip(), claims=[verified[n - 1] for n in numbers])
                )
            if bullets:
                sections.append(Section(heading=sd.heading.strip() or "Findings", bullets=bullets))
        leftover = [f for i, f in enumerate(verified, start=1) if i not in cited]
        if leftover:
            sections.append(
                Section(
                    heading="Other verified findings",
                    bullets=[Bullet(text=f.claim, claims=[f]) for f in leftover],
                )
            )
        return Briefing(sections=sections, usage=usage, notes=notes)
