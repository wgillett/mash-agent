"""Critic canary: corrupt real claims and count how many the critic still passes."""

from collections import defaultdict

from pydantic import BaseModel, Field

from mash_agent.agents.critic import Critic
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import Finding, SourceDoc, Usage
from mash_agent.evals.mutations import mutate
from mash_agent.graph.workflow import WorkflowResult


class MutationStats(BaseModel):
    n: int = 0
    caught: int = 0  # critic said unsupported
    unchecked: int = 0  # critic could not check (still excluded from the briefing)
    missed: int = 0  # critic said supported: a corrupted claim would reach the briefing


class Miss(BaseModel):
    mutation: str
    source_id: str
    original: str
    corrupted: str
    critic_reason: str


class CanaryReport(BaseModel):
    real_claims: int
    stats: dict[str, MutationStats]
    misses: list[Miss]
    usage: Usage = Field(default_factory=Usage)

    @property
    def total(self) -> int:
        return sum(s.n for s in self.stats.values())

    @property
    def missed(self) -> int:
        return sum(s.missed for s in self.stats.values())

    @property
    def miss_rate(self) -> float:
        return self.missed / self.total if self.total else 0.0


async def run_canary(
    result: WorkflowResult, llm: StructuredLLM, **critic_kwargs: object
) -> CanaryReport:
    sources: dict[str, SourceDoc] = {
        s.source_id: s for o in result.outcomes if o.result for s in o.result.sources
    }
    originals = [f for o in result.outcomes if o.result for f in o.result.findings]
    mutants: list[tuple[str, Finding, str]] = []
    for f in originals:
        for name, claim in mutate(f.claim).items():
            mutants.append((name, f.model_copy(update={"claim": claim}), f.claim))

    checked = await Critic(llm, **critic_kwargs).check([m[1] for m in mutants], sources)  # type: ignore[arg-type]
    verdict = {(c.finding.source_id, c.finding.claim): c for c in checked.checked}

    stats: dict[str, MutationStats] = defaultdict(MutationStats)
    misses: list[Miss] = []
    for name, finding, original in mutants:
        c = verdict[(finding.source_id, finding.claim)]
        s = stats[name]
        s.n += 1
        if c.outcome == "unsupported":
            s.caught += 1
        elif c.outcome == "unchecked":
            s.unchecked += 1
        else:
            s.missed += 1
            misses.append(
                Miss(
                    mutation=name,
                    source_id=finding.source_id,
                    original=original,
                    corrupted=finding.claim,
                    critic_reason=c.reason,
                )
            )
    return CanaryReport(
        real_claims=len(originals), stats=dict(stats), misses=misses, usage=checked.usage
    )
