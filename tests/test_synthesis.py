from datetime import date

from pydantic import BaseModel

from mash_agent.agents.models import Finding, SourceDoc
from mash_agent.agents.synthesis import (
    Briefing,
    BriefingDraft,
    BulletDraft,
    SectionDraft,
    Synthesizer,
)
from mash_agent.briefing import DISCLAIMER, citation_url, render_markdown
from tests.fakes import FunctionLLM

LABEL = "LABEL:e67ea09f-a840-439c-86c8-f98585f978b2/warnings_and_cautions"


def f(agent: str, claim: str, sid: str) -> Finding:
    return Finding(agent=agent, claim=claim, source_id=sid, evidence="q", evidence_verified=True)


VERIFIED = [
    f("literature", "lit claim", "PMID:38324483"),
    f("trials", "trial claim", "NCT07701993"),
    f("regulatory", "label claim", LABEL),
]


async def no_sleep(seconds: float) -> None:
    return None


def synth(draft: BriefingDraft | Exception) -> tuple[Synthesizer, FunctionLLM]:
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if isinstance(draft, Exception):
            raise draft
        return draft

    llm = FunctionLLM(handler)
    return Synthesizer(llm, backoff_s=0.0, sleep=no_sleep), llm


async def test_bullets_resolve_to_verified_claims_and_bad_ones_are_dropped() -> None:
    draft = BriefingDraft(
        sections=[
            SectionDraft(
                heading="Evidence",
                bullets=[
                    BulletDraft(text="Combined point.", claim_numbers=[1, 2, 1]),
                    BulletDraft(text="Cites a claim that does not exist.", claim_numbers=[7]),
                    BulletDraft(text="Cites nothing.", claim_numbers=[]),
                ],
            ),
            SectionDraft(
                heading="Empty after cleaning",
                bullets=[BulletDraft(text="No valid number.", claim_numbers=[0])],
            ),
        ]
    )
    s, llm = synth(draft)
    briefing = await s.synthesize(VERIFIED)

    assert [sec.heading for sec in briefing.sections] == ["Evidence", "Other verified findings"]
    first = briefing.sections[0].bullets[0]
    assert [c.claim for c in first.claims] == ["lit claim", "trial claim"]  # deduped, in order
    assert first.source_ids == ["PMID:38324483", "NCT07701993"]
    # claim 3 was never cited, so it is appended rather than lost
    assert [b.text for b in briefing.sections[1].bullets] == ["label claim"]
    assert len(briefing.notes) == 3 and all("dropped bullet" in n for n in briefing.notes)
    assert "1. (PMID:38324483) lit claim" in llm.users[0]  # the model saw numbered claims


async def test_synthesis_failure_falls_back_to_grouped_claims() -> None:
    s, _ = synth(RuntimeError("down"))
    briefing = await s.synthesize(VERIFIED)
    assert [sec.heading for sec in briefing.sections] == [
        "FDA label information",
        "Clinical trials",
        "Published literature",
    ]
    assert len(briefing.notes) == 1 and "synthesis failed" in briefing.notes[0]
    assert sum(len(sec.bullets) for sec in briefing.sections) == 3


def test_citation_urls() -> None:
    assert citation_url("PMID:38324483") == "https://pubmed.ncbi.nlm.nih.gov/38324483/"
    assert citation_url("NCT07701993") == "https://clinicaltrials.gov/study/NCT07701993"
    url = citation_url(LABEL)
    assert url is not None and url.startswith("https://api.fda.gov/drug/label.json?search=set_id")
    assert "e67ea09f-a840-439c-86c8-f98585f978b2" in url and "warnings_and_cautions" not in url
    assert citation_url("weird") is None


async def test_rendered_briefing_has_citations_sources_limitations_and_disclaimer() -> None:
    s, _ = synth(RuntimeError("down"))
    briefing: Briefing = await s.synthesize(VERIFIED)
    sources = {
        "PMID:38324483": SourceDoc(
            source_id="PMID:38324483", source_type="pubmed", title="NEJM trial", text="t"
        )
    }
    md = render_markdown(
        question="Q?",
        briefing=briefing,
        sources=sources,
        limitations=["The trials specialist failed."],
        generated_on=date(2026, 9, 29),
    )
    assert "**Question:** Q?" in md and "Generated 2026-09-29" in md
    assert "- lit claim [PMID:38324483](https://pubmed.ncbi.nlm.nih.gov/38324483/)" in md
    assert "- trial claim [NCT07701993](https://clinicaltrials.gov/study/NCT07701993)" in md
    assert "## Coverage and limitations\n\n- The trials specialist failed." in md
    assert "[PMID:38324483](https://pubmed.ncbi.nlm.nih.gov/38324483/) - NEJM trial" in md
    assert "(source text not available)" in md  # cited source missing from the map
    assert DISCLAIMER in md


def test_render_states_when_there_are_no_limitations() -> None:
    md = render_markdown(
        question="Q",
        briefing=Briefing(sections=[]),
        sources={},
        limitations=[],
        generated_on=date(2026, 1, 1),
    )
    assert "- No failures or dropped claims." in md
