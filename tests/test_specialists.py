"""Specialists with a scripted LLM and tool calls replayed from recorded live responses."""

import httpx
import pytest

from mash_agent.agents import literature, regulatory, trials
from mash_agent.agents.models import ExtractedFindings, RawFinding, SubTask
from mash_agent.agents.tools import in_process_caller
from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed
from mash_agent.mcp_servers.client import ToolError
from tests.conftest import api_for
from tests.fakes import ScriptedLLM

NEJM_QUOTE = (
    "Resmetirom is an oral, liver-directed, thyroid hormone receptor beta-selective agonist"
)
LABEL_QUOTE = (
    "Discontinue REZDIFFRA and continue to monitor the patient if hepatotoxicity is suspected."
)


async def test_literature_grounds_and_flags_findings(pubmed_transport: httpx.MockTransport) -> None:
    llm = ScriptedLLM(
        [
            literature.PubMedQuery(query="resmetirom AND (MASH OR NASH)", max_results=5),
            ExtractedFindings(
                findings=[
                    RawFinding(  # verbatim quote from a real retrieved abstract
                        claim="Resmetirom is an oral thyroid hormone receptor beta agonist.",
                        source_id="PMID:38324483",
                        evidence=NEJM_QUOTE,
                    ),
                    RawFinding(  # source was never retrieved -> dropped
                        claim="Resmetirom cures cirrhosis.", source_id="PMID:99999999", evidence="x"
                    ),
                    RawFinding(  # source retrieved, but the quote is not in it -> flagged
                        claim="Phase 3 enrolled 10,000 patients.",
                        source_id="PMID:38324483",
                        evidence="ten thousand patients were enrolled",
                    ),
                ]
            ),
        ]
    )
    tools = in_process_caller(pubmed.build_server(api_for(pubmed_transport)))
    result = await literature.build(llm, tools).run(SubTask(focus="resmetirom evidence"))

    assert result.queries == ["resmetirom AND (MASH OR NASH)"]
    assert {s.source_id for s in result.sources} == {
        "PMID:38851997",
        "PMID:38324483",
        "PMID:39159948",
        "PMID:38869512",
        "PMID:39038768",
    }
    assert [(f.source_id, f.evidence_verified) for f in result.findings] == [
        ("PMID:38324483", True),
        ("PMID:38324483", False),
    ]
    assert len(result.dropped) == 1 and "PMID:99999999" in result.dropped[0]
    assert (result.usage.input_tokens, result.usage.output_tokens) == (200, 40)
    # the extraction prompt carries the source text and ids
    assert "<source id='PMID:38324483'>" in llm.calls[1]["user"]
    assert NEJM_QUOTE in llm.calls[1]["user"]


async def test_trials_specialist(trials_transport: httpx.MockTransport) -> None:
    llm = ScriptedLLM(
        [
            trials.TrialsQuery(condition='MASH OR NASH OR "metabolic dysfunction-associated"'),
            ExtractedFindings(
                findings=[
                    RawFinding(
                        claim="GSK sponsors a Phase 3 efimosfermin study in MASH cirrhosis.",
                        source_id="NCT07701993",
                        evidence="Sponsor: GlaxoSmithKline",
                    )
                ]
            ),
        ]
    )
    tools = in_process_caller(clinicaltrials.build_server(api_for(trials_transport)))
    result = await trials.build(llm, tools).run(SubTask(focus="phase 3 pipeline"))

    assert len(result.sources) == 20
    by_id = {src.source_id: src for src in result.sources}
    # Titles often omit the disease, so relevance is judged from conditions: they must be shown.
    assert (
        "Conditions: Metabolic Dysfunction-associated Steatohepatitis" in by_id["NCT07701993"].text
    )
    aneurysm = by_id["NCT04876638"]  # "(MASH)" in the title but an unrelated disease
    assert "Conditions: Aneurysm, Ruptured" in aneurysm.text
    (finding,) = result.findings
    assert finding.evidence_verified and finding.agent == "trials"
    assert "MASH OR NASH" in llm.calls[0]["system"]  # prompt warns about the ambiguous acronym


async def test_regulatory_specialist_tags_label_and_section(
    label_transport: httpx.MockTransport,
) -> None:
    source_id = "LABEL:e67ea09f-a840-439c-86c8-f98585f978b2/warnings_and_cautions"
    llm = ScriptedLLM(
        [
            regulatory.LabelQuery(drugs=["Rezdiffra", "resmetirom"]),  # same label found twice
            ExtractedFindings(
                findings=[
                    RawFinding(
                        claim="Label advises discontinuing on suspected hepatotoxicity.",
                        source_id=source_id,
                        evidence=LABEL_QUOTE,
                    )
                ]
            ),
        ]
    )
    tools = in_process_caller(openfda.build_server(api_for(label_transport)))
    result = await regulatory.build(llm, tools).run(SubTask(focus="resmetirom label safety"))

    ids = [s.source_id for s in result.sources]
    assert len(ids) == len(set(ids)) == 4  # de-duplicated across the two drug names
    assert source_id in ids
    (finding,) = result.findings
    assert finding.evidence_verified


async def test_no_sources_skips_extraction() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"esearchresult": {"count": "0", "idlist": []}})

    llm = ScriptedLLM([literature.PubMedQuery(query="zzz")])
    tools = in_process_caller(pubmed.build_server(api_for(httpx.MockTransport(handler))))
    result = await literature.build(llm, tools).run(SubTask(focus="nothing"))
    assert result.findings == [] and result.sources == []
    assert len(llm.calls) == 1  # only the planning call


async def test_tool_failure_propagates_for_supervisor_to_handle() -> None:
    client = api_for(httpx.MockTransport(lambda r: httpx.Response(503)))
    client._max_retries = 0  # noqa: SLF001
    llm = ScriptedLLM([literature.PubMedQuery(query="x")])
    tools = in_process_caller(pubmed.build_server(client))
    with pytest.raises(ToolError):
        await literature.build(llm, tools).run(SubTask(focus="x"))


async def test_max_findings_caps_output_and_is_stated_in_prompt(
    trials_transport: httpx.MockTransport,
) -> None:
    def finding(nct: str) -> RawFinding:
        return RawFinding(claim=f"claim {nct}", source_id=nct, evidence="Sponsor:")

    llm = ScriptedLLM(
        [
            trials.TrialsQuery(condition="MASH OR NASH"),
            ExtractedFindings(
                findings=[finding("NCT07701993"), finding("NCT07631637"), finding("NCT06419374")]
            ),
        ]
    )
    tools = in_process_caller(clinicaltrials.build_server(api_for(trials_transport)))
    specialist = trials.build(llm, tools)
    specialist._max_findings = 2  # noqa: SLF001
    result = await specialist.run(SubTask(focus="pipeline"))
    assert [f.source_id for f in result.findings] == ["NCT07701993", "NCT07631637"]
    assert len(result.dropped) == 1 and "over max_findings=2" in result.dropped[0]
    assert "at most 12 findings" in llm.calls[1]["system"]  # default limit is what the model saw
