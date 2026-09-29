"""Tool wrappers against responses recorded from the live APIs (see scripts/live_smoke.py)."""

import re

import httpx

from mash_agent.mcp_servers.clinicaltrials import search_trials
from mash_agent.mcp_servers.openfda import get_drug_labels
from mash_agent.mcp_servers.pubmed import parse_articles, search_pubmed
from tests.conftest import EFETCH, api_for


async def test_pubmed_recorded_search(pubmed_transport: httpx.MockTransport) -> None:
    result = await search_pubmed(api_for(pubmed_transport), "resmetirom AND (MASH OR NASH)", 5)
    assert result.total_count == 384
    assert [a.pmid for a in result.articles] == [
        "38851997",
        "38324483",
        "39159948",
        "38869512",
        "39038768",
    ]
    nejm = result.articles[1]
    assert nejm.title.startswith("A Phase 3, Randomized, Controlled Trial of Resmetirom in NASH")
    assert nejm.journal == "The New England journal of medicine"
    assert nejm.year == "2024"
    assert nejm.abstract.startswith("BACKGROUND: Nonalcoholic steatohepatitis (NASH)")
    assert all(a.abstract and a.year for a in result.articles)


def test_pubmed_unlabeled_abstract_kept_as_plain_text() -> None:
    first = parse_articles(EFETCH.read_text())[0]
    assert first.pmid == "38851997"
    assert first.abstract.startswith("Metabolic dysfunction-associated steatotic liver disease")


def test_pubmed_edge_cases_no_abstract_and_medline_date() -> None:
    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article>
      <Journal><JournalIssue><PubDate><MedlineDate>2024 Jan-Feb</MedlineDate></PubDate>
      </JournalIssue></Journal><ArticleTitle>T with <i>markup</i></ArticleTitle>
      </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
    (article,) = parse_articles(xml)
    assert (article.abstract, article.year, article.title) == ("", "2024", "T with markup")


async def test_pubmed_no_hits_skips_fetch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("esearch.fcgi")
        assert request.url.params["tool"] == "mash-agent"
        return httpx.Response(200, json={"esearchresult": {"count": "0", "idlist": []}})

    result = await search_pubmed(api_for(httpx.MockTransport(handler)), "zzz")
    assert result.articles == []


async def test_trials_recorded_search(trials_transport: httpx.MockTransport) -> None:
    result = await search_trials(api_for(trials_transport), "MASH", max_results=20)
    assert len(result.trials) == 20
    assert all(re.fullmatch(r"NCT\d{8}", t.nct_id) for t in result.trials)
    assert all(set(t.phases) & {"PHASE2", "PHASE3"} for t in result.trials)
    assert all(t.conditions for t in result.trials)
    first = result.trials[0]
    assert first.nct_id == "NCT07701993"
    assert first.sponsor == "GlaxoSmithKline"
    assert first.phases == ["PHASE3"]
    assert first.conditions == ["Metabolic Dysfunction-associated Steatohepatitis"]
    assert first.status == "RECRUITING"
    assert first.interventions == ["Efimosfermin alfa", "Placebo"]
    assert first.primary_endpoints == [
        "Time from randomization to an adjudicated composite liver-related clinical outcome"
    ]


async def test_trials_sends_phase_filter() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["filter.advanced"] == "AREA[Phase](PHASE2 OR PHASE3)"
        assert request.url.params["query.cond"] == "MASH"
        return httpx.Response(200, json={"studies": []})

    result = await search_trials(api_for(httpx.MockTransport(handler)), "MASH")
    assert result.trials == []


async def test_openfda_recorded_label(label_transport: httpx.MockTransport) -> None:
    result = await get_drug_labels(api_for(label_transport), "Rezdiffra")
    (label,) = result.labels
    assert (
        label.label_id == "e67ea09f-a840-439c-86c8-f98585f978b2"
    )  # set_id, stable across versions
    assert label.brand_names == ["REZDIFFRA"]
    assert label.generic_names == ["RESMETIROM"]
    sections = {s.section: s for s in label.sections}
    # This label has no boxed warning or legacy "warnings" section, so neither is emitted.
    assert list(sections) == [
        "indications_and_usage",
        "contraindications",
        "warnings_and_cautions",
        "adverse_reactions",
    ]
    assert "Hepatotoxicity" in sections["warnings_and_cautions"].text
    assert all(s.label_id == label.label_id for s in label.sections)


async def test_openfda_no_results_returns_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})

    result = await get_drug_labels(api_for(httpx.MockTransport(handler)), "nope")
    assert result.labels == []
