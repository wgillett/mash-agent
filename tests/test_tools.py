from pathlib import Path

import httpx

from mash_agent.mcp_servers.client import ApiClient
from mash_agent.mcp_servers.clinicaltrials import search_trials
from mash_agent.mcp_servers.openfda import get_drug_labels
from mash_agent.mcp_servers.pubmed import parse_articles, search_pubmed
from mash_agent.ratelimit import RateLimiter

FIX = Path(__file__).parent / "fixtures"


def _api(handler: httpx.MockTransport) -> ApiClient:
    return ApiClient("https://example.test", RateLimiter(1000), transport=handler)


def test_parse_pubmed_articles() -> None:
    articles = parse_articles((FIX / "pubmed_efetch.xml").read_text())
    assert [a.pmid for a in articles] == ["38763796", "37363821"]
    assert articles[0].title == "Example resmetirom trial in MASH."
    assert articles[0].abstract == "BACKGROUND: Background text.\nRESULTS: Results text."
    assert articles[0].year == "2024"
    assert articles[1].abstract == ""


async def test_pubmed_search_two_step() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            assert request.url.params["tool"] == "mash-agent"
            return httpx.Response(200, text=(FIX / "pubmed_esearch.json").read_text())
        return httpx.Response(200, text=(FIX / "pubmed_efetch.xml").read_text())

    result = await search_pubmed(_api(httpx.MockTransport(handler)), "resmetirom")
    assert result.total_count == 2
    assert len(result.articles) == 2


async def test_pubmed_no_hits_skips_fetch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("esearch.fcgi")
        return httpx.Response(200, json={"esearchresult": {"count": "0", "idlist": []}})

    result = await search_pubmed(_api(httpx.MockTransport(handler)), "zzz")
    assert result.articles == []


async def test_trials_search_parses_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["filter.advanced"] == "AREA[Phase](PHASE2 OR PHASE3)"
        return httpx.Response(200, text=(FIX / "ctgov_studies.json").read_text())

    result = await search_trials(_api(httpx.MockTransport(handler)), "MASH")
    trial = result.trials[0]
    assert trial.nct_id == "NCT00000001"
    assert trial.sponsor == "Example Pharma"
    assert trial.phases == ["PHASE3"]
    assert trial.primary_endpoints == ["MASH resolution"]


async def test_openfda_sections_tagged_and_empty_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=(FIX / "openfda_label.json").read_text())

    result = await get_drug_labels(_api(httpx.MockTransport(handler)), "resmetirom")
    label = result.labels[0]
    assert label.label_id == "abc-123"
    names = [s.section for s in label.sections]
    assert names == ["warnings_and_cautions", "adverse_reactions"]
    assert all(s.label_id == "abc-123" for s in label.sections)


async def test_openfda_no_results_returns_empty() -> None:
    result = await get_drug_labels(
        _api(httpx.MockTransport(lambda r: httpx.Response(404, json={"error": {}}))), "nope"
    )
    assert result.labels == []
