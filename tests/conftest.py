from pathlib import Path

import httpx
import pytest

from mash_agent.mcp_servers.client import ApiClient
from mash_agent.ratelimit import RateLimiter

RECORDED = Path(__file__).parent / "fixtures" / "recorded"

ESEARCH = RECORDED / "pubmed_01_entrez_eutils_esearch_fcgi_200.json"
EFETCH = RECORDED / "pubmed_02_entrez_eutils_efetch_fcgi_200.xml"
TRIALS = RECORDED / "clinicaltrials_01_api_v2_studies_200.json"
LABEL = RECORDED / "openfda_01_drug_label_json_200.json"


def api_for(handler: httpx.MockTransport) -> ApiClient:
    return ApiClient("https://example.test", RateLimiter(1000), transport=handler)


def serve(routes: dict[str, Path]) -> httpx.MockTransport:
    """Mock transport replaying recorded responses, keyed by URL path suffix."""

    def handler(request: httpx.Request) -> httpx.Response:
        for suffix, path in routes.items():
            if request.url.path.endswith(suffix):
                return httpx.Response(200, content=path.read_bytes())
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture
def pubmed_transport() -> httpx.MockTransport:
    return serve({"esearch.fcgi": ESEARCH, "efetch.fcgi": EFETCH})


@pytest.fixture
def trials_transport() -> httpx.MockTransport:
    return serve({"/studies": TRIALS})


@pytest.fixture
def label_transport() -> httpx.MockTransport:
    return serve({"/drug/label.json": LABEL})
