"""PubMed E-utilities MCP server: search and fetch abstracts, each tagged with a PMID."""

import os
import xml.etree.ElementTree as ET
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from mash_agent.mcp_servers.client import ApiClient
from mash_agent.mcp_servers.config import TOOL_NAME, cache_for, contact_email
from mash_agent.ratelimit import RateLimiter

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedArticle(BaseModel):
    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    year: str | None = None


class PubMedSearchResult(BaseModel):
    query: str
    total_count: int
    articles: list[PubMedArticle]


def _text(el: ET.Element | None) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_articles(xml_text: str) -> list[PubMedArticle]:
    root = ET.fromstring(xml_text)
    articles: list[PubMedArticle] = []
    for node in root.iter("PubmedArticle"):
        pmid = _text(node.find("./MedlineCitation/PMID"))
        parts = []
        for at in node.findall(".//Abstract/AbstractText"):
            label = at.get("Label")
            body = _text(at)
            parts.append(f"{label}: {body}" if label else body)
        year = _text(node.find(".//JournalIssue/PubDate/Year")) or None
        articles.append(
            PubMedArticle(
                pmid=pmid,
                title=_text(node.find(".//ArticleTitle")),
                abstract="\n".join(p for p in parts if p),
                journal=_text(node.find(".//Journal/Title")) or None,
                year=year,
            )
        )
    return articles


async def search_pubmed(client: ApiClient, query: str, max_results: int = 10) -> PubMedSearchResult:
    common: dict[str, Any] = {"tool": TOOL_NAME, "email": contact_email()}
    if key := os.environ.get("NCBI_API_KEY"):
        common["api_key"] = key
    search = await client.get_json(
        "/esearch.fcgi",
        {
            **common,
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
        },
    )
    result = search.get("esearchresult", {})
    ids: list[str] = result.get("idlist", [])
    articles: list[PubMedArticle] = []
    if ids:
        xml_text = await client.get_text(
            "/efetch.fcgi",
            {
                **common,
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
                "rettype": "abstract",
            },
        )
        articles = parse_articles(xml_text)
    return PubMedSearchResult(
        query=query, total_count=int(result.get("count", 0)), articles=articles
    )


def default_client() -> ApiClient:
    rate = 9.0 if os.environ.get("NCBI_API_KEY") else 2.5  # stay under 10 / 3 req/s
    return ApiClient(BASE_URL, RateLimiter(rate), cache_for("pubmed"))


def build_server(client: ApiClient | None = None) -> MCPServer:
    api = client or default_client()
    server = MCPServer("pubmed")

    @server.tool()
    async def pubmed_search(
        query: str = Field(description="PubMed query, e.g. 'resmetirom AND MASH'"),
        max_results: int = Field(default=10, ge=1, le=50),
    ) -> PubMedSearchResult:
        """Search PubMed and return abstracts tagged with PMIDs."""
        return await search_pubmed(api, query, max_results)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
