"""Literature specialist: PubMed abstracts -> findings tagged with PMIDs."""

from pydantic import BaseModel, Field

from mash_agent.agents import prompts
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import SourceDoc
from mash_agent.agents.specialist import Specialist
from mash_agent.agents.tools import ToolCaller
from mash_agent.mcp_servers.pubmed import PubMedSearchResult

NAME = "literature"


class PubMedQuery(BaseModel):
    query: str = Field(description="PubMed query string, e.g. 'resmetirom AND (MASH OR NASH)'")
    max_results: int = Field(default=10, ge=1, le=20)


async def fetch(tools: ToolCaller, q: PubMedQuery) -> list[SourceDoc]:
    raw = await tools("pubmed_search", {"query": q.query, "max_results": q.max_results})
    result = PubMedSearchResult.model_validate(raw)
    return [
        SourceDoc(
            source_id=f"PMID:{a.pmid}",
            source_type="pubmed",
            title=f"{a.title} ({a.journal}, {a.year})",
            text=a.abstract,
        )
        for a in result.articles
        if a.abstract  # nothing to verify a claim against without an abstract
    ]


def build(
    llm: StructuredLLM, tools: ToolCaller, extra_extract: str = ""
) -> Specialist[PubMedQuery]:
    return Specialist(
        name=NAME,
        llm=llm,
        tools=tools,
        query_schema=PubMedQuery,
        query_system=prompts.LITERATURE_QUERY,
        extract_system=prompts.LITERATURE_EXTRACT,
        fetch=fetch,
        extra_instructions=extra_extract,
        query_label=lambda q: q.query,
    )
