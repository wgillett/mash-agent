"""Trials specialist: ClinicalTrials.gov studies -> findings tagged with NCT IDs."""

from pydantic import BaseModel, Field

from mash_agent.agents import prompts
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import SourceDoc
from mash_agent.agents.specialist import Specialist
from mash_agent.agents.tools import ToolCaller
from mash_agent.mcp_servers.clinicaltrials import Trial, TrialSearchResult

NAME = "trials"


class TrialsQuery(BaseModel):
    condition: str = Field(
        description="ClinicalTrials.gov condition query; may use OR, e.g. 'MASH OR NASH'."
    )
    max_results: int = Field(default=20, ge=1, le=50)


def render_trial(t: Trial) -> str:
    return "\n".join(
        [
            f"Title: {t.title}",
            f"Conditions: {'; '.join(t.conditions) or 'none listed'}",
            f"Sponsor: {t.sponsor or 'unknown'}",
            f"Phase: {', '.join(t.phases) or 'unknown'}",
            f"Status: {t.status or 'unknown'}",
            f"Interventions: {'; '.join(t.interventions) or 'none listed'}",
            f"Primary endpoints: {'; '.join(t.primary_endpoints) or 'none listed'}",
        ]
    )


async def fetch(tools: ToolCaller, q: TrialsQuery) -> list[SourceDoc]:
    raw = await tools("trials_search", {"condition": q.condition, "max_results": q.max_results})
    result = TrialSearchResult.model_validate(raw)
    return [
        SourceDoc(
            source_id=t.nct_id, source_type="clinicaltrials", title=t.title, text=render_trial(t)
        )
        for t in result.trials
    ]


def build(
    llm: StructuredLLM, tools: ToolCaller, extra_extract: str = ""
) -> Specialist[TrialsQuery]:
    return Specialist(
        name=NAME,
        llm=llm,
        tools=tools,
        query_schema=TrialsQuery,
        query_system=prompts.TRIALS_QUERY,
        extract_system=prompts.TRIALS_EXTRACT,
        fetch=fetch,
        extra_instructions=extra_extract,
        query_label=lambda q: q.condition,
    )
