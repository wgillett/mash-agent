"""Regulatory specialist: openFDA label sections -> findings tagged with label ID + section."""

from pydantic import BaseModel, Field

from mash_agent.agents import prompts
from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import SourceDoc
from mash_agent.agents.specialist import Specialist
from mash_agent.agents.tools import ToolCaller
from mash_agent.mcp_servers.openfda import LabelSearchResult

NAME = "regulatory"


class LabelQuery(BaseModel):
    drugs: list[str] = Field(
        min_length=1, max_length=3, description="Brand or generic drug names, e.g. ['Rezdiffra']."
    )


async def fetch(tools: ToolCaller, q: LabelQuery) -> list[SourceDoc]:
    docs: list[SourceDoc] = []
    seen: set[str] = set()
    for drug in q.drugs:
        raw = await tools("label_search", {"drug": drug, "limit": 1})
        for label in LabelSearchResult.model_validate(raw).labels:
            name = "/".join(label.brand_names) or "/".join(label.generic_names) or label.label_id
            for s in label.sections:
                source_id = f"LABEL:{s.label_id}/{s.section}"
                if source_id in seen:
                    continue
                seen.add(source_id)
                docs.append(
                    SourceDoc(
                        source_id=source_id,
                        source_type="openfda",
                        title=f"{name} label, {s.section}",
                        text=s.text,
                    )
                )
    return docs


def build(llm: StructuredLLM, tools: ToolCaller) -> Specialist[LabelQuery]:
    return Specialist(
        name=NAME,
        llm=llm,
        tools=tools,
        query_schema=LabelQuery,
        query_system=prompts.REGULATORY_QUERY,
        extract_system=prompts.REGULATORY_EXTRACT,
        fetch=fetch,
        query_label=lambda q: ", ".join(q.drugs),
    )
