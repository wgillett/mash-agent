"""openFDA drug label MCP server: label sections tagged with label ID and section name.

openFDA data is not validated for clinical use.
"""

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from mash_agent.mcp_servers.client import ApiClient
from mash_agent.mcp_servers.config import cache_for
from mash_agent.ratelimit import RateLimiter

BASE_URL = "https://api.fda.gov"

DEFAULT_SECTIONS = (
    "boxed_warning",
    "indications_and_usage",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "adverse_reactions",
)


class LabelSection(BaseModel):
    label_id: str
    section: str
    text: str


class DrugLabel(BaseModel):
    label_id: str
    brand_names: list[str]
    generic_names: list[str]
    effective_time: str | None = None
    sections: list[LabelSection]


class LabelSearchResult(BaseModel):
    drug: str
    labels: list[DrugLabel]


def parse_label(raw: dict[str, Any], sections: tuple[str, ...]) -> DrugLabel:
    label_id = raw.get("set_id") or raw.get("id", "")
    extracted = [
        LabelSection(label_id=label_id, section=name, text="\n".join(raw[name]))
        for name in sections
        if isinstance(raw.get(name), list) and raw[name]
    ]
    openfda = raw.get("openfda", {})
    return DrugLabel(
        label_id=label_id,
        brand_names=openfda.get("brand_name", []),
        generic_names=openfda.get("generic_name", []),
        effective_time=raw.get("effective_time"),
        sections=extracted,
    )


async def get_drug_labels(
    client: ApiClient,
    drug: str,
    sections: tuple[str, ...] = DEFAULT_SECTIONS,
    limit: int = 3,
) -> LabelSearchResult:
    quoted = f'"{drug}"'
    params: dict[str, Any] = {
        "search": f"openfda.brand_name:{quoted} OR openfda.generic_name:{quoted}",
        "limit": limit,
    }
    if key := os.environ.get("OPENFDA_API_KEY"):
        params["api_key"] = key
    data = await client.get_json("/drug/label.json", params)
    labels = [parse_label(r, sections) for r in data.get("results", [])]
    return LabelSearchResult(drug=drug, labels=labels)


def default_client() -> ApiClient:
    rate = 4.0 if os.environ.get("OPENFDA_API_KEY") else 0.5  # keyless: 240/min, be conservative
    return ApiClient(BASE_URL, RateLimiter(rate), cache_for("openfda"))


def build_server(client: ApiClient | None = None) -> MCPServer:
    api = client or default_client()
    server = MCPServer("openfda")

    @server.tool()
    async def label_search(
        drug: str = Field(description="Brand or generic name, e.g. 'Rezdiffra' or 'resmetirom'"),
        limit: int = Field(default=3, ge=1, le=10),
    ) -> LabelSearchResult:
        """Fetch FDA label sections (warnings, adverse reactions, ...) tagged by label ID."""
        return await get_drug_labels(api, drug, limit=limit)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
