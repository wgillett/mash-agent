"""ClinicalTrials.gov API v2 MCP server: Phase 2/3 studies tagged with NCT IDs."""

from typing import Any

from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel, Field

from mash_agent.mcp_servers.client import ApiClient
from mash_agent.mcp_servers.config import cache_for
from mash_agent.ratelimit import RateLimiter

BASE_URL = "https://clinicaltrials.gov/api/v2"


class Trial(BaseModel):
    nct_id: str
    title: str
    sponsor: str | None = None
    phases: list[str] = []
    status: str | None = None
    primary_endpoints: list[str] = []
    interventions: list[str] = []


class TrialSearchResult(BaseModel):
    condition: str
    trials: list[Trial]


def parse_study(study: dict[str, Any]) -> Trial:
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    design = proto.get("designModule", {})
    return Trial(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle", ""),
        sponsor=proto.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name"),
        phases=design.get("phases", []),
        status=proto.get("statusModule", {}).get("overallStatus"),
        primary_endpoints=[
            m.get("measure", "") for m in proto.get("outcomesModule", {}).get("primaryOutcomes", [])
        ],
        interventions=[
            i.get("name", "")
            for i in proto.get("armsInterventionsModule", {}).get("interventions", [])
        ],
    )


async def search_trials(
    client: ApiClient,
    condition: str,
    phases: tuple[str, ...] = ("PHASE2", "PHASE3"),
    max_results: int = 20,
) -> TrialSearchResult:
    phase_expr = " OR ".join(phases)
    data = await client.get_json(
        "/studies",
        {
            "query.cond": condition,
            "filter.advanced": f"AREA[Phase]({phase_expr})",
            "pageSize": max_results,
            "format": "json",
        },
    )
    trials = [parse_study(s) for s in data.get("studies", [])]
    return TrialSearchResult(condition=condition, trials=trials)


def default_client() -> ApiClient:
    return ApiClient(BASE_URL, RateLimiter(3.0), cache_for("clinicaltrials"))


def build_server(client: ApiClient | None = None) -> MCPServer:
    api = client or default_client()
    server = MCPServer("clinicaltrials")

    @server.tool()
    async def trials_search(
        condition: str = Field(description="Condition, e.g. 'MASH' or 'NASH'"),
        max_results: int = Field(default=20, ge=1, le=100),
    ) -> TrialSearchResult:
        """Search ClinicalTrials.gov for Phase 2/3 studies; results carry NCT IDs."""
        return await search_trials(api, condition, max_results=max_results)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
