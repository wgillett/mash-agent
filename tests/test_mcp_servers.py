"""Round-trip each MCP server's tool in-process, with HTTP replayed from recorded responses."""

from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult

from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed
from tests.conftest import api_for


async def _call(server: MCPServer, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await server.call_tool(tool, args)
    assert isinstance(result, CallToolResult)
    assert not result.is_error
    assert result.structured_content is not None
    return dict(result.structured_content)


async def test_pubmed_tool_round_trip(pubmed_transport: httpx.MockTransport) -> None:
    server = pubmed.build_server(api_for(pubmed_transport))
    assert [t.name for t in await server.list_tools()] == ["pubmed_search"]
    out = await _call(server, "pubmed_search", {"query": "resmetirom", "max_results": 5})
    assert out["total_count"] == 384
    assert out["articles"][0]["pmid"] == "38851997"


async def test_trials_tool_round_trip(trials_transport: httpx.MockTransport) -> None:
    server = clinicaltrials.build_server(api_for(trials_transport))
    out = await _call(server, "trials_search", {"condition": "MASH"})
    assert out["trials"][0]["nct_id"] == "NCT07701993"


async def test_openfda_tool_round_trip(label_transport: httpx.MockTransport) -> None:
    server = openfda.build_server(api_for(label_transport))
    out = await _call(server, "label_search", {"drug": "Rezdiffra"})
    sections = [s["section"] for s in out["labels"][0]["sections"]]
    assert "warnings_and_cautions" in sections
