"""Tool-calling seam between agents and the MCP servers."""

from typing import Any, Protocol

from mcp.server.mcpserver import MCPServer
from mcp_types import CallToolResult

from mash_agent.mcp_servers.client import ToolError


class ToolCaller(Protocol):
    async def __call__(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool and return its structured output; raise ``ToolError`` on failure."""
        ...


def in_process_caller(*servers: MCPServer) -> ToolCaller:
    """Dispatch tool calls to MCP servers living in this process (same tool contracts)."""
    by_tool: dict[str, MCPServer] = {}

    async def register() -> None:
        for server in servers:
            for tool in await server.list_tools():
                by_tool[tool.name] = server

    async def call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not by_tool:
            await register()
        if tool not in by_tool:
            raise ToolError(f"unknown tool: {tool}")
        try:
            result = await by_tool[tool].call_tool(tool, arguments)
        except Exception as exc:  # MCP surfaces tool crashes as exceptions with generic text
            raise ToolError(f"tool {tool} failed: {exc!r}") from exc
        if not isinstance(result, CallToolResult) or result.is_error:
            raise ToolError(f"tool {tool} failed: {result}")
        if result.structured_content is None:
            raise ToolError(f"tool {tool} returned no structured content")
        return dict(result.structured_content)

    return call
