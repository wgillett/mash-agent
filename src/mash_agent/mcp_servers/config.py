"""Environment-driven configuration shared by the MCP servers."""

import os
from pathlib import Path

from mash_agent.cache import DiskCache

TOOL_NAME = "mash-agent"


def cache_for(name: str) -> DiskCache:
    root = Path(os.environ.get("MASH_AGENT_CACHE_DIR", ".cache/mash-agent"))
    return DiskCache(root / name)


def contact_email() -> str:
    return os.environ.get("NCBI_EMAIL", "unset@example.com")
