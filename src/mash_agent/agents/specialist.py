"""Generic specialist: plan a query -> fetch sources via tools -> extract grounded findings."""

import re
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.models import (
    ExtractedFindings,
    Finding,
    SourceDoc,
    SpecialistResult,
    SubTask,
    Usage,
)
from mash_agent.agents.tools import ToolCaller

MAX_PROMPT_CHARS_PER_SOURCE = 8000

Fetch = Callable[[ToolCaller, BaseModel], Awaitable[list[SourceDoc]]]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def render_sources(sources: list[SourceDoc]) -> str:
    blocks = []
    for s in sources:
        body = s.text[:MAX_PROMPT_CHARS_PER_SOURCE]
        blocks.append(f"<source id={s.source_id!r}>\n{s.title}\n{body}\n</source>")
    return "\n\n".join(blocks)


class Specialist[Q: BaseModel]:
    """Failures (tool errors, LLM errors) propagate; the supervisor owns retry and isolation."""

    def __init__(
        self,
        *,
        name: str,
        llm: StructuredLLM,
        tools: ToolCaller,
        query_schema: type[Q],
        query_system: str,
        extract_system: str,
        fetch: Callable[[ToolCaller, Q], Awaitable[list[SourceDoc]]],
        query_label: Callable[[Q], str],
    ) -> None:
        self.name = name
        self._llm = llm
        self._tools = tools
        self._query_schema = query_schema
        self._query_system = query_system
        self._extract_system = extract_system
        self._fetch = fetch
        self._query_label = query_label

    async def run(self, task: SubTask) -> SpecialistResult:
        usage = Usage()
        planned = await self._llm.generate(
            self._query_schema, system=self._query_system, user=task.focus
        )
        usage += planned.usage
        query = planned.value
        sources = await self._fetch(self._tools, query)
        result = SpecialistResult(
            agent=self.name,
            subtask=task,
            queries=[self._query_label(query)],
            sources=sources,
            findings=[],
            usage=usage,
        )
        if not sources:
            return result

        user = f"Task: {task.focus}\n\nSources:\n\n{render_sources(sources)}"
        extracted = await self._llm.generate(
            ExtractedFindings, system=self._extract_system, user=user
        )
        result.usage += extracted.usage
        by_id = {s.source_id: s for s in sources}
        for raw in extracted.value.findings:
            source = by_id.get(raw.source_id)
            if source is None:
                result.dropped.append(f"unknown source_id {raw.source_id!r}: {raw.claim[:80]}")
                continue
            result.findings.append(
                Finding(
                    agent=self.name,
                    claim=raw.claim,
                    source_id=raw.source_id,
                    evidence=raw.evidence,
                    evidence_verified=bool(raw.evidence.strip())
                    and _norm(raw.evidence) in _norm(source.text),
                )
            )
        return result
