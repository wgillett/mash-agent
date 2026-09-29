"""Run the three specialists with real Claude and the real APIs (network + ANTHROPIC_API_KEY).

Usage:
    ANTHROPIC_API_KEY=... NCBI_EMAIL=you@example.com uv run python scripts/live_specialists.py

Prints each specialist's queries, findings, whether each quote was verified against the source,
dropped findings, and token usage. Not part of pytest.
"""

import asyncio

from mash_agent.agents import literature, regulatory, trials
from mash_agent.agents.llm import AnthropicLLM
from mash_agent.agents.models import SpecialistResult, SubTask
from mash_agent.agents.tools import in_process_caller
from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed

TASKS = {
    "literature": SubTask(focus="Recent publications on resmetirom efficacy in MASH"),
    "trials": SubTask(focus="Phase 2/3 MASH pipeline: sponsors, interventions, primary endpoints"),
    "regulatory": SubTask(
        focus="Safety information (warnings, adverse reactions) in the resmetirom label"
    ),
}


def show(result: SpecialistResult) -> None:
    print(f"\n=== {result.agent} | queries={result.queries} | sources={len(result.sources)}")
    for f in result.findings:
        mark = "OK " if f.evidence_verified else "UNVERIFIED"
        print(f"  [{mark}] {f.source_id}: {f.claim}\n         quote: {f.evidence[:120]!r}")
    for d in result.dropped:
        print(f"  [DROPPED] {d}")
    print(f"  usage: {result.usage.input_tokens} in / {result.usage.output_tokens} out")


async def main() -> None:
    llm = AnthropicLLM()
    print(f"model: {llm.model}")
    tools = in_process_caller(
        pubmed.build_server(), clinicaltrials.build_server(), openfda.build_server()
    )
    specialists = {
        "literature": literature.build(llm, tools),
        "trials": trials.build(llm, tools),
        "regulatory": regulatory.build(llm, tools),
    }
    for name, specialist in specialists.items():
        show(await specialist.run(TASKS[name]))


if __name__ == "__main__":
    asyncio.run(main())
