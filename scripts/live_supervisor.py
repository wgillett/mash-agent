"""Run the full supervisor (planner + parallel specialists) with real Claude and real APIs.

Usage:
    ANTHROPIC_API_KEY=... NCBI_EMAIL=you@example.com uv run python scripts/live_supervisor.py \
        ["your question"]

Prints the plan, per-agent status/attempts/latency/errors, finding counts, and token usage.
Not part of pytest.
"""

import asyncio
import sys

from mash_agent.agents.llm import AnthropicLLM
from mash_agent.agents.tools import in_process_caller
from mash_agent.graph.supervisor import Supervisor
from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed

DEFAULT_QUESTION = (
    "Summarize the current state of late-stage MASH therapies and the safety information "
    "in resmetirom's label."
)


async def main(question: str) -> None:
    llm = AnthropicLLM()
    tools = in_process_caller(
        pubmed.build_server(), clinicaltrials.build_server(), openfda.build_server()
    )
    result = await Supervisor(llm, tools).run(question)

    print(f"\nquestion: {question}\nstatus: {result.status}   model: {llm.model}")
    for note in result.notes:
        print(f"note: {note}")
    print("\nplan:")
    for t in result.plan.tasks:
        print(f"  - {t.agent}: {t.focus}")
    print("\nagents:")
    for o in result.outcomes:
        n = len(o.result.findings) if o.result else 0
        line = (
            f"  {o.agent:<10} {o.status:<6} attempts={o.attempts} "
            f"latency={o.latency_s:5.1f}s findings={n}"
        )
        print(line + (f"  ERROR: {o.error}" if o.error else ""))
        for e in o.retry_errors:
            print(f"      retried after: {e}")
    unverified = sum(not f.evidence_verified for f in result.findings)
    u = result.report.usage
    print(f"\nfindings: {len(result.findings)} ({unverified} with unverified quotes)")
    print(f"tokens (specialists only): {u.input_tokens} in / {u.output_tokens} out")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION))
