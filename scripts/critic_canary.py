"""Measure the critic's miss rate by feeding it deliberately corrupted claims (real LLM).

Takes a run_report.json from `mash-agent`, corrupts each real claim in known ways
(see mash_agent.evals.mutations), and asks the production critic to check the corrupted claim
against the original source. A corrupted claim the critic calls `supported` is a miss.

Usage:
    ANTHROPIC_API_KEY=... uv run python scripts/critic_canary.py out/run_report.json \
        [--out out/critic_canary.json]

Cost: about one critic call per source in the report. Not part of pytest.
"""

import argparse
import asyncio
from pathlib import Path

from mash_agent.agents.llm import AnthropicLLM
from mash_agent.evals.canary import run_canary
from mash_agent.graph.workflow import WorkflowResult


async def main(report_path: Path, out_path: Path | None) -> None:
    result = WorkflowResult.model_validate_json(report_path.read_text())
    llm = AnthropicLLM()
    report = await run_canary(result, llm)

    print(f"model: {llm.model}   real claims: {report.real_claims}   mutants: {report.total}\n")
    print(f"{'mutation':<16}{'n':>4}{'caught':>8}{'missed':>8}{'unchecked':>11}")
    for name, s in report.stats.items():
        print(f"{name:<16}{s.n:>4}{s.caught:>8}{s.missed:>8}{s.unchecked:>11}")
    print(f"\nmiss rate: {report.missed}/{report.total} = {report.miss_rate:.0%}")
    print(f"tokens: {report.usage.input_tokens} in / {report.usage.output_tokens} out")
    for m in report.misses:
        print(
            f"\nMISSED [{m.mutation}] {m.source_id}\n  original:  {m.original}"
            f"\n  corrupted: {m.corrupted}\n  critic said: {m.critic_reason}"
        )
    if out_path:
        out_path.write_text(report.model_dump_json(indent=2))
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("report", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.report, args.out))
