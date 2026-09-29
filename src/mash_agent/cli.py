"""Command-line entry point: ask a question, review the briefing, approve or reject it."""

import argparse
import asyncio
import os
import sys
from collections.abc import Callable
from pathlib import Path

from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.tools import ToolCaller
from mash_agent.graph.state import Decision
from mash_agent.graph.workflow import ApprovalRequest, Decide, Workflow, WorkflowResult

EXIT_APPROVED = 0
EXIT_REJECTED = 1
EXIT_NOTHING_VERIFIED = 2


def interactive_decide(
    input_fn: Callable[[str], str] = input, print_fn: Callable[[str], None] = print
) -> Decide:
    async def decide(request: ApprovalRequest) -> Decision:
        print_fn("\n" + "=" * 72 + "\nPROPOSED BRIEFING (awaiting your approval)\n" + "=" * 72)
        print_fn(request.markdown)
        print_fn(
            f"[{request.verified_claims} claim(s) passed the critic, "
            f"{request.excluded_claims} excluded]"
        )
        answer = await asyncio.to_thread(input_fn, "Approve and write briefing.md? [y/N]: ")
        if answer.strip().lower() in {"y", "yes"}:
            return Decision(approved=True)
        reason = await asyncio.to_thread(input_fn, "Reason for rejecting (optional): ")
        return Decision(approved=False, comment=reason.strip())

    return decide


async def auto_approve(request: ApprovalRequest) -> Decision:
    return Decision(approved=True, comment="auto-approved (--auto-approve)")


def summary(result: WorkflowResult) -> str:
    lines = [f"status: {result.status} (agents: {result.agents_status})"]
    for o in result.outcomes:
        n = len(o.result.findings) if o.result else 0
        lines.append(
            f"  {o.agent:<10} {o.status:<6} attempts={o.attempts} "
            f"latency={o.latency_s:.1f}s findings={n}"
        )
    if result.critic:
        c = result.critic
        lines.append(
            f"  critic: {len(c.passed)} supported, {c.count('unsupported')} unsupported, "
            f"{c.count('unchecked')} unchecked"
        )
    u = result.total_usage
    lines.append(f"  tokens: {u.input_tokens} in / {u.output_tokens} out")
    return "\n".join(lines)


async def execute(
    question: str,
    llm: StructuredLLM,
    tools: ToolCaller,
    out_dir: Path,
    decide: Decide,
) -> tuple[WorkflowResult, int]:
    result = await Workflow(llm, tools).run(question, decide)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_report.json").write_text(result.model_dump_json(indent=2))
    if result.status == "approved" and result.markdown:
        (out_dir / "briefing.md").write_text(result.markdown)
        return result, EXIT_APPROVED
    if result.status == "rejected":
        return result, EXIT_REJECTED
    return result, EXIT_NOTHING_VERIFIED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mash-agent",
        description="Cited MASH/MASLD landscape briefing with claim-level verification. "
        "Demo only; not a clinical tool.",
    )
    parser.add_argument("question", help="natural-language question about the MASH/MASLD landscape")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="where to write outputs")
    parser.add_argument(
        "--auto-approve", action="store_true", help="skip the human gate (for evals/CI only)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")
    # Imported here so `--help` and tests do not need the network-facing servers.
    from mash_agent.agents.llm import AnthropicLLM
    from mash_agent.agents.tools import in_process_caller
    from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed

    tools = in_process_caller(
        pubmed.build_server(), clinicaltrials.build_server(), openfda.build_server()
    )
    decide = auto_approve if args.auto_approve else interactive_decide()
    result, code = asyncio.run(execute(args.question, AnthropicLLM(), tools, args.out_dir, decide))
    print("\n" + summary(result))
    if code == EXIT_APPROVED:
        print(f"wrote {args.out_dir / 'briefing.md'} and {args.out_dir / 'run_report.json'}")
    else:
        print(f"no briefing written; trace in {args.out_dir / 'run_report.json'}")
    sys.exit(code)


if __name__ == "__main__":
    main()
