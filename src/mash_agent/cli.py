"""Command-line interface: a Click group with Rich terminal output.

``mash-agent "question"`` is shorthand for ``mash-agent run "question"``. Rich only formats what is
shown in the terminal; ``briefing.md``, ``run_report.json`` and ``trace.jsonl`` stay plain.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.tools import ToolCaller
from mash_agent.graph.state import Decision
from mash_agent.graph.workflow import ApprovalRequest, Decide, Workflow, WorkflowResult
from mash_agent.observability.render import (
    load_spans,
    summary_renderable,
    trace_tree,
)
from mash_agent.observability.summary import RunSummary
from mash_agent.observability.tracing import JsonlSpanExporter, configure_tracing

EXIT_APPROVED = 0
EXIT_REJECTED = 1
EXIT_NOTHING_VERIFIED = 2


class Reporter:
    """Live status line plus the messages shown at the approval gate."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._status: Status | None = None
        self._last = "starting"

    def start(self) -> None:
        self._status = self.console.status(self._last, spinner="dots")
        self._status.start()

    def progress(self, message: str) -> None:
        self._last = message
        if self._status is not None:
            self._status.update(message)

    def pause(self) -> None:
        if self._status is not None:
            self._status.stop()

    def resume(self) -> None:
        if self._status is not None:
            self._status.start()

    def stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None


def interactive_decide(
    reporter: Reporter,
    confirm: Callable[..., bool] = click.confirm,
    ask: Callable[..., str] = click.prompt,
) -> Decide:
    """The human approval gate: shows the exact briefing, defaults to reject."""

    async def decide(request: ApprovalRequest) -> Decision:
        reporter.pause()
        console = reporter.console
        console.print()
        console.print(
            Panel(
                Markdown(request.markdown),
                title="Proposed briefing (awaiting your approval)",
                border_style="cyan",
            )
        )
        console.print(
            f"[bold]{request.verified_claims}[/] claim(s) passed the critic, "
            f"[bold]{request.excluded_claims}[/] excluded."
        )
        approved = await asyncio.to_thread(confirm, "Approve and write briefing.md?", default=False)
        comment = ""
        if not approved:
            comment = await asyncio.to_thread(
                ask, "Reason for rejecting (optional)", default="", show_default=False
            )
        reporter.resume()
        return Decision(approved=bool(approved), comment=str(comment).strip())

    return decide


async def auto_approve(request: ApprovalRequest) -> Decision:
    return Decision(approved=True, comment="auto-approved (--auto-approve)")


async def execute(
    question: str,
    llm: StructuredLLM,
    tools: ToolCaller,
    out_dir: Path,
    decide: Decide,
    progress: Callable[[str], None] | None = None,
    trace: bool = False,
) -> tuple[WorkflowResult, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if trace:
        configure_tracing(JsonlSpanExporter(out_dir / "trace.jsonl"))
    try:
        result = await Workflow(llm, tools, progress=progress).run(question, decide)
    finally:
        if trace:
            configure_tracing(None)
    (out_dir / "run_report.json").write_text(result.model_dump_json(indent=2))
    if result.status == "approved" and result.markdown:
        (out_dir / "briefing.md").write_text(result.markdown)
        return result, EXIT_APPROVED
    if result.status == "rejected":
        return result, EXIT_REJECTED
    return result, EXIT_NOTHING_VERIFIED


def build_services() -> tuple[StructuredLLM, ToolCaller]:
    """The real model and MCP tool servers. Tests replace this."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise click.ClickException("ANTHROPIC_API_KEY is not set.")
    from mash_agent.agents.llm import AnthropicLLM
    from mash_agent.agents.tools import in_process_caller
    from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed

    tools = in_process_caller(
        pubmed.build_server(), clinicaltrials.build_server(), openfda.build_server()
    )
    return AnthropicLLM(), tools


def set_log_level(verbose: bool) -> None:
    """Library HTTP logging is noise unless asked for.

    Must run after ``build_services``: creating an MCP server installs a root ``RichHandler`` and
    sets the root level to INFO. The Anthropic SDK logs through ``httpx2``, not ``httpx``.
    """
    level = logging.INFO if verbose else logging.WARNING
    logging.getLogger().setLevel(level)
    for name in ("httpx", "httpx2", "httpcore", "anthropic", "mcp"):
        logging.getLogger(name).setLevel(level)


class DefaultToRun(click.Group):
    """``mash-agent "question"`` means ``mash-agent run "question"``.

    Any first argument that is not an option or a known command is treated as the question.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            args = ["run", *args]
        return super().parse_args(ctx, args)


@click.group(
    cls=DefaultToRun,
    help="Cited MASH/MASLD landscape briefing with claim-level verification. "
    "Demo only; not a clinical tool.\n\n"
    'Shorthand: mash-agent "your question" is the same as mash-agent run "your question".',
)
def cli() -> None:
    pass


@cli.command("run")
@click.argument("question")
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("."),
    show_default=True,
    help="Where to write briefing.md, run_report.json and trace.jsonl.",
)
@click.option("--auto-approve", is_flag=True, help="Skip the human gate (for evals/CI only).")
@click.option("--no-trace", is_flag=True, help="Do not write trace.jsonl.")
@click.option("-v", "--verbose", is_flag=True, help="Show library HTTP request logging.")
def run_command(
    question: str, out_dir: Path, auto_approve: bool, no_trace: bool, verbose: bool
) -> None:
    """Answer QUESTION with a verified, cited briefing and ask for approval before saving it."""
    console = Console()
    llm, tools = build_services()
    set_log_level(verbose)  # after build_services, which reconfigures logging
    reporter = Reporter(console)
    decide = auto_approve_decide() if auto_approve else interactive_decide(reporter)
    reporter.start()
    try:
        result, code = asyncio.run(
            execute(question, llm, tools, out_dir, decide, reporter.progress, trace=not no_trace)
        )
    finally:
        reporter.stop()
    if result.summary is not None:
        console.print()
        console.print(summary_renderable(result.summary))
    if code == EXIT_APPROVED:
        console.print(f"\n[green]wrote[/] {out_dir / 'briefing.md'}, {out_dir / 'run_report.json'}")
    elif result.status == "rejected":
        console.print(f"\n[yellow]rejected[/]; no briefing written. Trace data in {out_dir}")
    else:
        console.print(
            f"\n[red]no claim passed verification[/]; no briefing written. Trace data in {out_dir}"
        )
    raise SystemExit(code)


def auto_approve_decide() -> Decide:
    return auto_approve


@cli.command("report")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("run_report.json"),
)
def report_command(path: Path) -> None:
    """Show the cost/latency summary from a run_report.json."""
    data: Any = WorkflowResult.model_validate_json(path.read_text())
    if data.summary is None:
        raise click.ClickException(f"{path} has no run summary.")
    summary: RunSummary = data.summary
    Console().print(summary_renderable(summary))


@cli.command("trace")
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("trace.jsonl"),
)
def trace_command(path: Path) -> None:
    """Show the span tree (agents, LLM calls, tool calls, failures) from a trace.jsonl."""
    Console().print(trace_tree(load_spans(path)))


def main() -> None:
    cli(prog_name="mash-agent")


if __name__ == "__main__":
    main()
