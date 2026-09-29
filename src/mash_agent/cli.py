"""Command-line interface: a Click group with Rich terminal output.

``mash-agent "question"`` is shorthand for ``mash-agent run "question"``. Rich only formats what is
shown in the terminal; ``briefing.md``, ``run_report.json`` and ``trace.jsonl`` stay plain.
"""

import asyncio
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

from mash_agent.agents.llm import StructuredLLM
from mash_agent.agents.prompts import EXTRACT_VARIANTS
from mash_agent.agents.tools import ToolCaller
from mash_agent.evals.questions import DEFAULT_QUESTIONS_PATH, load_questions, select
from mash_agent.evals.report import EvalReport, compare, render_markdown
from mash_agent.evals.runner import run_eval, write_artifacts
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


# ---- evaluation ------------------------------------------------------------------------------

JUDGE_MODEL_ENV_VAR = "MASH_EVAL_JUDGE_MODEL"
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"
# Rough dollars per question from one measured run: system ~0.29, canary ~0.13, judge ~0.22.
ESTIMATED_COST_PER_QUESTION = 0.65


def build_judge_llm(model: str) -> StructuredLLM:
    """The independent judge model. Tests replace this."""
    from mash_agent.agents.llm import AnthropicLLM

    return AnthropicLLM(model=model)


@cli.group("eval")
def eval_group() -> None:
    """Score the system on a fixed question set (citation accuracy, critic, cost, latency)."""


@eval_group.command("run")
@click.option(
    "--questions",
    "questions_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_QUESTIONS_PATH,
    show_default=True,
    help="Question set (YAML).",
)
@click.option("--id", "ids", multiple=True, help="Only this question id (repeatable).")
@click.option("--limit", type=int, default=None, help="Only the first N questions.")
@click.option(
    "--variant",
    type=click.Choice(sorted(EXTRACT_VARIANTS)),
    default="baseline",
    show_default=True,
    help="Extraction-prompt variant to evaluate.",
)
@click.option(
    "--judge-model",
    default=lambda: os.environ.get(JUDGE_MODEL_ENV_VAR, DEFAULT_JUDGE_MODEL),
    show_default=f"${JUDGE_MODEL_ENV_VAR} or {DEFAULT_JUDGE_MODEL}",
    help="Model that independently grades every claim (use one different from the system's).",
)
@click.option(
    "--parallel",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Questions run at once.",
)
@click.option(
    "--no-canary", is_flag=True, help="Skip the corrupted-claim critic test (saves cost)."
)
@click.option(
    "--spotcheck",
    type=int,
    default=25,
    show_default=True,
    help="Claims to write to spotcheck.jsonl for hand labelling.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Default: eval_results/<timestamp>-<variant>.",
)
@click.option("--yes", is_flag=True, help="Do not ask before spending money.")
@click.option("-v", "--verbose", is_flag=True, help="Show library HTTP request logging.")
def eval_run(
    questions_path: Path,
    ids: tuple[str, ...],
    limit: int | None,
    variant: str,
    judge_model: str,
    parallel: int,
    no_canary: bool,
    spotcheck: int,
    out_dir: Path | None,
    yes: bool,
    verbose: bool,
) -> None:
    """Run each selected question end to end, judge all claims, write report.json and summary.md."""
    console = Console()
    try:
        questions = select(load_questions(questions_path), list(ids) or None, limit)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    llm, tools = build_services()
    set_log_level(verbose)
    judge_llm = build_judge_llm(judge_model)
    estimate = ESTIMATED_COST_PER_QUESTION * len(questions) * (0.8 if no_canary else 1.0)
    console.print(
        f"{len(questions)} question(s), variant [bold]{variant}[/], judge [bold]{judge_model}[/]. "
        f"Rough cost estimate: about ${estimate:.0f} (the report shows measured cost)."
    )
    if not yes and not click.confirm("Run the evaluation (this calls paid APIs)?", default=False):
        raise click.Abort()
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    target = out_dir or Path("eval_results") / f"{stamp}-{variant}"
    reporter = Reporter(console)
    reporter.start()
    try:
        report, artifacts = asyncio.run(
            run_eval(
                questions,
                llm=llm,
                judge_llm=judge_llm,
                tools=tools,
                variant=variant,
                canary=not no_canary,
                parallel=parallel,
                on_progress=reporter.progress,
            )
        )
    finally:
        reporter.stop()
    paths = write_artifacts(report, artifacts, target, spotcheck_n=spotcheck)
    console.print(Markdown(render_markdown(report)))
    console.print(f"\n[green]wrote[/] {paths['report']}, {paths['summary']}, {paths['spotcheck']}")


def _load_report(path: Path) -> EvalReport:
    file = path / "report.json" if path.is_dir() else path
    return EvalReport.model_validate_json(file.read_text())


@eval_group.command("compare")
@click.argument("base", type=click.Path(exists=True, path_type=Path))
@click.argument("other", type=click.Path(exists=True, path_type=Path))
def eval_compare(base: Path, other: Path) -> None:
    """Compare two eval runs (report.json files or their directories)."""
    Console().print(Markdown(compare(_load_report(base), _load_report(other))))


def main() -> None:
    cli(prog_name="mash-agent")


if __name__ == "__main__":
    main()
