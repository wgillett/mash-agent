import json
import logging
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from mash_agent import cli as cli_module
from mash_agent.agents.tools import ToolCaller
from mash_agent.cli import (
    EXIT_APPROVED,
    EXIT_NOTHING_VERIFIED,
    EXIT_REJECTED,
    cli,
    execute,
)
from mash_agent.graph.state import Decision
from tests.fakes import FunctionLLM
from tests.test_supervisor import QUESTION
from tests.test_workflow import Recorder, handler_with


@pytest.fixture
def services(all_tools: ToolCaller, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the CLI at the scripted model and recorded API responses."""
    monkeypatch.setattr(
        cli_module, "build_services", lambda: (FunctionLLM(handler_with()), all_tools)
    )


@pytest.fixture
def failing_critic(all_tools: ToolCaller, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_services",
        lambda: (FunctionLLM(handler_with(critic_rejects_all=True)), all_tools),
    )


def invoke(args: list[str], user_input: str | None = None) -> Result:
    # Rich sizes tables to the terminal; give the runner a realistic width
    return CliRunner().invoke(cli, args, input=user_input, env={"COLUMNS": "140"})


# ---- execute() -------------------------------------------------------------------------------


async def test_approved_writes_briefing_report_and_trace(
    all_tools: ToolCaller, tmp_path: Path
) -> None:
    result, code = await execute(
        QUESTION,
        FunctionLLM(handler_with()),
        all_tools,
        tmp_path,
        Recorder(Decision(approved=True)),
        trace=True,
    )
    assert code == EXIT_APPROVED
    briefing = (tmp_path / "briefing.md").read_text()
    assert briefing == result.markdown and "Not medical advice" in briefing
    report = json.loads((tmp_path / "run_report.json").read_text())
    assert report["status"] == "approved"
    assert {"plan", "outcomes", "critic", "metering", "summary", "decision"} <= report.keys()
    assert report["summary"]["input_tokens"] == 110
    assert len(report["critic"]["checked"]) == 3  # verdicts and reasons are in the trace
    spans = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert {s["name"] for s in spans} >= {"mash.run", "specialist", "llm.generate", "approval"}


async def test_trace_file_is_optional(all_tools: ToolCaller, tmp_path: Path) -> None:
    await execute(
        QUESTION,
        FunctionLLM(handler_with()),
        all_tools,
        tmp_path,
        Recorder(Decision(approved=True)),
    )
    assert not (tmp_path / "trace.jsonl").exists()


async def test_rejected_writes_report_but_no_briefing(
    all_tools: ToolCaller, tmp_path: Path
) -> None:
    _, code = await execute(
        QUESTION,
        FunctionLLM(handler_with()),
        all_tools,
        tmp_path,
        Recorder(Decision(approved=False, comment="no")),
    )
    assert code == EXIT_REJECTED
    assert not (tmp_path / "briefing.md").exists()
    assert json.loads((tmp_path / "run_report.json").read_text())["decision"]["comment"] == "no"


async def test_nothing_verified_writes_no_briefing(all_tools: ToolCaller, tmp_path: Path) -> None:
    human = Recorder(Decision(approved=True))
    _, code = await execute(
        QUESTION, FunctionLLM(handler_with(critic_rejects_all=True)), all_tools, tmp_path, human
    )
    assert code == EXIT_NOTHING_VERIFIED and human.requests == []
    assert not (tmp_path / "briefing.md").exists() and (tmp_path / "run_report.json").exists()


# ---- the Click commands ----------------------------------------------------------------------


def test_help_lists_commands_and_shorthand() -> None:
    result = invoke(["--help"])
    assert result.exit_code == 0
    for word in ("run", "report", "trace", "not a clinical tool", 'mash-agent "your question"'):
        assert word in result.output


@pytest.mark.usefixtures("services")
@pytest.mark.parametrize("prefix", [[], ["run"]])
def test_bare_question_is_shorthand_for_run(prefix: list[str], tmp_path: Path) -> None:
    result = invoke([*prefix, QUESTION, "--out-dir", str(tmp_path), "--auto-approve"])
    assert result.exit_code == EXIT_APPROVED, result.output
    for name in ("briefing.md", "run_report.json", "trace.jsonl"):
        assert (tmp_path / name).exists(), name
    # Rich summary is shown after the run
    for text in ("Agents", "Stages (LLM)", "Tools", "specialist:literature", "wrote"):
        assert text in result.output
    assert "\x1b[" not in result.output  # no ANSI escapes when not attached to a terminal


@pytest.mark.usefixtures("services")
def test_interactive_gate_shows_briefing_and_approves(tmp_path: Path) -> None:
    result = invoke([QUESTION, "--out-dir", str(tmp_path)], user_input="y\n")
    assert result.exit_code == EXIT_APPROVED, result.output
    assert "Proposed briefing" in result.output and "Summary: lit" in result.output
    assert "3 claim(s) passed the critic" in result.output
    assert (tmp_path / "briefing.md").exists()


@pytest.mark.usefixtures("services")
def test_interactive_gate_rejection_records_reason_and_writes_no_briefing(tmp_path: Path) -> None:
    result = invoke([QUESTION, "--out-dir", str(tmp_path)], user_input="n\ntoo thin\n")
    assert result.exit_code == EXIT_REJECTED, result.output
    assert not (tmp_path / "briefing.md").exists()
    assert json.loads((tmp_path / "run_report.json").read_text())["decision"] == {
        "approved": False,
        "comment": "too thin",
    }
    assert "rejected" in result.output


@pytest.mark.usefixtures("services")
def test_gate_defaults_to_reject_on_empty_answers(tmp_path: Path) -> None:
    result = invoke([QUESTION, "--out-dir", str(tmp_path)], user_input="\n\n")
    assert result.exit_code == EXIT_REJECTED and not (tmp_path / "briefing.md").exists()


@pytest.mark.usefixtures("failing_critic")
def test_nothing_verified_exits_2_without_asking(tmp_path: Path) -> None:
    result = invoke([QUESTION, "--out-dir", str(tmp_path)])  # no input: must not prompt
    assert result.exit_code == EXIT_NOTHING_VERIFIED, result.output
    assert "no claim passed verification" in result.output
    assert not (tmp_path / "briefing.md").exists()


@pytest.mark.usefixtures("services")
def test_no_trace_flag(tmp_path: Path) -> None:
    result = invoke([QUESTION, "--out-dir", str(tmp_path), "--auto-approve", "--no-trace"])
    assert result.exit_code == 0 and not (tmp_path / "trace.jsonl").exists()


@pytest.mark.usefixtures("services")
@pytest.mark.parametrize(("flags", "level"), [([], logging.WARNING), (["-v"], logging.INFO)])
def test_http_logging_is_quiet_unless_verbose(flags: list[str], level: int, tmp_path: Path) -> None:
    invoke([QUESTION, "--out-dir", str(tmp_path), "--auto-approve", *flags])
    assert logging.getLogger("httpx").level == level


def test_missing_api_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = invoke([QUESTION, "--out-dir", str(tmp_path)])
    assert result.exit_code == 1 and "ANTHROPIC_API_KEY is not set." in result.output
    assert "Traceback" not in result.output


@pytest.mark.usefixtures("services")
def test_report_and_trace_commands_render_saved_files(tmp_path: Path) -> None:
    invoke([QUESTION, "--out-dir", str(tmp_path), "--auto-approve"])
    report = invoke(["report", str(tmp_path / "run_report.json")])
    assert report.exit_code == 0 and "Stages (LLM)" in report.output and "planner" in report.output
    tree = invoke(["trace", str(tmp_path / "trace.jsonl")])
    assert tree.exit_code == 0
    for text in ("mash.run", "specialist", "tool.call", "llm.generate", "critic.source"):
        assert text in tree.output


def test_report_and_trace_fail_cleanly_on_missing_file(tmp_path: Path) -> None:
    for command in ("report", "trace"):
        result = invoke([command, str(tmp_path / "nope.json")])
        assert result.exit_code == 2 and "does not exist" in result.output
