import json
from pathlib import Path

import pytest

from mash_agent.agents.tools import ToolCaller
from mash_agent.cli import (
    EXIT_APPROVED,
    EXIT_NOTHING_VERIFIED,
    EXIT_REJECTED,
    build_parser,
    execute,
    interactive_decide,
    summary,
)
from mash_agent.graph.state import Decision
from mash_agent.graph.workflow import ApprovalRequest
from tests.fakes import FunctionLLM
from tests.test_supervisor import QUESTION
from tests.test_workflow import Recorder, handler_with


def request() -> ApprovalRequest:
    return ApprovalRequest(
        question="Q", markdown="# Briefing", verified_claims=3, excluded_claims=1, limitations=[]
    )


async def test_approved_writes_briefing_and_report(all_tools: ToolCaller, tmp_path: Path) -> None:
    result, code = await execute(
        QUESTION,
        FunctionLLM(handler_with()),
        all_tools,
        tmp_path,
        Recorder(Decision(approved=True)),
    )
    assert code == EXIT_APPROVED
    briefing = (tmp_path / "briefing.md").read_text()
    assert briefing == result.markdown and "Not medical advice" in briefing
    report = json.loads((tmp_path / "run_report.json").read_text())
    assert report["status"] == "approved"
    assert {"plan", "outcomes", "critic", "stage_usage", "decision"} <= report.keys()
    assert len(report["critic"]["checked"]) == 3  # verdicts and reasons are in the trace


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


@pytest.mark.parametrize(
    ("answers", "approved", "comment"),
    [
        (["y"], True, ""),
        (["YES "], True, ""),
        (["n", " too thin "], False, "too thin"),
        (["", ""], False, ""),
    ],
)
async def test_interactive_gate_defaults_to_reject(
    answers: list[str], approved: bool, comment: str
) -> None:
    shown: list[str] = []
    replies = iter(answers)
    decide = interactive_decide(lambda prompt: next(replies), shown.append)
    decision = await decide(request())
    assert (decision.approved, decision.comment) == (approved, comment)
    assert any("# Briefing" in s for s in shown)  # the human is shown the full briefing


def test_parser_and_summary(all_tools: ToolCaller) -> None:
    args = build_parser().parse_args(["a question", "--auto-approve", "--out-dir", "x"])
    assert (args.question, args.auto_approve, args.out_dir) == ("a question", True, Path("x"))


async def test_summary_lists_agents_critic_and_tokens(
    all_tools: ToolCaller, tmp_path: Path
) -> None:
    result, _ = await execute(
        QUESTION,
        FunctionLLM(handler_with()),
        all_tools,
        tmp_path,
        Recorder(Decision(approved=True)),
    )
    text = summary(result)
    assert "status: approved (agents: ok)" in text
    assert "critic: 3 supported, 0 unsupported, 0 unchecked" in text
    assert "tokens: 110 in / 55 out" in text
