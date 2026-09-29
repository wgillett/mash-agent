import re
from pathlib import Path

from pydantic import BaseModel

from mash_agent.agents.critic import ClaimVerdict, SourceVerdicts
from mash_agent.agents.tools import ToolCaller
from mash_agent.cli import execute
from mash_agent.evals.canary import run_canary
from mash_agent.graph.state import Decision
from mash_agent.graph.workflow import WorkflowResult
from tests.fakes import FunctionLLM
from tests.test_supervisor import QUESTION, no_sleep
from tests.test_workflow import Recorder, handler_with


async def _report(all_tools: ToolCaller, tmp_path: Path) -> WorkflowResult:
    await execute(
        QUESTION,
        FunctionLLM(handler_with()),
        all_tools,
        tmp_path,
        Recorder(Decision(approved=True)),
    )
    # the trace written to disk must load back into the model (the canary script relies on it)
    return WorkflowResult.model_validate_json((tmp_path / "run_report.json").read_text())


def critic_saying(verdict: str):  # type: ignore[no-untyped-def]
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        assert schema is SourceVerdicts
        n = len(re.findall(r"^\d+\. ", user, flags=re.M))
        return SourceVerdicts(
            verdicts=[
                ClaimVerdict(claim_number=i, verdict=verdict, reason="fake")  # type: ignore[arg-type]
                for i in range(1, n + 1)
            ]
        )

    return handler


async def test_a_critic_that_rejects_everything_has_zero_misses(
    all_tools: ToolCaller, tmp_path: Path
) -> None:
    result = await _report(all_tools, tmp_path)
    report = await run_canary(result, FunctionLLM(critic_saying("unsupported")), sleep=no_sleep)
    assert report.real_claims == 3 and report.total > 0
    assert report.missed == 0 and report.miss_rate == 0.0 and report.misses == []
    assert all(s.caught == s.n for s in report.stats.values())


async def test_a_rubber_stamp_critic_misses_everything_and_misses_are_listed(
    all_tools: ToolCaller, tmp_path: Path
) -> None:
    result = await _report(all_tools, tmp_path)
    report = await run_canary(result, FunctionLLM(critic_saying("supported")), sleep=no_sleep)
    assert report.miss_rate == 1.0 and len(report.misses) == report.total
    miss = report.misses[0]
    assert miss.original != miss.corrupted and miss.critic_reason == "fake"
    # every corrupted claim differs from a real one
    real = {f.claim for o in result.outcomes if o.result for f in o.result.findings}
    assert all(m.corrupted not in real for m in report.misses)
