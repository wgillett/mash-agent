"""`mash-agent eval ...` commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from mash_agent import cli as cli_module
from mash_agent.agents.prompts import DEFAULT_EXTRACT_VARIANT
from mash_agent.agents.tools import ToolCaller
from mash_agent.cli import cli
from mash_agent.evals.report import EvalReport
from tests.fakes import FunctionLLM
from tests.test_eval_runner import judge_handler, system_handler

QUESTIONS_YAML = """\
questions:
  - {id: one, category: regulatory, question: "First question?", coverage_terms: [Summary]}
  - {id: two, category: trials, question: "Second question?"}
"""


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    path = tmp_path / "questions.yaml"
    path.write_text(QUESTIONS_YAML)
    return path


@pytest.fixture
def judge_models(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch the services; records which judge model the CLI asked for."""
    asked: list[str] = []

    def build_judge(model: str) -> FunctionLLM:
        asked.append(model)
        judge = FunctionLLM(judge_handler)
        judge.model = model  # type: ignore[attr-defined]
        return judge

    monkeypatch.setattr(cli_module, "build_judge_llm", build_judge)
    return asked


@pytest.fixture
def eval_services(
    all_tools: ToolCaller, monkeypatch: pytest.MonkeyPatch, judge_models: list[str]
) -> None:
    def build() -> tuple[FunctionLLM, ToolCaller]:
        llm = FunctionLLM(system_handler())
        llm.model = "claude-sonnet-5-5"  # type: ignore[attr-defined]
        return llm, all_tools

    monkeypatch.setattr(cli_module, "build_services", build)


def invoke(args: list[str], user_input: str | None = None) -> Result:
    return CliRunner().invoke(cli, args, input=user_input, env={"COLUMNS": "150"})


@pytest.mark.usefixtures("eval_services")
def test_eval_run_writes_report_summary_runs_and_spotcheck(
    questions_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    result = invoke(
        ["eval", "run", "--questions", str(questions_file), "--out-dir", str(out), "--yes"]
    )
    assert result.exit_code == 0, result.output
    for name in ("report.json", "summary.md", "spotcheck.jsonl", "runs/one.json", "runs/two.json"):
        assert (out / name).exists(), name
    report = EvalReport.model_validate_json((out / "report.json").read_text())
    assert report.config.variant == DEFAULT_EXTRACT_VARIANT and report.config.canary is True
    assert report.aggregate.claims.proposed == 8 and report.canary is not None
    # the summary is also shown in the terminal
    assert "Unsupported before the critic" in result.output and "wrote" in result.output


@pytest.mark.usefixtures("eval_services")
def test_no_canary_variant_and_selection_flags(questions_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = invoke(
        [
            "eval",
            "run",
            "--questions",
            str(questions_file),
            "--out-dir",
            str(out),
            "--yes",
            "--no-canary",
            "--variant",
            "source-terms",
            "--id",
            "two",
            "--spotcheck",
            "3",
        ]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text())
    assert report["config"]["variant"] == "source-terms" and report["config"]["canary"] is False
    assert report["config"]["question_ids"] == ["two"] and report["canary"] is None
    assert len((out / "spotcheck.jsonl").read_text().splitlines()) == 3


@pytest.mark.usefixtures("eval_services")
def test_declining_the_cost_prompt_runs_nothing(questions_file: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = invoke(
        ["eval", "run", "--questions", str(questions_file), "--out-dir", str(out)], user_input="n\n"
    )
    assert result.exit_code != 0 and "Rough cost estimate" in result.output
    assert not out.exists()


@pytest.mark.usefixtures("eval_services")
def test_bad_arguments_fail_cleanly(questions_file: Path) -> None:
    unknown = invoke(["eval", "run", "--questions", str(questions_file), "--id", "nope", "--yes"])
    assert unknown.exit_code == 1 and "unknown question id" in unknown.output
    assert "Traceback" not in unknown.output
    bad_variant = invoke(["eval", "run", "--variant", "nonsense", "--yes"])
    assert bad_variant.exit_code == 2 and "Invalid value" in bad_variant.output


def test_judge_model_defaults_to_opus_and_can_come_from_the_environment(
    questions_file: Path,
    tmp_path: Path,
    eval_services: None,
    judge_models: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MASH_EVAL_JUDGE_MODEL", raising=False)
    invoke(
        [
            "eval",
            "run",
            "--questions",
            str(questions_file),
            "--out-dir",
            str(tmp_path / "a"),
            "--yes",
            "--no-canary",
            "--limit",
            "1",
        ]
    )
    monkeypatch.setenv("MASH_EVAL_JUDGE_MODEL", "some-judge")
    invoke(
        [
            "eval",
            "run",
            "--questions",
            str(questions_file),
            "--out-dir",
            str(tmp_path / "b"),
            "--yes",
            "--no-canary",
            "--limit",
            "1",
        ]
    )
    invoke(
        [
            "eval",
            "run",
            "--questions",
            str(questions_file),
            "--out-dir",
            str(tmp_path / "c"),
            "--yes",
            "--no-canary",
            "--limit",
            "1",
            "--judge-model",
            "explicit",
        ]
    )
    assert judge_models == ["claude-opus-5-5", "some-judge", "explicit"]


@pytest.mark.usefixtures("eval_services")
def test_compare_accepts_directories_and_report_files(questions_file: Path, tmp_path: Path) -> None:
    for name, variant in (("a", "legacy"), ("b", "quote-anchored")):
        result = invoke(
            [
                "eval",
                "run",
                "--questions",
                str(questions_file),
                "--yes",
                "--no-canary",
                "--out-dir",
                str(tmp_path / name),
                "--variant",
                variant,
            ]
        )
        assert result.exit_code == 0, result.output
    by_dir = invoke(["eval", "compare", str(tmp_path / "a"), str(tmp_path / "b")])
    by_file = invoke(
        [
            "eval",
            "compare",
            str(tmp_path / "a" / "report.json"),
            str(tmp_path / "b" / "report.json"),
        ]
    )
    for r in (by_dir, by_file):
        assert r.exit_code == 0, r.output
        assert "Comparison" in r.output and "legacy" in r.output and "quote-anchored" in r.output
        assert "unsupported after critic" in r.output


def test_the_real_question_file_is_the_default(eval_services: None, tmp_path: Path) -> None:
    result = invoke(
        ["eval", "run", "--yes", "--no-canary", "--limit", "1", "--out-dir", str(tmp_path / "o")]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "o" / "report.json").read_text())
    assert report["config"]["question_ids"] == ["label-safety"]


@pytest.mark.usefixtures("eval_services")
def test_reports_from_older_versions_load_and_get_the_new_metrics(
    questions_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "old"
    invoke(
        [
            "eval",
            "run",
            "--questions",
            str(questions_file),
            "--out-dir",
            str(out),
            "--yes",
            "--no-canary",
        ]
    )
    data = json.loads((out / "report.json").read_text())
    for field in ("not_fully_supported_after", "critic_recall_partial"):
        del data["aggregate"]["claims"][field]  # what an earlier version's file looks like
    (out / "report.json").write_text(json.dumps(data))
    result = invoke(["eval", "compare", str(out), str(out)])
    assert result.exit_code == 0, result.output
    assert "not fully supported after critic" in result.output


def test_eval_run_help_shows_the_default_variant_and_all_choices() -> None:
    out = invoke(["eval", "run", "--help"]).output
    assert f"[default: {DEFAULT_EXTRACT_VARIANT}]" in out
    for name in ("legacy", "source-terms", "quote-anchored", "no-commentary"):
        assert name in out
    assert "baseline" not in out


@pytest.mark.usefixtures("eval_services")
def test_eval_results_root_can_come_from_the_environment(
    questions_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MASH_AGENT_EVAL_DIR", str(tmp_path / "evals-out"))
    result = invoke(
        ["eval", "run", "--questions", str(questions_file), "--yes", "--no-canary", "--limit", "1"]
    )
    assert result.exit_code == 0, result.output
    (run_dir,) = list((tmp_path / "evals-out").iterdir())
    assert run_dir.name.endswith(f"-{DEFAULT_EXTRACT_VARIANT}")
    assert (run_dir / "report.json").exists()
