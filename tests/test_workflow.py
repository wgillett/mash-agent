"""Full workflow: specialists -> critic -> synthesis -> human approval gate."""

import re
from datetime import date

import httpx
from pydantic import BaseModel

from mash_agent.agents.critic import ClaimVerdict, SourceVerdicts
from mash_agent.agents.models import ExtractedFindings, RawFinding
from mash_agent.agents.synthesis import BriefingDraft, BulletDraft, SectionDraft
from mash_agent.agents.tools import ToolCaller, in_process_caller
from mash_agent.graph.state import Decision
from mash_agent.graph.supervisor import SupervisorConfig
from mash_agent.graph.workflow import ApprovalRequest, Workflow
from mash_agent.mcp_servers import clinicaltrials, openfda, pubmed
from tests.conftest import api_for
from tests.fakes import FunctionLLM
from tests.test_specialists import NEJM_QUOTE
from tests.test_supervisor import QUESTION, happy, no_sleep


def handler_with(bad_lit_claim: bool = False, critic_rejects_all: bool = False):  # type: ignore[no-untyped-def]
    def handler(schema: type[BaseModel], system: str, user: str) -> BaseModel:
        if schema is ExtractedFindings and "PubMed abstracts" in system and bad_lit_claim:
            good = happy(schema, system, user)
            assert isinstance(good, ExtractedFindings)
            bad = RawFinding(
                claim="BAD: resmetirom cured all patients",
                source_id="PMID:38324483",
                evidence=NEJM_QUOTE,
            )
            return ExtractedFindings(findings=[*good.findings, bad])
        if schema is SourceVerdicts:
            claims = re.findall(r"^(\d+)\. (.*)$", user, flags=re.M)
            return SourceVerdicts(
                verdicts=[
                    ClaimVerdict(
                        claim_number=int(n),
                        verdict="unsupported" if critic_rejects_all or "BAD" in t else "supported",
                        reason="checked",
                    )
                    for n, t in claims
                ]
            )
        if schema is BriefingDraft:
            numbered = re.findall(r"^(\d+)\. \((.*?)\) (.*)$", user, flags=re.M)
            return BriefingDraft(
                sections=[
                    SectionDraft(
                        heading="Landscape",
                        bullets=[
                            BulletDraft(text=f"Summary: {t}", claim_numbers=[int(n)])
                            for n, _, t in numbered
                        ],
                    )
                ]
            )
        return happy(schema, system, user)

    return handler


def workflow(llm: FunctionLLM, tools: ToolCaller) -> Workflow:
    cfg = SupervisorConfig(backoff_s=0.0)
    return Workflow(llm, tools, cfg, sleep=no_sleep, today=lambda: date(2026, 9, 29))


class Recorder:
    """A stand-in for the human: records the request and answers with a fixed decision."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        self.requests: list[ApprovalRequest] = []

    async def __call__(self, request: ApprovalRequest) -> Decision:
        self.requests.append(request)
        return self.decision


async def test_approved_run_contains_only_critic_passed_claims(all_tools: ToolCaller) -> None:
    human = Recorder(Decision(approved=True, comment="looks fine"))
    result = await workflow(FunctionLLM(handler_with(bad_lit_claim=True)), all_tools).run(
        QUESTION, human
    )

    assert result.status == "approved" and result.decision == Decision(
        approved=True, comment="looks fine"
    )
    assert result.agents_status == "ok"
    # 4 claims proposed (lit, bad lit, trial, label); the critic excludes the bad one
    assert result.critic is not None and len(result.critic.checked) == 4
    assert result.critic.count("unsupported") == 1
    md = result.markdown
    assert md is not None
    assert "cured all patients" not in md
    for claim in ("lit", "trial", "label"):
        assert f"Summary: {claim}" in md
    assert "[PMID:38324483](https://pubmed.ncbi.nlm.nih.gov/38324483/)" in md
    assert "The critic excluded 1 of 4 claims" in md and "Generated 2026-09-29" in md
    # the gate saw exactly what would be published
    (request,) = human.requests
    assert request.markdown == md
    assert (request.verified_claims, request.excluded_claims) == (3, 1)


async def test_rejection_is_recorded_and_status_is_rejected(all_tools: ToolCaller) -> None:
    human = Recorder(Decision(approved=False, comment="too thin"))
    result = await workflow(FunctionLLM(handler_with()), all_tools).run(QUESTION, human)
    assert result.status == "rejected" and result.decision is not None
    assert result.decision.comment == "too thin"
    assert result.markdown is not None  # kept for inspection; callers must not publish it


async def test_gate_is_skipped_when_nothing_passes_the_critic(all_tools: ToolCaller) -> None:
    human = Recorder(Decision(approved=True))
    result = await workflow(FunctionLLM(handler_with(critic_rejects_all=True)), all_tools).run(
        QUESTION, human
    )
    assert result.status == "no_verified_claims"
    assert human.requests == []  # nothing to approve, so the human is not asked
    assert result.markdown is None and result.briefing is None
    assert result.critic is not None and result.critic.passed == []
    assert any("critic excluded 3 of 3" in item for item in result.limitations)


async def test_failed_specialist_shows_up_in_limitations(
    pubmed_transport: httpx.MockTransport, trials_transport: httpx.MockTransport
) -> None:
    broken = api_for(httpx.MockTransport(lambda r: httpx.Response(503)))
    broken._max_retries = 0  # noqa: SLF001
    tools = in_process_caller(
        pubmed.build_server(api_for(pubmed_transport)),
        clinicaltrials.build_server(api_for(trials_transport)),
        openfda.build_server(broken),
    )
    human = Recorder(Decision(approved=True))
    result = await workflow(FunctionLLM(handler_with()), tools).run(QUESTION, human)
    assert result.status == "approved" and result.agents_status == "partial"
    assert (
        result.markdown is not None
        and "The regulatory specialist failed after 3 attempts" in result.markdown
    )
    assert "label claim" not in result.markdown


async def test_usage_is_tracked_per_stage_including_planner(all_tools: ToolCaller) -> None:
    result = await workflow(FunctionLLM(handler_with()), all_tools).run(
        QUESTION, Recorder(Decision(approved=True))
    )
    by_stage: dict[str, int] = {}
    for s in result.stage_usage:
        by_stage[s.stage] = by_stage.get(s.stage, 0) + s.usage.input_tokens
    # the fake charges 10 input tokens per call
    assert by_stage == {
        "planner": 10,  # 1 call
        "specialist:literature": 20,  # query + extraction
        "specialist:trials": 20,
        "specialist:regulatory": 20,
        "critic": 30,  # one call per cited source
        "synthesis": 10,
    }
    assert result.total_usage.input_tokens == 110
