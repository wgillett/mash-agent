# AGENTS.md

## Project Overview

**MASH Landscape Briefing Agent**: a multi-agent system that takes a natural-language question about the MASH/MASLD (metabolic dysfunction-associated steatohepatitis / steatotic liver disease) landscape and returns a cited briefing in which every factual claim has been checked against a retrieved source, together with a trace of how the agents produced it.

This is a demo and learning exercise. Its purpose is to exercise production-grade patterns for multi-agent systems in one small, polished repo: orchestration, tool contracts, verification, human-in-the-loop approval, observability, cost tracking, and evaluation. The MASH topic is a vehicle; the system is not a novel product and is **not a clinical tool**.

**Example input:** "Summarize the current state of late-stage MASH therapies and the safety information in resmetirom's label."

## Goals

- Demonstrate a supervisor/specialist multi-agent architecture with clear responsibilities and failure isolation.
- Demonstrate claim-level verification: no claim reaches the final briefing without support from a retrieved source.
- Provide observability (tracing, per-run token/cost/latency) and a small evaluation harness.
- Keep scope small and polished. A clean, well-documented repo is preferred over a sprawling one.

## Non-Goals

- Clinical decision support or any medical advice.
- Full-text literature retrieval beyond what public APIs provide (abstracts; open-access full text only via PubMed Central if added later).
- Supporting data sources beyond the three public APIs below (initially).

## Architecture

```
User question
     |
     v
Planner / Supervisor
     |  (decomposes question, assigns sub-tasks, runs specialists in parallel)
     +--> Literature agent   --> PubMed E-utilities
     +--> Trials agent       --> ClinicalTrials.gov API v2
     +--> Regulatory agent   --> openFDA drug label API
     |
     v
Critic / Verifier agent  (checks each claim against source text)
     |
     v
Synthesis  -->  Human approval gate  -->  briefing.md + trace + run report
```

### Components

1. **Planner/Supervisor**: breaks the question into sub-tasks (e.g., approved drugs and label safety, Phase 3 pipeline, recent publications) and delegates to specialists. Handles retries, timeouts, and partial failure so one failing specialist does not sink the run.
2. **Literature agent**: searches PubMed, fetches abstracts, returns structured findings each tagged with a PMID.
3. **Trials agent**: queries ClinicalTrials.gov for Phase 2/3 MASH studies; returns sponsor, phase, status, endpoints, and NCT ID.
4. **Regulatory agent**: pulls drug labels from openFDA (e.g., resmetirom/Rezdiffra) and extracts sections such as warnings and adverse reactions, tagged with label ID and section.
5. **Critic/Verifier agent**: for each drafted claim, checks it against the underlying source text. Unsupported claims are dropped or flagged.
6. **Synthesis step**: assembles verified claims into a markdown briefing with an inline citation per claim.
7. **Human approval gate**: pauses for a person to approve or reject before the briefing is finalized.
8. **Run report**: per-agent token usage, cost, latency, retries, and failures.

## Data Sources (all free, public)

| API | Registration | Notes |
|---|---|---|
| PubMed E-utilities (NCBI) | Not required; free API key optional | ~3 req/s without key, ~10 req/s with key. Include `tool` and `email` parameters per NCBI guidelines. |
| ClinicalTrials.gov API v2 (`/api/v2/studies`) | None | Use v2 only; the classic API is retired. Keep usage reasonable. |
| openFDA (`/drug/label.json`) | Not required; free key optional | Lower limits without a key; much higher with one. Data is not validated for clinical use. |

Rate limits above are approximate and may have changed; verify against current provider docs before building:

- https://www.ncbi.nlm.nih.gov/books/NBK25497/
- https://clinicaltrials.gov/data-api/api
- https://open.fda.gov/apis/

## Technical Direction

- **Language:** Python 3.13 (recent, and well supported by the AI/LLM library ecosystem).
- **LLM provider:** Anthropic (Claude), via `langchain-anthropic` inside LangGraph. API key in `ANTHROPIC_API_KEY`; model IDs are configurable via environment variables, not hardcoded.
- **Orchestration:** LangGraph.
- **Tools:** expose each data source as a thin, self-written **MCP server** with an explicit tool contract (typed inputs/outputs), rather than depending on community servers.
- **Tracing:** Langfuse or OpenTelemetry, with per-run token and cost tracking.
- **CLI:** [Click](https://click.palletsprojects.com/) for commands (a command group; more commands are expected, for example evals and report viewing). Do not add further `argparse`-based commands.
- **Terminal output:** [Rich](https://rich.readthedocs.io/) for formatted output (rendered briefing at the approval gate, status tables, progress display). Rich affects terminal output only: `briefing.md`, `run_report.json` and other files stay plain and machine-readable, and output must degrade cleanly when not attached to a terminal.
- **Packaging:** Dockerized; runnable with a single documented command.
- **Optional extension:** reimplement the same workflow in CrewAI or Pydantic AI and write a one-page comparison of trade-offs (state management, delegation model, observability, ergonomics).

## Tooling

- **Project manager:** [uv](https://docs.astral.sh/uv/). Dependencies live in `pyproject.toml`, `uv.lock` is committed, and the Python version is pinned in `.python-version` (`3.13`). Do not use pip, Poetry, or requirements files directly.
- **Build backend:** hatchling, with a `src/` layout (`src/mash_agent/`).
- **Dev dependencies** go in `[dependency-groups] dev`: `ruff`, `mypy`, `pytest`, `pytest-asyncio`, `pytest-cov`, `pre-commit`.
- **Linting/formatting:** ruff (line length 100; rules `E`, `F`, `I`, `UP`, `B`, `SIM`).
- **Type checking:** mypy in strict mode over `src` and `tests`.
- **Tests:** pytest with `testpaths = ["tests"]`. Tests must not hit the network; use recorded fixtures for API responses.
- **Pre-commit:** hooks for ruff (lint + format) and mypy.
- **Docker:** base the image on the official uv image and install with `uv sync --frozen --no-dev`.

Starting point for `pyproject.toml`:

```toml
[project]
name = "mash-agent"
version = "0.1.0"
description = "Multi-agent MASH/MASLD landscape briefing demo with claim-level verification"
readme = "README.md"
requires-python = ">=3.13"
dependencies = []  # add runtime deps (langgraph, langchain-anthropic, mcp, httpx, pydantic, ...) via `uv add`

[project.scripts]
mash-agent = "mash_agent.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mash_agent"]

[dependency-groups]
dev = ["ruff", "mypy", "pytest", "pytest-asyncio", "pytest-cov", "pre-commit"]

[tool.ruff]
line-length = 100
target-version = "py313"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.13"
strict = true
files = ["src", "tests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

### Common commands

```bash
uv sync                          # create .venv and install all deps (incl. dev)
uv add <pkg>                     # add a runtime dependency
uv add --dev <pkg>               # add a dev dependency
uv run pytest                    # run tests
uv run ruff check . && uv run ruff format .   # lint and format
uv run mypy                      # type check
uv run pre-commit install        # enable git hooks
```

Before committing, `ruff check`, `ruff format --check`, `mypy`, and `pytest` must all pass.

## Production Requirements

- **Caching:** simple on-disk cache for API responses.
- **Rate limiting:** per-tool limiters; exponential backoff on HTTP 429. Parallel specialists must not exceed PubMed limits.
- **Resilience:** timeouts and retries per tool call; failure isolation between agents; the run degrades gracefully with a report of what failed.
- **Verification:** every claim in the final briefing must map to a source ID (PMID, NCT ID, or label ID + section) and pass the critic.
- **Human-in-the-loop:** explicit approve/reject step before finalizing output.
- **Observability:** full agent graph, tool calls, latencies, token usage, cost, and failures visible in traces.
- **Security/data:** public data only; no secrets in the repo; API keys via environment variables.

## Outputs

- `briefing.md`: the cited briefing.
- Trace (Langfuse/OpenTelemetry) showing agent graph, tool calls, and failures.
- Eval report over a fixed question set (~10-20 questions): share of claims correctly cited, and number of unsupported claims caught by the critic.

## Evaluation

- Maintain a fixed question set in the repo (e.g., `evals/questions.yaml`).
- Metrics: citation accuracy (claim supported by the cited source), hallucination/unsupported-claim rate before and after the critic, cost and latency per run.
- Evals must be runnable from a single command and produce a machine-readable report plus a short human-readable summary.

## Suggested Repository Layout

```
.
├── AGENTS.md
├── README.md              # overview, architecture diagram, design decisions, disclaimers
├── pyproject.toml
├── uv.lock
├── .python-version
├── .pre-commit-config.yaml
├── Dockerfile
├── src/
│   └── mash_agent/
│       ├── cli.py             # entry point (Click command group; Rich output)
│       ├── graph/             # LangGraph definition: supervisor, specialists, critic, synthesis
│       ├── agents/            # agent prompts and logic
│       ├── mcp_servers/       # pubmed, clinicaltrials, openfda MCP servers
│       ├── cache/             # on-disk cache
│       ├── ratelimit/         # per-tool limiters and backoff
│       └── observability/     # tracing and cost tracking
├── evals/
│   ├── questions.yaml
│   └── run_evals.py
├── tests/
└── docs/
    └── design-decisions.md
```

## Development Guidelines

- Keep scope small; finish and polish before adding features.
- Type-annotate all code; validate tool inputs/outputs with Pydantic models.
- Write tests for tool wrappers (with recorded fixtures), rate limiting, and the critic's claim-checking logic.
- Document non-obvious design choices in `docs/design-decisions.md` (why supervisor/specialist, why a separate critic, how failures are isolated, how cost is tracked).
- Never present output as medical advice.

## README Requirements

- What the system does, with the example input and sample output.
- Architecture diagram.
- Design-decisions section.
- Setup and run instructions (including Docker).
- Eval instructions and latest results.
- Disclaimers: demo and learning exercise; not a clinical tool; openFDA data is not validated for clinical use; all data from public sources.

## Milestones

1. **Tools:** project skeleton per **Tooling** (uv, ruff, mypy, pytest, pre-commit), then three MCP servers with typed contracts, caching, rate limiting, and tests.
2. **Specialists:** literature, trials, and regulatory agents returning source-tagged structured findings.
3. **Orchestration:** supervisor with parallel delegation, retries, timeouts, and failure isolation.
4. **Verification and synthesis:** critic agent, cited briefing, human approval gate.
5. **Observability and CLI:** tracing plus per-run token/cost/latency report. Also move the CLI from `argparse` to a Click command group (keep `mash-agent "question"` working) and add Rich terminal output: formatted briefing review at the approval gate, per-agent status table, and a progress display during runs. Tests for CLI behaviour use Click's `CliRunner`.
6. **Evals:** fixed question set, metrics, and report.
7. **Polish:** Dockerization, README, architecture diagram, design-decisions doc.
8. **Optional:** CrewAI or Pydantic AI reimplementation and trade-off comparison. **Skipped (decision, 2026-09-29):** the project is complete without it.

## Follow-ups (optional, not started)

Ideas noted during Milestones 1 to 7. None is required; keep scope small and finish the item before starting another.

- **Trials relevance:** the trials specialist keeps some low-relevance results (for example old academic NAFLD trials) that take slots from the current MASH pipeline. Prefer active or recently completed industry-sponsored MASH trials, and measure relevance in the evals (today they measure accuracy and coarse topic coverage only).
- **Calibrate the eval judge:** hand-label 10 to 25 rows of `spotcheck.jsonl` and compare with the judge's grades. Optionally run a judge from a different vendor on a subset (for example the spot-check sample or the claims where the critic and judge disagree) to reduce correlated blind spots; this would need a provider decision, since Anthropic is the specified provider.
- **Stronger eval evidence:** replicate the extraction-prompt variants (`quote-anchored` and `no-commentary` have one run each), add the harder canary mutations (wrong arm or population, primary versus secondary endpoint, dropped qualifier), and extend the advice-phrase check beyond its narrow phrase list.
- **Re-check final wording:** synthesis wording is not re-verified against the sources; only the claims it is built from are.
- **Resilience:** back off longer on overloaded (529) errors; retry policy currently retries every exception, including deterministic ones.
- **Observability:** export traces over OTLP (for example to Langfuse); today spans go to `trace.jsonl`.
- **Housekeeping:** add CI (ruff, mypy, pytest) on the repository; remove the `source-terms` prompt variant, which had no measurable effect.
- **Milestone 8** above (CrewAI or Pydantic AI reimplementation with a trade-off comparison) was skipped. If revisited, the plan was Pydantic AI, returning the same `WorkflowResult` type so the existing eval harness could score both implementations head to head.
