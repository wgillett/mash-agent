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

- **Language:** Python.
- **Orchestration:** LangGraph.
- **Tools:** expose each data source as a thin, self-written **MCP server** with an explicit tool contract (typed inputs/outputs), rather than depending on community servers.
- **Tracing:** Langfuse or OpenTelemetry, with per-run token and cost tracking.
- **Packaging:** Dockerized; runnable with a single documented command.
- **Optional extension:** reimplement the same workflow in CrewAI or Pydantic AI and write a one-page comparison of trade-offs (state management, delegation model, observability, ergonomics).

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
├── Dockerfile
├── src/
│   ├── graph/             # LangGraph definition: supervisor, specialists, critic, synthesis
│   ├── agents/            # agent prompts and logic
│   ├── mcp_servers/       # pubmed, clinicaltrials, openfda MCP servers
│   ├── cache/             # on-disk cache
│   ├── ratelimit/         # per-tool limiters and backoff
│   └── observability/     # tracing and cost tracking
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

1. **Tools:** three MCP servers with typed contracts, caching, rate limiting, and tests.
2. **Specialists:** literature, trials, and regulatory agents returning source-tagged structured findings.
3. **Orchestration:** supervisor with parallel delegation, retries, timeouts, and failure isolation.
4. **Verification and synthesis:** critic agent, cited briefing, human approval gate.
5. **Observability:** tracing plus per-run token/cost/latency report.
6. **Evals:** fixed question set, metrics, and report.
7. **Polish:** Dockerization, README, architecture diagram, design-decisions doc.
8. **Optional:** CrewAI or Pydantic AI reimplementation and trade-off comparison.
