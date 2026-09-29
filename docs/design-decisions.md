# Design decisions

Short notes on non-obvious choices. Extended as milestones land.

## Supervisor / specialist architecture (Milestone 3)

- **Why a supervisor with specialists:** each data source has different query syntax, rate limits
  and failure modes. A specialist owns one source and returns findings tagged with a source ID, so
  a failure or bad query in one source is contained there.
- **Why LangGraph fan-out (`Send`):** the planner emits N sub-tasks and the graph runs one
  specialist node per task in parallel (`max_concurrency` bounds it). Two tasks for the same
  specialist are allowed (for example, two drugs).
- **Failure isolation:** every specialist call goes through `run_resilient`: a per-attempt timeout,
  retries with doubling backoff, and a result that records the error instead of raising. A failed
  specialist becomes an `AgentOutcome(status="failed")` with its attempts and error; the run
  continues with the others. Run status is `ok`, `partial` (some failed) or `failed` (all failed).
- **Planner failure:** if planning fails or returns nothing usable, the supervisor falls back to a
  default plan (every specialist gets the whole question) and records a note. Plans are capped at
  six tasks.
- **Rate limits under parallelism:** limits live in the tool layer (one limiter per API client),
  not in the agents, so parallel specialists share them.
- **Retry policy:** all exceptions are retried up to `max_attempts` (default 3). This is coarse:
  deterministic errors are retried too. It is acceptable at this scale; revisit if cost matters.
- **Not yet done:** the critic, synthesis, human approval gate, and cost tracking (later
  milestones); the run report currently covers attempts, latency and token counts only.

## Verification, synthesis and the approval gate (Milestone 4)

- **Why a separate critic:** the specialist that proposed a claim is the worst judge of it. The
  critic is a different call with a different prompt and sees only the full source text and the
  claim, not the extractor's quote, so it cannot anchor on it. `evidence_verified` (the quote is
  in the source) is a cheap mechanical check; it does not prove the quote supports the claim.
- **Fail closed:** a claim reaches the briefing only if the critic returns `supported`. Claims
  the critic marks `unsupported`, could not check (call failed, no verdict, source never
  retrieved) or whose verdict is missing are excluded and counted in the briefing's
  "Coverage and limitations" section. The critic runs one call per distinct source, in parallel,
  with the same timeout/retry wrapper as the specialists.
- **Provenance is structural:** synthesis asks the model only to group and word bullets, each
  listing claim numbers. Citations are rendered in code from those numbers. A bullet with no
  valid number is dropped, and verified claims left uncited are appended so none are lost. If
  synthesis fails, claims are listed grouped by specialist. Known limitation: the model can still
  paraphrase a cited claim loosely; nothing re-checks the final wording against the sources.
- **Human approval gate:** implemented with LangGraph `interrupt()` and an in-memory checkpointer,
  so the graph genuinely pauses after synthesis and resumes with the person's decision. The person
  sees the exact markdown that would be published. Rejection, or a run where nothing passed the
  critic, writes no `briefing.md`; `run_report.json` (plan, per-agent outcomes, every critic
  verdict and reason, token usage per stage) is always written as the trace.
- **Token accounting:** usage is recorded per stage (planner, each specialist, critic,
  synthesis). Tokens spent on failed attempts are not captured.

## Observability, cost and the CLI (Milestone 5)

- **OpenTelemetry, not Langfuse:** the code is instrumented with the OpenTelemetry API only.
  Langfuse needs an account or a hosted service, which a demo repo should not require; it can
  ingest OTLP, so exporting there later means adding an exporter, not changing agent code.
  Today spans go to `trace.jsonl` (one JSON object per span). `mash-agent trace` renders the span
  tree; failures are red. The provider is held in `observability/tracing.py` rather than set
  globally, so tests and the CLI each configure their own, and with tracing off every `span()` is
  a no-op.
- **What is traced:** run, plan, each specialist, every retry attempt, every LLM call (model,
  stage, tokens, cost), every tool call, the critic (per source), synthesis, and the approval
  decision. Failures we turn into data (a failed specialist, a rejected briefing) are marked as
  errors on their spans, not only raised exceptions.
- **Stage attribution without plumbing:** context variables record which stage is running and
  which meter is active. They propagate into LangGraph's parallel node tasks, so an LLM call
  deep inside a specialist is attributed to that specialist. A test asserts every span shares
  one trace and has a parent, because this is the easiest thing to break silently.
- **One place for bookkeeping:** `InstrumentedLLM` and `instrument_tools` wrap the LLM and tool
  seams, so agents contain no timing or token code. Calls that fail or get retried are recorded
  too: a structured-output error carries the tokens it cost, so retries no longer disappear from
  the totals (the earlier per-stage counts missed them). A call cancelled by a timeout is
  recorded with no tokens, since none are known.
- **Cost:** tokens are priced from a table of list prices (`observability/pricing.py`, dated),
  overridable with `MASH_AGENT_PRICES`. An unknown model yields no cost, never a guess. Prompt
  caching is not used, so cache prices are not modelled. Cost is an estimate from list prices,
  not a bill.
- **Run summary:** `run_report.json` includes a summary by stage, agent and tool (tokens, cost,
  attempts, latency, failures). LLM time per stage is summed across parallel calls, so it can
  exceed wall time.
- **CLI:** Click group; `mash-agent "question"` routes to `run` when the first argument is not a
  known command or option (so a question that is exactly `run`, `report` or `trace` needs the
  explicit `run` form). Rich formats terminal output only (the briefing at the gate, the
  summary tables, a live status line); files stay plain, and output has no colour codes when not
  attached to a terminal. The status line is paused while the approval prompt is shown. HTTP
  library logging is off unless `-v`.

## Evaluation (Milestone 6)

- **One command:** `mash-agent eval run` (or `python evals/run_evals.py`) runs a fixed question
  set (`evals/questions.yaml`, 14 questions across regulatory, trials, literature, mixed and two
  safety cases) end to end and writes `report.json` (machine-readable), `summary.md`,
  `runs/<question>.json` and a `spotcheck.jsonl` sample. `mash-agent eval compare A B` diffs two
  runs. It asks before spending money; `--yes` skips that. Results land in `eval_results/`
  (git-ignored); commit a `summary.md` deliberately when you want it recorded.
- **An independent judge:** the critic cannot grade itself. Every claim the specialists propose
  is also graded by a *judge* (a different model by default, `claude-opus-5-5`, and a different,
  graded prompt: supported / partial / unsupported) against the same source text, before the
  critic's exclusions are applied. That gives, against the judge: the unsupported rate before the
  critic, the residual rate in what reaches the briefing (after), how many unsupported claims the
  critic caught or missed, and how many good claims it wrongly excluded. Strict citation accuracy
  counts only fully supported claims; lenient also counts partial. The judge is an LLM, so it is
  an independent check, not ground truth: `spotcheck.jsonl` (disagreements first) is there for
  hand-labelling a sample.
- **Known-truth canary as a second view:** the corrupted-claim test (see Milestone 4) runs on
  every question's claims and its miss rate is aggregated. It does not depend on the judge.
- **Deterministic checks:** structural guarantees of each briefing (every bullet cites a source,
  every cited source was retrieved, every claim was critic-passed), advice-like phrases, topic
  coverage terms, and out-of-scope handling (`expect_no_claims`). Coverage terms are a coarse
  proxy, not correctness.
- **Honest statistics:** every rate is reported with counts and a 95% Wilson interval. `compare`
  reports the difference between two runs with a Newcombe interval and says whether it excludes
  zero. (An earlier version called a difference "noise" whenever the two runs' intervals overlapped;
  that is too conservative, since overlapping intervals can still hide a real difference.) Claims are
  not independent: the same claim recurs across questions and several come from one source, so
  intervals are optimistic. Model output varies run to run; repeat before acting on small gaps.
  Runs within a day also share the on-disk API cache, so variants see the same retrieved records
  when their queries match.
- **Failures are data:** a question that crashes is recorded as `crashed`; unjudgeable claims are
  `unjudged` and excluded from rates (never counted as right or wrong); costs are reported for the
  system and separately for eval overhead (judge, canary).
- **Prompt variants:** named additions to the extraction prompt (`EXTRACT_VARIANTS`) can be
  evaluated with `--variant`. A variant becomes the default only if the evals show it helps
  without cutting useful claims. Two are defined: `source-terms` (keep the source's own
  terminology; motivated by a real run where the extractor wrote "NAFLD/MASH" over a source that
  said NAFLD and the critic rightly excluded it) and `quote-anchored` (every number, drug,
  population and qualifier in a claim must appear in its quote), and `no-commentary` (state only
  what the source states: no interpretation, grouping, parenthetical asides, or claims about what a
  source does not say or about other sources). The third was added after the first full baseline
  run, in which the critic excluded 19 of 259 claims: about half were interpretive glosses ("the
  adverse reactions are gastrointestinal") or absence and cross-source claims ("the provided
  Rezdiffra sections contain no GLP-1 information"), and most of the rest added outside knowledge
  (a trial's name or phase, "biopsy-confirmed" where the source said "clinical evidence").
- **What the first baseline showed** (14 questions, 259 claims): the judge rated no claim
  unsupported, so critic recall on unsupported claims is undefined; 13 claims (5.0%) were partial,
  the critic excluded 11 of them, and 2 reached the briefing (0.8% residual). The critic passed
  0 of 536 deliberately corrupted claims. The critic was stricter than the judge on 8 claims (mostly
  glosses and absence claims). The informative metrics are therefore the not-fully-supported rate
  before and after the critic and the exclusion rate, not the unsupported rate.
