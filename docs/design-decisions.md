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
