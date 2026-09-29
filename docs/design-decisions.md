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
