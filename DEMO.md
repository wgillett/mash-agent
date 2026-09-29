# Demo script (notes to self)

How I run the mash-agent demo. About 5 minutes of talking, about 1 minute of waiting on the live run. Full details are in the README; this is just the order I do things in.

## The point of the demo

One idea: **no claim gets into the briefing unless a second model call has checked it against the source text it cites.** Everything I show is either that idea working, or how I know it works.

If I only get two minutes: run the example prompt, show the approval panel, show what the critic rejected.

## Before I start (the day before, or 15 minutes ahead)

- [ ] Fresh terminal in the repo, on `main`, `git pull` done
- [ ] `uv sync` done
- [ ] `ANTHROPIC_API_KEY` and `NCBI_EMAIL` exported in *this* shell (check with `echo ${ANTHROPIC_API_KEY:+set}`)
- [ ] Network works (PubMed, ClinicalTrials.gov and openFDA need no keys)
- [ ] **Do one full dry run** with `--out-dir rehearsal`. It warms the on-disk API cache so retrieval is fast on stage, and it proves the key and the network work. The wording and the trials will differ on the live run; that's normal.
- [ ] Font size up. The approval panel is wide and the tables need room.
- [ ] Have the README open at "Latest results" and the architecture diagram in another tab
- [ ] Optional: `eval_results/` folder handy (it's git-ignored, so only on this laptop) with the `summary.md` of the latest run

**zsh gotcha:** don't paste command blocks that contain `#` comments. zsh tries to run `#` as a command, and a `;` inside the comment then makes it try to run the rest too. Keep pasted commands comment-free.

## The demo

### 1. Run the example prompt (about 1 minute, about $0.30)

```bash
uv run mash-agent "Summarize the current state of late-stage MASH therapies and the safety information in resmetirom's label." --out-dir demo
```

What I say while it runs:

- "A planner splits the question and three specialists run in parallel: literature, trials, regulatory. Each one talks to one public API through its own MCP server."
- "Then a separate critic checks every single claim against the full text of the source it cites."
- "Then it stops and waits for me. Nothing is saved yet."

If I want something quicker: `"What safety information does the FDA label for resmetirom contain?"` takes about 30 seconds and costs about $0.08. It only needs the regulatory specialist, so the parallelism isn't visible. Use the long prompt when I want to show parallel agents.

### 2. The approval panel

When the panel appears, point at:

- a citation on every bullet (PMID, NCT ID, or label ID plus section)
- "What was searched": says exactly which queries ran, and that other things may exist
- "Coverage and limitations": says how many claims the critic excluded

Say: "This is exactly what would be written to disk. I approve or reject." Then type `y`.

Then the tables print: per-agent status and retries, per-stage tokens and cost, per-tool time. Say: "Cost is an estimate from list prices, and it includes retries."

### 3. How it worked

```bash
uv run mash-agent trace demo/trace.jsonl
```

The span tree: parallel specialists, each attempt, every LLM call and tool call with tokens and cost. Failures would be red. Say: "It's plain OpenTelemetry, written to a file. I could send it to Langfuse, but I didn't want the demo to need an account."

```bash
uv run mash-agent report demo/run_report.json
```

Same cost table, from the saved file.

### 4. What the critic rejected (my favourite part)

```bash
uv run python -c "import json; d=json.load(open('demo/run_report.json')); [print(c['finding']['claim'], '->', c['critic_reason'], '\n') for c in d['critic']['checked'] if c['outcome']!='supported']"
```

This prints the excluded claims with the critic's reasons. If nothing prints, that run had no exclusions; skip to the next step.

The pattern I've seen: the extractor adds something the source doesn't say (a trial's name or phase, "biopsy-confirmed" where the source said "clinical evidence"). Say: "These are subtle. The numbers are usually right; it's the extra detail that gets caught."

### 5. Two short contrasts (only if there's time)

Reject:

```bash
uv run mash-agent "What safety information does the FDA label for resmetirom contain?" --out-dir demo-reject
```

Answer `n`. No `briefing.md` gets written, but the run report is. Say: "Rejection is the default; only an explicit yes saves."

Out of scope:

```bash
uv run mash-agent "What is the capital of France?" --out-dir demo2
```

Verifies nothing, writes no briefing, exit code 2. Say: "There's no separate relevance filter. It just can't find anything that survives the critic." (Honest caveat: on a borderline question it could still return loosely related MASH claims.)

### 6. How I know it works: the evals

Open the README "Latest results" table (or the local `summary.md`).

- The critic passed **0 of 536** deliberately corrupted claims (changed numbers, flipped directions, swapped drugs, added mortality claims).
- A separate, different model (Opus) grades every claim independently, so the critic isn't grading itself.
- I changed the extraction prompt to `no-commentary` because two runs of the old prompt agreed closely, and the new prompt cut wasted claims from about 6.8% to about 2%.

Say the caveats out loud. They make it more credible:

- Single runs on 14 questions, so the intervals are optimistic.
- The judge is an LLM and hasn't been checked against hand labels.
- The evals measure accuracy, not whether a briefing is useful.

## If someone asks

- **Is this medical advice?** No. It's a demo, the footer of every briefing says so, and openFDA data isn't validated for clinical use.
- **Why not use an existing MCP server?** I wanted stable, source-tagged outputs with the source text intact, because the verification depends on it. It's in `docs/design-decisions.md`.
- **Does the critic catch everything?** No. It caught every blunt corruption, but about 1% of claims that reach the briefing are still only partly supported in the evals (a handful of claims), and it didn't change with the prompt.
- **Cost?** About $0.05 for a one-specialist question, up to about $0.40 for a broad one. A full eval run is about $5.
- **What's not done?** The trials specialist keeps some low-relevance old trials. The judge isn't calibrated by hand. Both are listed in `AGENTS.md` under follow-ups.

## Docker version (if I can't rely on the laptop's Python setup)

```bash
docker build -t mash-agent .
```

```bash
docker run --rm -it -e ANTHROPIC_API_KEY -e NCBI_EMAIL -v "$PWD/out:/out" -v mash-cache:/cache mash-agent "What safety information does the FDA label for resmetirom contain?"
```

`-it` is required for the approval prompt. Results land in `./out` on the laptop (the run prints the container paths, `/out/...`).

## If something goes wrong

- **"ANTHROPIC_API_KEY is not set."** Exported in a different shell. Export it again here.
- **"Overloaded" from the Anthropic API.** Happens occasionally. The pipeline retries. If it keeps happening, wait a minute and re-run.
- **Docker: nothing to approve / "Aborted!"** Forgot `-it`. (Or use `--auto-approve`, but that skips the best part of the demo.)
- **Docker on Linux, permission denied writing `./out`.** Add `--user "$(id -u):$(id -g)"`.
- **The run looks slow.** First-time retrieval is slower; the rehearsal run warms the cache.
- **Output differs from what I rehearsed.** Expected. The model and the trial retrieval vary run to run. Don't promise specific claims in advance.

## Cheat sheet

| | Time | Cost |
|---|---|---|
| Label-only question | about 30 s | about $0.08 |
| Example prompt (all three specialists) | about 45 to 60 s | about $0.30 |
| Full eval run (with judge and canary) | about 6 min | about $5 |

(Times exclude however long I spend reviewing at the approval gate.)
