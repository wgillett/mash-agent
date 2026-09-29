# Demo script

A step-by-step guide to presenting mash-agent live. It takes about 5 minutes of talking plus about 1 minute of waiting on the live run, and costs about $0.30 in API calls (a little more with the optional steps).

Full details are in the README. This script tells you what to do, in what order, and what to say.

## What this demo is about

The whole demo serves one idea: **no claim gets into the briefing unless a second model call has checked it against the source text it cites.** Everything you show either demonstrates that idea or shows how you know it works.

Short on time? With only two minutes, do steps 1, 2 and 4: run the example prompt, show the approval panel, show what the critic rejected.

## What you'll need

- A laptop with the repo cloned and [uv](https://docs.astral.sh/uv/) installed
- An Anthropic API key
- An email address for NCBI (PubMed asks for one; no key needed)
- A working network connection (PubMed, ClinicalTrials.gov and openFDA need no keys)
- Optional: Docker, as a backup (see "Running it in Docker" below)

## Before you start (the day before, or at least 15 minutes ahead)

1. Open a fresh terminal in the repo. Switch to `main` and run `git pull`.
2. Run `uv sync`.
3. Export `ANTHROPIC_API_KEY` and `NCBI_EMAIL` in *this* shell. Check the key with `echo ${ANTHROPIC_API_KEY:+set}`; it should print `set`.
4. **Do one full dry run** with `--out-dir rehearsal` (use the command from step 1 below). This warms the on-disk API cache so retrieval is fast during the demo, and it proves the key and the network work. Expect the wording and the trials to differ on the live run; that's normal.
5. Turn the terminal font size up. The approval panel is wide and the tables need room.
6. Open the README at "Latest results" in one tab and the architecture diagram in another.
7. Optional: keep the `eval_results/` folder handy, with the `summary.md` of the latest run. It's git-ignored, so it only exists on the laptop that ran the evals.

> **Note (zsh):** don't paste command blocks that contain `#` comments. zsh tries to run `#` as a command, and a `;` inside the comment then makes it try to run the rest too. Keep pasted commands comment-free.

## Running the demo

### Step 1: Run the example prompt (about 1 minute, about $0.30)

Run:

```bash
uv run mash-agent "Summarize the current state of late-stage MASH therapies and the safety information in resmetirom's label." --out-dir demo
```

While it runs, explain what's happening:

- "A planner splits the question and three specialists run in parallel: literature, trials, regulatory. Each one talks to one public API through its own MCP server."
- "Then a separate critic checks every single claim against the full text of the source it cites."
- "Then it stops and waits for me. Nothing is saved yet."

> **Quicker alternative:** `"What safety information does the FDA label for resmetirom contain?"` takes about 30 seconds and costs about $0.08. It only needs the regulatory specialist, so the audience won't see agents running in parallel. Use the long prompt when you want to show that.

### Step 2: Walk through the approval panel

When the panel appears, point out:

- the citation on every bullet (PMID, NCT ID, or label ID plus section)
- "What was searched", which states exactly which queries ran and that other relevant material may exist
- "Coverage and limitations", which states how many claims the critic excluded

Say: "This is exactly what would be written to disk. I approve or reject." Then type `y`.

The run then prints its tables: per-agent status and retries, per-stage tokens and cost, and per-tool time. Say: "Cost is an estimate from list prices, and it includes retries."

### Step 3: Show how it worked

Open the trace:

```bash
uv run mash-agent trace demo/trace.jsonl
```

Walk through the span tree: the parallel specialists, each attempt, and every LLM call and tool call with its tokens and cost. Failures would show in red. Say: "It's plain OpenTelemetry, written to a file. I could send it to Langfuse, but I didn't want the demo to need an account."

Then show the same cost table, loaded from the saved file:

```bash
uv run mash-agent report demo/run_report.json
```

### Step 4: Show what the critic rejected (the most interesting part)

Run:

```bash
uv run python -c "import json; d=json.load(open('demo/run_report.json')); [print(c['finding']['claim'], '->', c['critic_reason'], '\n') for c in d['critic']['checked'] if c['outcome']!='supported']"
```

This prints each excluded claim with the critic's reason. If nothing prints, that run had no exclusions; move on to step 5.

What you'll usually see: the extractor added something the source doesn't say, such as a trial's name or phase, or "biopsy-confirmed" where the source said "clinical evidence". Say: "These are subtle. The numbers are usually right; it's the extra detail that gets caught."

### Step 5: Show two contrasts (only if there's time)

**Rejection.** Run:

```bash
uv run mash-agent "What safety information does the FDA label for resmetirom contain?" --out-dir demo-reject
```

Answer `n` at the approval prompt. No `briefing.md` is written, but the run report is. Say: "Rejection is the default; only an explicit yes saves."

**An out-of-scope question.** Run:

```bash
uv run mash-agent "What is the capital of France?" --out-dir demo2
```

It verifies nothing, writes no briefing and exits with code 2. Say: "There's no separate relevance filter. It just can't find anything that survives the critic." Be honest that on a borderline question it could still return loosely related MASH claims.

### Step 6: Finish with the evals

Open the README "Latest results" table, or the local `summary.md`. Make three points:

- The critic passed **0 of 536** deliberately corrupted claims (changed numbers, flipped directions, swapped drugs, added mortality claims).
- A separate, different model (Opus) grades every claim independently, so the critic isn't grading itself.
- The extraction prompt was switched to `no-commentary` because two runs of the old prompt agreed closely, and the new prompt cut wasted claims from about 6.8% to about 2%.

Then state the caveats out loud. They make the results more credible, not less:

- These are single runs on 14 questions, so the intervals are optimistic.
- The judge is an LLM and hasn't been checked against hand labels.
- The evals measure accuracy, not whether a briefing is useful.

## Answers to common questions

- **Is this medical advice?** No. It's a demo, the footer of every briefing says so, and openFDA data isn't validated for clinical use.
- **Why not use an existing MCP server?** Verification depends on stable, source-tagged outputs with the source text intact, and writing the servers guarantees that. The reasoning is in `docs/design-decisions.md`.
- **Does the critic catch everything?** No. It caught every blunt corruption, but in the evals about 1% of claims that reach the briefing are still only partly supported (a handful of claims), and changing the prompt didn't change that.
- **What does it cost?** About $0.05 for a one-specialist question, up to about $0.40 for a broad one. A full eval run is about $5.
- **What's not done?** The trials specialist keeps some low-relevance old trials, and the judge isn't calibrated against hand labels. Both are listed under follow-ups in `AGENTS.md`.

## Running it in Docker

Use this if you can't rely on the laptop's Python setup. Build the image ahead of time:

```bash
docker build -t mash-agent .
```

Then run the demo with:

```bash
docker run --rm -it -e ANTHROPIC_API_KEY -e NCBI_EMAIL -v "$PWD/out:/out" -v mash-cache:/cache mash-agent "What safety information does the FDA label for resmetirom contain?"
```

Keep `-it`: the approval prompt needs it. Results land in `./out` on the laptop, even though the run prints container paths (`/out/...`).

## Troubleshooting

- **"ANTHROPIC_API_KEY is not set."** The key was exported in a different shell. Export it again in this one.
- **"Overloaded" from the Anthropic API.** This happens occasionally, and the pipeline retries. If it keeps happening, wait a minute and run again.
- **Docker: nothing to approve, or "Aborted!"** You left out `-it`. (`--auto-approve` also works, but it skips the most important part of the demo.)
- **Docker on Linux: permission denied writing `./out`.** Add `--user "$(id -u):$(id -g)"`.
- **The run is slow.** First-time retrieval is slower. The rehearsal run warms the cache.
- **The output differs from the rehearsal.** That's expected: the model and the trial retrieval vary from run to run. Don't promise specific claims in advance.

## At a glance

| Run | Time | Cost |
| --- | --- | --- |
| Label-only question | about 30 s | about $0.08 |
| Example prompt (all three specialists) | about 45 to 60 s | about $0.30 |
| Full eval run (with judge and canary) | about 6 min | about $5 |

Times don't include however long you spend reviewing at the approval gate.
