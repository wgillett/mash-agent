"""Run the evaluation: `uv run python evals/run_evals.py [options]`.

Equivalent to `uv run mash-agent eval run [options]`; see `--help` for options. It calls paid APIs
(the system's model, an independent judge model, and the three public data APIs).
"""

import sys

from mash_agent.cli import cli

if __name__ == "__main__":
    cli(["eval", "run", *sys.argv[1:]], prog_name="mash-agent")
