"""Rich renderers for the run summary and the JSONL trace. Terminal output only."""

import json
from pathlib import Path
from typing import Any

from rich.console import Group, RenderableType
from rich.table import Table
from rich.tree import Tree

from mash_agent.observability.summary import RunSummary


def fmt_cost(cost: float | None) -> str:
    return "n/a" if cost is None else f"${cost:.4f}"


def summary_renderable(s: RunSummary) -> RenderableType:
    head = Table.grid(padding=(0, 2))
    head.add_row("run", s.run_id, "model", s.model)
    head.add_row(
        "status", f"{s.status} (agents: {s.agents_status})", "wall time", f"{s.wall_time_s:.1f}s"
    )
    head.add_row(
        "tokens",
        f"{s.input_tokens:,} in / {s.output_tokens:,} out",
        "cost",
        fmt_cost(s.cost_usd),
    )
    head.add_row(
        "LLM calls", f"{s.llm_calls} ({s.failed_llm_calls} failed)", "agent retries", str(s.retries)
    )

    agents = Table(title="Agents", title_justify="left")
    for col in ("agent", "status", "attempts", "latency", "findings", "note"):
        agents.add_column(
            col, justify="right" if col in {"attempts", "latency", "findings"} else "left"
        )
    for a in s.agents:
        status = f"[green]{a.status}[/]" if a.status == "ok" else f"[red]{a.status}[/]"
        agents.add_row(
            a.agent, status, str(a.attempts), f"{a.latency_s:.1f}s", str(a.findings), a.error or ""
        )

    stages = Table(title="Stages (LLM)", title_justify="left")
    for col in ("stage", "calls", "failed", "tokens in", "tokens out", "cost", "llm time"):
        stages.add_column(col, justify="left" if col == "stage" else "right")
    for r in s.stages:
        stages.add_row(
            r.stage,
            str(r.calls),
            str(r.failed_calls),
            f"{r.input_tokens:,}",
            f"{r.output_tokens:,}",
            fmt_cost(r.cost_usd),
            f"{r.latency_s:.1f}s",
        )

    parts: list[RenderableType] = [head, agents, stages]
    if s.tools:
        tools = Table(title="Tools", title_justify="left")
        for col in ("tool", "calls", "failed", "time"):
            tools.add_column(col, justify="left" if col == "tool" else "right")
        for t in s.tools:
            tools.add_row(t.tool, str(t.calls), str(t.failed_calls), f"{t.latency_s:.1f}s")
        parts.append(tools)
    return Group(*parts)


def load_spans(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _label(span: dict[str, Any]) -> str:
    a = span["attributes"]
    bits = [f"{span['duration_ms'] / 1000:.2f}s"]
    if span["name"] == "llm.generate":
        bits += [
            str(a.get("mash.schema", "")),
            f"{a.get('gen_ai.usage.input_tokens', 0)}→{a.get('gen_ai.usage.output_tokens', 0)} tok",
        ]
        if "mash.cost_usd" in a:
            bits.append(fmt_cost(a["mash.cost_usd"]))
    elif span["name"] == "tool.call":
        bits.append(str(a.get("mash.tool", "")))
    elif span["name"] == "specialist":
        bits += [str(a.get("mash.agent", "")), str(a.get("mash.status", ""))]
    elif span["name"] == "attempt":
        bits.append(f"attempt {a.get('mash.attempt')}/{a.get('mash.max_attempts')}")
    elif "mash.source_id" in a:
        bits.append(str(a["mash.source_id"]))
    text = f"{span['name']}  " + "  ".join(b for b in bits if b)
    if span["status"] == "ERROR":
        text = f"[red]{text}  ✗ {span['status_description'] or 'error'}[/]"
    return text


def trace_tree(spans: list[dict[str, Any]]) -> Tree:
    """Span tree ordered by start time; failed spans are red."""
    by_parent: dict[str | None, list[dict[str, Any]]] = {}
    ids = {s["span_id"] for s in spans}
    for s in sorted(spans, key=lambda s: s["start_ns"]):
        parent = s["parent_id"] if s["parent_id"] in ids else None
        by_parent.setdefault(parent, []).append(s)

    root = Tree("trace")

    def add(node: Tree, parent_id: str | None) -> None:
        for s in by_parent.get(parent_id, []):
            add(node.add(_label(s)), s["span_id"])

    add(root, None)
    return root
