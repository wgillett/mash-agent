"""Render the final markdown briefing with inline citations and coverage notes."""

import re
from datetime import date
from urllib.parse import quote

from mash_agent.agents.models import SourceDoc
from mash_agent.agents.synthesis import Briefing

DISCLAIMER = (
    "This briefing is a demo and learning exercise, not a clinical tool, and is not medical "
    "advice. Data comes from public sources (PubMed, ClinicalTrials.gov, openFDA); openFDA data "
    "is not validated for clinical use. Claims were checked by an automated critic against "
    "retrieved source text and were approved by a person, but errors are possible."
)


def citation_url(source_id: str) -> str | None:
    if source_id.startswith("PMID:"):
        return f"https://pubmed.ncbi.nlm.nih.gov/{source_id.removeprefix('PMID:')}/"
    if re.fullmatch(r"NCT\d{8}", source_id):
        return f"https://clinicaltrials.gov/study/{source_id}"
    if source_id.startswith("LABEL:"):
        set_id = source_id.removeprefix("LABEL:").split("/", 1)[0]
        return f"https://api.fda.gov/drug/label.json?search={quote(f'set_id:"{set_id}"')}"
    return None


def cite(source_id: str) -> str:
    url = citation_url(source_id)
    return f"[{source_id}]({url})" if url else f"[{source_id}]"


def render_markdown(
    *,
    question: str,
    briefing: Briefing,
    sources: dict[str, SourceDoc],
    limitations: list[str],
    generated_on: date,
    scope: list[str] | None = None,
) -> str:
    lines = ["# MASH/MASLD landscape briefing", "", f"**Question:** {question}", ""]
    lines.append(f"*Generated {generated_on.isoformat()}. Not medical advice.*")
    for section in briefing.sections:
        lines += ["", f"## {section.heading}", ""]
        for b in section.bullets:
            refs = " ".join(cite(sid) for sid in b.source_ids)
            lines.append(f"- {b.text} {refs}")
    if scope:
        lines += ["", "## What was searched", ""]
        lines += [f"- {item}" for item in scope]
        lines.append(
            "- Only retrieved sources are covered; other therapies, trials or publications "
            "may exist."
        )
    lines += ["", "## Coverage and limitations", ""]
    lines += [f"- {item}" for item in limitations] or ["- No failures or dropped claims."]
    lines += ["", "## Sources cited", ""]
    for sid in briefing.cited_source_ids:
        title = sources[sid].title if sid in sources else "(source text not available)"
        lines.append(f"- {cite(sid)} - {title}")
    lines += ["", "---", "", f"*{DISCLAIMER}*", ""]
    return "\n".join(lines)
