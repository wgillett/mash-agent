"""Shared data contracts for specialist agents."""

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["pubmed", "clinicaltrials", "openfda"]


class Usage(BaseModel):
    """Token counts for one or more calls.

    ``input_tokens`` is the total input, including the cache tokens; ``cache_read_tokens`` and
    ``cache_creation_tokens`` are the parts of it served from, or written to, the prompt cache.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )


class SubTask(BaseModel):
    """A unit of work the supervisor hands to one specialist."""

    focus: str = Field(description="What this specialist should find out, in plain language.")


class SourceDoc(BaseModel):
    """Retrieved source text. ``source_id`` is the citation key the critic later checks against.

    Formats: ``PMID:<n>``, ``<NCT id>``, ``LABEL:<set_id>/<section>``.
    """

    source_id: str
    source_type: SourceType
    title: str
    text: str


class RawFinding(BaseModel):
    """What the LLM proposes; validated by the specialist before becoming a ``Finding``."""

    claim: str = Field(description="One self-contained factual statement supported by the source.")
    source_id: str = Field(description="Exactly one source_id copied from the provided sources.")
    evidence: str = Field(
        description="A short verbatim quote (copied exactly) from that source supporting the claim."
    )


class ExtractedFindings(BaseModel):
    findings: list[RawFinding]


class Finding(BaseModel):
    agent: str
    claim: str
    source_id: str
    evidence: str
    evidence_verified: bool = Field(
        description="True if ``evidence`` occurs verbatim in the source (ignoring case/whitespace)."
    )


class SpecialistResult(BaseModel):
    agent: str
    subtask: SubTask
    queries: list[str]
    sources: list[SourceDoc]
    findings: list[Finding]
    dropped: list[str] = Field(
        default_factory=list, description="Why proposed findings were dropped."
    )
    usage: Usage = Field(default_factory=Usage)
