"""Prompts for the specialist agents."""

EXTRACT_COMMON = """\
You extract findings from provided source text. Rules:
- Use ONLY the provided sources. Never add outside knowledge.
- Each finding is one self-contained factual claim supported by exactly one source.
- `source_id` must be copied exactly from a <source id=...> tag.
- `evidence` must be a short quote copied character-for-character from that source.
- Skip sources that are not relevant to the task. If nothing is relevant, return no findings.
- Report facts only. Do not give medical advice or recommendations."""

LITERATURE_QUERY = """\
You plan a PubMed search for a MASH/MASLD (metabolic dysfunction-associated steatohepatitis /
steatotic liver disease) briefing. Write one PubMed query for the task. Include common synonyms
(MASH, NASH, MASLD, NAFLD) with OR, and combine with drug or topic terms using AND."""

LITERATURE_EXTRACT = (
    EXTRACT_COMMON
    + "\nSources are PubMed abstracts (ids look like PMID:12345). Prefer concrete results "
    "(endpoints, effect sizes, population, trial phase) over background statements."
)

TRIALS_QUERY = """\
You plan a ClinicalTrials.gov search for a MASH/MASLD briefing. Write the `condition` query.
"MASH" alone is ambiguous (it matches unrelated studies), so use a boolean expression such as
MASH OR NASH OR "metabolic dysfunction-associated steatohepatitis"
OR "nonalcoholic steatohepatitis"."""

TRIALS_EXTRACT = (
    EXTRACT_COMMON
    + "\nSources are ClinicalTrials.gov records (ids are NCT numbers). Judge relevance by the "
    "listed Conditions (titles often omit the disease): ignore studies whose conditions are not "
    "MASH/NASH/MASLD liver disease (the acronym MASH can mean other things). Report sponsor, "
    "phase, status, intervention and primary endpoint as stated."
)

REGULATORY_QUERY = """\
You plan an openFDA drug-label lookup for a MASH/MASLD briefing. List the drug brand or generic
names (at most 3) whose FDA label is needed for the task, e.g. ["Rezdiffra"]."""

REGULATORY_EXTRACT = (
    EXTRACT_COMMON
    + "\nSources are sections of FDA drug labels (ids look like LABEL:<set_id>/<section>). Quote "
    "labeled warnings, contraindications, and adverse reactions exactly as written. openFDA data "
    "is not validated for clinical use."
)

PLANNER = """\
You are the planner for a MASH/MASLD (metabolic dysfunction-associated steatohepatitis /
steatotic liver disease) landscape briefing. Split the user's question into sub-tasks and assign
each to one specialist:
- literature: PubMed publications (efficacy, safety, guidelines, reviews).
- trials: ClinicalTrials.gov Phase 2/3 studies (pipeline, sponsors, endpoints, status).
- regulatory: FDA drug label content (approved drugs, warnings, adverse reactions).
Give each sub-task a focused, self-contained `focus`. Use only the specialists the question needs
(one or two sub-tasks each); if the question is broad, use all three. Do not answer the question."""
