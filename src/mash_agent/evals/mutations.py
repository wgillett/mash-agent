"""Deterministic claim corruptions with a known expected verdict (``unsupported``).

Used to measure the critic's miss rate: a mutated claim is not stated by the source, so a critic
that returns ``supported`` for it is wrong. Heuristic: a mutation could, rarely, produce a
sentence the source happens to state, so review misses by hand.
"""

import re
from collections.abc import Callable

DRUGS = (
    "resmetirom",
    "semaglutide",
    "tirzepatide",
    "survodutide",
    "pegozafermin",
    "efruxifermin",
    "lanifibranor",
    "pemvidutide",
    "selonsertib",
)

# Direction / status flips. Longer phrases first so they win over their substrings.
FLIPS: tuple[tuple[str, str], ...] = (
    ("more often", "less often"),
    ("not significantly", "significantly"),
    ("significantly better", "significantly worse"),
    ("higher", "lower"),
    ("lower", "higher"),
    ("increased", "decreased"),
    ("decreased", "increased"),
    ("improved", "worsened"),
    ("was terminated", "was completed"),
    ("is recruiting", "has completed enrollment"),
    ("is not yet recruiting", "is recruiting"),
)

_NUMBER = re.compile(r"(?<![\w.:/-])(\d+(?:\.\d+)?)(?![\w/-])")


def bump_number(claim: str) -> str | None:
    """Change the first standalone number (skips IDs like NCT0123... and PMID:1)."""
    m = _NUMBER.search(claim)
    if not m:
        return None
    text = m.group(1)
    if "." in text:
        new = f"{float(text) * 1.5:.{len(text.split('.')[1])}f}"
    else:
        n = int(text)
        new = str(n * 2 if n > 1 else n + 3)
    if new == text:
        return None
    return claim[: m.start(1)] + new + claim[m.end(1) :]


def flip_direction(claim: str) -> str | None:
    lowered = claim.lower()
    for old, new in FLIPS:
        i = lowered.find(old)
        if i != -1:
            return claim[:i] + new + claim[i + len(old) :]
    return None


def swap_drug(claim: str) -> str | None:
    lowered = claim.lower()
    for drug in DRUGS:
        i = lowered.find(drug)
        if i != -1:
            other = next(d for d in DRUGS if d != drug and d not in lowered)
            return claim[:i] + other + claim[i + len(drug) :]
    return None


def add_overclaim(claim: str) -> str | None:
    return claim.rstrip().rstrip(".") + ", and this reduced all-cause mortality."


MUTATIONS: dict[str, Callable[[str], str | None]] = {
    "bump_number": bump_number,
    "flip_direction": flip_direction,
    "swap_drug": swap_drug,
    "add_overclaim": add_overclaim,
}


def mutate(claim: str) -> dict[str, str]:
    """All applicable mutations of ``claim``, keyed by mutation name."""
    out: dict[str, str] = {}
    for name, fn in MUTATIONS.items():
        mutated = fn(claim)
        if mutated is not None and mutated != claim:
            out[name] = mutated
    return out
