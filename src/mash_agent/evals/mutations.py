"""Deterministic claim corruptions with a known expected verdict (``unsupported``).

Used to measure the critic's miss rate: a mutated claim is not stated by the source, so a critic
that returns ``supported`` for it is wrong.

To make that hold, mutations that *introduce* content (a drug name, a number, a mortality claim)
only introduce content that does not occur in the source text; when that is impossible they do
not apply. Direction flips cannot be checked this way (their truth depends on context), so a
flip could in principle yield a claim the source also supports; review such misses by hand.
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


def _in_source(token: str, source: str) -> bool:
    """Whole-token match; a trailing sentence period is not part of a number."""
    pattern = rf"(?<!\w)(?<!\d\.){re.escape(token)}(?!\w)(?!\.\d)"
    return re.search(pattern, source, flags=re.I) is not None


def bump_number(claim: str, source: str = "") -> str | None:
    """Change the first standalone number to one that does not appear in the source."""
    m = _NUMBER.search(claim)
    if not m:
        return None
    text = m.group(1)
    if "." in text:
        decimals = len(text.split(".")[1])
        candidates = [f"{float(text) * k:.{decimals}f}" for k in (1.5, 2.0, 3.0)]
    else:
        n = int(text)
        candidates = [str(n * 2 if n > 1 else n + 3), str(n * 3 + 1), str(n * 7 + 5)]
    new = next((c for c in candidates if c != text and not _in_source(c, source)), None)
    if new is None:
        return None
    return claim[: m.start(1)] + new + claim[m.end(1) :]


def flip_direction(claim: str, source: str = "") -> str | None:
    lowered = claim.lower()
    for old, new in FLIPS:
        i = lowered.find(old)
        if i != -1:
            return claim[:i] + new + claim[i + len(old) :]
    return None


def swap_drug(claim: str, source: str = "") -> str | None:
    """Replace a named drug with one that appears in neither the claim nor the source."""
    lowered = claim.lower()
    for drug in DRUGS:
        i = lowered.find(drug)
        if i != -1:
            other = next(
                (d for d in DRUGS if d != drug and d not in lowered and not _in_source(d, source)),
                None,
            )
            if other is None:
                return None
            return claim[:i] + other + claim[i + len(drug) :]
    return None


def add_overclaim(claim: str, source: str = "") -> str | None:
    if "mortality" in source.lower():
        return None
    return claim.rstrip().rstrip(".") + ", and this reduced all-cause mortality."


MUTATIONS: dict[str, Callable[[str, str], str | None]] = {
    "bump_number": bump_number,
    "flip_direction": flip_direction,
    "swap_drug": swap_drug,
    "add_overclaim": add_overclaim,
}


def mutate(claim: str, source: str = "") -> dict[str, str]:
    """All applicable mutations of ``claim`` given its source text, keyed by mutation name."""
    out: dict[str, str] = {}
    for name, fn in MUTATIONS.items():
        mutated = fn(claim, source)
        if mutated is not None and mutated != claim:
            out[name] = mutated
    return out
