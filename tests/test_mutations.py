from mash_agent.evals.mutations import (
    add_overclaim,
    bump_number,
    flip_direction,
    mutate,
    swap_drug,
)


def test_bump_number_changes_first_standalone_number_only() -> None:
    assert bump_number("Diarrhea was 14 with placebo and 23 with 80 mg.") == (
        "Diarrhea was 28 with placebo and 23 with 80 mg."
    )
    assert bump_number("Rate was 31.3% in group A.") == "Rate was 47.0% in group A."
    assert bump_number("Trial NCT07701993 is recruiting.") is None  # IDs are not numbers
    assert bump_number("No digits here.") is None
    assert bump_number("Only 1 patient.") == "Only 4 patient."


def test_flip_direction_prefers_specific_phrases() -> None:
    assert flip_direction("Events occurred more often with drug.") == (
        "Events occurred less often with drug."
    )
    assert flip_direction("It was not significantly different.") == (
        "It was significantly different."
    )
    assert flip_direction("The trial was terminated.") == "The trial was completed."
    assert flip_direction("Nothing to flip.") is None


def test_swap_drug_picks_a_different_drug() -> None:
    assert swap_drug("Resmetirom reduced liver fat.") == "semaglutide reduced liver fat."
    # matches in DRUGS order and never substitutes a drug already named in the claim
    assert swap_drug("Semaglutide beat placebo; resmetirom did not.") == (
        "Semaglutide beat placebo; tirzepatide did not."
    )
    assert swap_drug("A study of an unnamed agent.") is None


def test_add_overclaim_appends_a_mortality_claim() -> None:
    assert add_overclaim("Drug X improved fibrosis.") == (
        "Drug X improved fibrosis, and this reduced all-cause mortality."
    )


def test_mutate_returns_only_applicable_and_changed_claims() -> None:
    out = mutate("Resmetirom lowered LDL by 20% in 900 patients.")
    assert set(out) == {"bump_number", "flip_direction", "swap_drug", "add_overclaim"}
    assert all(m != "Resmetirom lowered LDL by 20% in 900 patients." for m in out.values())
    assert set(mutate("A plain statement.")) == {"add_overclaim"}


def test_swap_drug_never_introduces_a_drug_the_source_mentions() -> None:
    # The real-world miss: the source lists tirzepatide as significant, so swapping it in
    # produced a claim that was also true.
    source = "Tirzepatide and semaglutide were significantly better than placebo."
    out = swap_drug("Resmetirom and semaglutide were significantly better than placebo.", source)
    assert out is not None and "tirzepatide" not in out.lower()
    assert out == "survodutide and semaglutide were significantly better than placebo."


def test_swap_drug_does_not_apply_when_every_replacement_is_in_the_source() -> None:
    from mash_agent.evals.mutations import DRUGS

    assert swap_drug("Resmetirom worked.", " ".join(DRUGS)) is None


def test_bump_number_avoids_numbers_present_in_the_source() -> None:
    # 14 -> 28 would coincide with a number the source states, so the next candidate is used
    assert bump_number("Rate was 14.", source="Rates were 14 and 28.") == "Rate was 43."
    assert bump_number("Rate was 14.", source="14 28 43 103") is None


def test_add_overclaim_skipped_when_source_discusses_mortality() -> None:
    assert add_overclaim("X improved fibrosis.", "All-cause mortality was reported.") is None


def test_mutate_passes_source_through() -> None:
    out = mutate("Resmetirom cut 14 events.", "tirzepatide semaglutide 28 mortality")
    assert out["swap_drug"] == "survodutide cut 14 events."
    assert out["bump_number"] == "Resmetirom cut 43 events."
    assert "add_overclaim" not in out


def test_source_number_matching_handles_periods_and_decimals() -> None:
    from mash_agent.evals.mutations import _in_source

    assert _in_source("28", "Rates were 14 and 28.")  # sentence-final period
    assert not _in_source("14", "The mean was 3.14 overall.")  # part of a decimal
    assert not _in_source("28", "n=280 patients")  # part of a longer number
    assert _in_source("tirzepatide", "Tirzepatide, semaglutide")  # case-insensitive
