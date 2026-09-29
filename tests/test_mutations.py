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
