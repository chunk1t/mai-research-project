"""The prose-claim guard in `scripts/verify_pack.py`.

The ledger check covers every headline VALUE, but one headline claim is a
COUNT over the nine paired comparisons -- "six of nine differences are
statistically significant" -- and a count cannot be expressed as a ledger row.
That sentence was wrong for a day, saying five while the table beside it
correctly marked six, and the ledger could not have caught it.

Expected counts here are hand-computed from the fixture below, not read back
from the function under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from verify_pack import check_of_nine_claims  # noqa: E402

# Nine comparisons. Counted by hand: rows 1, 3, 4 and 8 are statistically
# significant -> 4. Row 4 alone is practically significant -> 1.
COMPARISONS = [
    {"statistically_significant": True, "practically_significant": False},
    {"statistically_significant": False, "practically_significant": False},
    {"statistically_significant": True, "practically_significant": False},
    {"statistically_significant": True, "practically_significant": True},
    {"statistically_significant": False, "practically_significant": False},
    {"statistically_significant": False, "practically_significant": False},
    {"statistically_significant": False, "practically_significant": False},
    {"statistically_significant": True, "practically_significant": False},
    {"statistically_significant": False, "practically_significant": False},
]
N_STAT = 4
N_PRAC = 1


def test_correct_statistical_count_passes():
    text = "Four of nine differences are statistically significant."
    assert check_of_nine_claims(text, COMPARISONS) == []


def test_wrong_statistical_count_is_reported():
    text = "Five of nine differences are statistically significant."
    failures = check_of_nine_claims(text, COMPARISONS)
    assert len(failures) == 1
    assert "says 5" in failures[0] and str(N_STAT) in failures[0]


def test_practical_claim_is_checked_against_the_practical_count():
    assert check_of_nine_claims(
        "One of nine clears its practical threshold.", COMPARISONS
    ) == []
    assert len(check_of_nine_claims(
        "Four of nine clears its practical threshold.", COMPARISONS
    )) == 1


def test_both_counts_in_one_sentence_resolve_by_which_marker_comes_first():
    """The real README sentence names both counts; the first marker wins.

    A keyword-anywhere test read this as a practical claim and reported a false
    mismatch on a correct sentence.
    """
    text = (
        "Four of nine paired differences are statistically significant and none "
        "reaches its pre-declared practical threshold."
    )
    assert check_of_nine_claims(text, COMPARISONS) == []


def test_a_marker_split_across_a_line_break_is_still_matched():
    """These documents hard-wrap, so "exclude zero" is routinely split."""
    text = "Four of nine comparisons exclude\nzero; none clears its practical threshold."
    assert check_of_nine_claims(text, COMPARISONS) == []


def test_digit_form_is_read_as_a_count():
    assert check_of_nine_claims("4 of nine comparisons exclude zero.", COMPARISONS) == []
    assert len(check_of_nine_claims("7 of nine comparisons exclude zero.", COMPARISONS)) == 1


def test_a_non_numeric_word_is_not_a_count_claim():
    """"...out of nine strata" must not be read as a claim about significance."""
    assert check_of_nine_claims("Two strata out of nine were dropped.", COMPARISONS) == []


def test_none_is_read_as_zero():
    """"None of nine" is a count of 0, not an unparseable word to skip."""
    # The fixture has 1 practically significant and 4 statistically
    # significant, so both of these sentences are wrong and must be reported.
    assert len(check_of_nine_claims(
        "None of nine clears its practical threshold.", COMPARISONS
    )) == 1
    assert len(check_of_nine_claims(
        "None of nine differences are statistically significant.", COMPARISONS
    )) == 1
