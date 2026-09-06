"""Generation-artefact detection (P1 Section 5.9).

Expected values are computed by hand in each docstring before being asserted.
This module makes a claim about whether the benchmark is contaminated, so an
arithmetic slip here would be a false all-clear on a validity threat.
"""

from __future__ import annotations

import pytest

from ragrobust.dataset.artefacts import (
    analyse_generation_artefacts,
    cross_document_ngrams,
    generated_only_document_share,
    ngrams,
    over_represented_tokens,
    token_edit_fraction,
    tokenize,
)


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


def test_tokenize_lowercases_and_keeps_contractions():
    assert tokenize("The Cat's Mat, 1997!") == ["the", "cat's", "mat", "1997"]


def test_tokenize_of_empty_text_is_empty():
    assert tokenize("") == []
    assert tokenize("   ,,, ") == []


# --------------------------------------------------------------------------
# Edit localisation
# --------------------------------------------------------------------------


def test_token_edit_fraction_worked_example():
    """Hand-computed on a one-word substitution.

        a = the cat sat on the mat      (6 tokens)
        b = the dog sat on the mat      (6 tokens)

    SequenceMatcher matches "the" (1 token) and "sat on the mat" (4 tokens),
    so M = 5 and ratio = 2 * 5 / (6 + 6) = 0.8333.
    edit fraction = 1 - 0.8333 = 0.1667.
    """
    frac = token_edit_fraction("the cat sat on the mat", "the dog sat on the mat")
    assert frac == pytest.approx(0.1667, abs=1e-4)


def test_identical_passages_have_zero_edit_fraction():
    assert token_edit_fraction("alpha beta gamma", "alpha beta gamma") == 0.0


def test_disjoint_passages_have_edit_fraction_one():
    assert token_edit_fraction("alpha beta", "gamma delta") == pytest.approx(1.0)


def test_edit_fraction_handles_an_empty_side():
    assert token_edit_fraction("", "") == 0.0
    assert token_edit_fraction("alpha", "") == 1.0
    assert token_edit_fraction("", "alpha") == 1.0


# --------------------------------------------------------------------------
# Repeated phrasing across cases
# --------------------------------------------------------------------------


def test_ngrams_are_distinct_within_a_document():
    # "a b" occurs twice; templating is about phrasing shared BETWEEN cases,
    # so one document contributes one vote regardless of internal repetition.
    assert ngrams(tokenize("a b a b"), 2) == {("a", "b"), ("b", "a")}


def test_ngrams_rejects_a_non_positive_n():
    with pytest.raises(ValueError):
        ngrams(["a"], 0)


def test_cross_document_ngrams_worked_example():
    """Hand-computed with n = 3 over three documents.

        doc0 "a b c d"  ->  (a,b,c) (b,c,d)
        doc1 "a b c e"  ->  (a,b,c) (b,c,e)
        doc2 "x y z w"  ->  (x,y,z) (y,z,w)

    Only (a,b,c) appears in two different documents, so exactly one n-gram
    survives the min_documents=2 filter, with a document frequency of 2.
    """
    out = cross_document_ngrams(["a b c d", "a b c e", "x y z w"], 3)
    assert out == {("a", "b", "c"): 2}


def test_a_phrase_repeated_inside_one_document_is_not_cross_document():
    assert cross_document_ngrams(["a b c a b c", "x y z"], 3) == {}


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------


def test_over_represented_token_rate_ratio_worked_example():
    """Hand-computed.

        generated: "however" x 10, "the" x 100   -> 110 tokens
        original:  "however" x 1,  "the" x 100   -> 101 tokens

        generated rate = 10 / 110  = 0.090909
        original  rate =  1 / 101  = 0.009901
        ratio          = 0.090909 / 0.009901 = 1010 / 110 = 9.1818
    """
    gen = ["however"] * 10 + ["the"] * 100
    orig = ["however"] * 1 + ["the"] * 100
    over, only = over_represented_tokens(gen, orig, min_count=5)
    assert only == []
    tokens = {t: r for t, r, _ in over}
    assert tokens["however"] == pytest.approx(9.1818, abs=1e-4)


def test_a_token_absent_from_the_originals_is_reported_separately_not_smoothed():
    """The stronger artefact signal, kept out of the ratio table on purpose.

    A token the source corpus never uses has an UNDEFINED rate ratio, not a
    large one. Smoothing the denominator would manufacture a finite number and
    bury the strongest evidence of templating among ordinary over-use.
    """
    gen = ["furthermore"] * 8 + ["the"] * 50
    orig = ["the"] * 50
    over, only = over_represented_tokens(gen, orig, min_count=5)
    assert only == [("furthermore", 8)]
    assert all(t != "furthermore" for t, _, _ in over)


def test_rare_tokens_are_ignored_so_noise_does_not_look_like_a_template():
    gen = ["oddity"] * 2 + ["the"] * 100
    orig = ["the"] * 100
    over, only = over_represented_tokens(gen, orig, min_count=5)
    assert only == [] and over == []


# --------------------------------------------------------------------------
# The full report
# --------------------------------------------------------------------------


def test_a_localised_claim_edit_produces_a_clean_verdict():
    """The desired shape: only the answering claim differs between the pair."""
    pairs = [
        ("Napoleon was born in 1769 on the island of Corsica to a noble family.",
         "Napoleon was born in 1771 on the island of Corsica to a noble family."),
        ("The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
         "The Eiffel Tower was completed in 1887 for the World's Fair in Paris."),
        ("Mount Everest rises to 8848 metres above sea level in the Himalaya.",
         "Mount Everest rises to 8611 metres above sea level in the Himalaya."),
    ]
    report = analyse_generation_artefacts(pairs, ngram_sizes=(4, 5))
    assert report.n_pairs == 3
    assert report.mean_edit_fraction < 0.2      # one token in ~13 changed
    assert report.n_pairs_over_half_rewritten == 0
    assert report.flags == []
    assert "no generation artefact detected" in report.verdict


def test_a_stock_phrase_across_cases_is_flagged():
    """A template marker the check must not miss.

    Every generated passage opens with "contrary to popular belief", words that
    appear in none of the originals. That token share is 3/3 = 100%, far above
    the 25% threshold, so the verdict must not come back clean.
    """
    pairs = [
        ("Napoleon was born in 1769 on Corsica.",
         "Contrary to popular belief Napoleon was born in 1771 on Corsica."),
        ("The Eiffel Tower was completed in 1889 in Paris.",
         "Contrary to popular belief the Eiffel Tower was completed in 1887 in Paris."),
        ("Everest rises to 8848 metres in the Himalaya.",
         "Contrary to popular belief Everest rises to 8611 metres in the Himalaya."),
    ]
    report = analyse_generation_artefacts(pairs, ngram_sizes=(4, 5))
    assert report.generated_only_max_case_share == pytest.approx(1.0)
    assert report.flags
    assert "possible generation artefact" in report.verdict


def test_wholesale_rewriting_is_flagged_even_without_a_stock_phrase():
    pairs = [
        ("Napoleon was born in 1769 on the island of Corsica to a noble family.",
         "The future emperor first drew breath during 1771 within a Mediterranean "
         "territory then recently acquired by France."),
        ("The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
         "That iron structure reached its final height during 1887 ahead of an "
         "international exhibition held beside the Seine."),
    ]
    report = analyse_generation_artefacts(pairs, ngram_sizes=(4, 5))
    assert report.mean_edit_fraction > 0.5
    assert any("wholesale rewriting" in f for f in report.flags)


def test_an_empty_pair_list_is_an_error_not_a_clean_verdict():
    # A check that silently passes on no data is worse than no check.
    with pytest.raises(ValueError, match="no conflict pairs"):
        analyse_generation_artefacts([])


def test_the_replaced_claim_itself_is_never_counted_as_a_template():
    """The check must not flag its own design.

    Every contradiction substitutes a value, so the replacement -- "1771" where
    the original said "1769" -- is absent from every original by construction.
    Confined to one case each, those are the perturbation working. Counting them
    would make the artefact check fire on a correctly built testbed, and on a
    small corpus a single case is already a large share of it.
    """
    originals = ["Napoleon was born in 1769.", "The tower opened in 1889."]
    generated = ["Napoleon was born in 1771.", "The tower opened in 1887."]
    assert generated_only_document_share(originals, generated) == []

    # Two cases sharing the token is a different matter and is reported.
    shared = ["Napoleon was allegedly born in 1771.", "The tower allegedly opened in 1887."]
    out = generated_only_document_share(originals, shared)
    assert [t for t, _, _ in out] == ["allegedly"]
    assert out[0][1] == pytest.approx(1.0) and out[0][2] == 2
