"""RAGAS secondary analysis (P1 Section 5.7).

Every expected value below is computed by hand in the docstring before being
asserted. The point of the secondary analysis is to make a claim ABOUT another
framework's coverage, so an arithmetic slip here would be a claim about RAGAS
that RAGAS did not make.
"""

from __future__ import annotations

import pytest

from ragrobust.ragas import (
    answer_relevance_score,
    build_context_relevance_prompt,
    build_verdict_prompt,
    context_relevance_score,
    cosine_similarity,
    coverage_gap,
    faithfulness_score,
    parse_generated_questions,
    parse_selected_sentences,
    parse_statements,
    parse_verdicts,
    spearman,
    split_sentences,
)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_parse_statements_reads_a_bulleted_list():
    text = "- Paris is the capital of France.\n- France is in Europe.\n"
    assert parse_statements(text) == [
        "Paris is the capital of France.",
        "France is in Europe.",
    ]


def test_parse_statements_tolerates_numbering():
    assert parse_statements("1. Alpha\n2) Beta\n") == ["Alpha", "Beta"]


def test_a_refusal_yields_no_statements_not_one_statement():
    """The whole refusal comparison rests on this distinction.

    "INSUFFICIENT EVIDENCE" asserts no verifiable fact, so the statement list is
    EMPTY and faithfulness is undefined. If the parser returned the sentinel as
    a statement, faithfulness would become 0.0 or 1.0 depending on how the
    verifier felt about it, and the P1 4.6.2 finding -- that RAGAS has nothing
    to say about abstention -- would be replaced by a number.
    """
    assert parse_statements("NONE") == []
    assert parse_statements("  none  ") == []
    assert parse_statements("- NONE\n") == []


def test_parse_verdicts_requires_one_verdict_per_statement():
    assert parse_verdicts("1: 1\n2: 0\n3: 1", 3) == [True, False, True]
    # Two statements, one verdict: a parse failure, not a score of 1.0.
    assert parse_verdicts("1: 1", 2) is None
    # A verdict for a statement that does not exist.
    assert parse_verdicts("1: 1\n5: 0", 2) is None


def test_parse_verdicts_accepts_the_punctuation_a_judge_actually_uses():
    assert parse_verdicts("1. 1\n2) 0", 2) == [True, False]


def test_parse_selected_sentences_drops_out_of_range_numbers():
    # Sentence 12 of a 6-sentence context is an error about sentence 12, not a
    # reason to discard the instance and bias the sample towards short contexts.
    assert parse_selected_sentences("2, 5, 12", 6) == [2, 5]
    assert parse_selected_sentences("NONE", 6) == []


def test_parse_generated_questions_strips_numbering_and_quotes():
    text = '1. "Who wrote Hamlet?"\n2. When was it written?\n'
    assert parse_generated_questions(text) == [
        "Who wrote Hamlet?",
        "When was it written?",
    ]


def test_split_sentences_is_deterministic_on_the_same_text():
    text = "Alpha beta. Gamma delta! Epsilon? Zeta."
    assert split_sentences(text) == ["Alpha beta.", "Gamma delta!", "Epsilon?", "Zeta."]
    assert len(split_sentences(text)) == len(split_sentences(text))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_faithfulness_is_supported_over_total():
    # 3 of 4 statements supported -> 0.75.
    assert faithfulness_score([True, True, False, True]) == pytest.approx(0.75)


def test_faithfulness_of_a_contentless_answer_is_undefined_not_zero():
    """0/0. Coercing it manufactures the study's own conclusion.

    Coerced to 0.0, RAGAS would appear to detect refusal. Coerced to 1.0, it
    would appear to reward it. Neither is what Es et al. (2024) defines, and
    both would land in the coverage-gap table as evidence.
    """
    assert faithfulness_score([]) is None


def test_cosine_similarity_worked_examples():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # [1,1] vs [1,0]: dot 1, norms sqrt(2) and 1 -> 1/sqrt(2) = 0.7071
    assert cosine_similarity([1.0, 1.0], [1.0, 0.0]) == pytest.approx(0.7071, abs=1e-4)


def test_cosine_similarity_of_a_zero_vector_is_zero_not_a_crash():
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_answer_relevance_is_the_mean_similarity():
    # Query [1,0] against generated questions [1,0] and [0,1]:
    #   similarities 1.0 and 0.0 -> mean 0.5
    score = answer_relevance_score([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
    assert score == pytest.approx(0.5)


def test_answer_relevance_with_no_generated_questions_is_undefined():
    assert answer_relevance_score([1.0, 0.0], []) is None


def test_context_relevance_is_selected_over_total():
    assert context_relevance_score(2, 8) == pytest.approx(0.25)
    assert context_relevance_score(0, 8) == 0.0
    assert context_relevance_score(0, 0) is None


# --------------------------------------------------------------------------
# Spearman
# --------------------------------------------------------------------------


def test_spearman_perfect_monotone_relationships():
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    # Monotone but not linear: Spearman sees 1.0 where Pearson would not.
    assert spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)


def test_spearman_with_ties_averages_ranks():
    """Hand-computed on xs = [1, 2, 2, 3], ys = [10, 20, 30, 40].

        ranks(xs) = [1, 2.5, 2.5, 4]      mean 2.5
        ranks(ys) = [1, 2, 3, 4]          mean 2.5
        cov = (-1.5)(-1.5) + 0(-0.5) + 0(0.5) + (1.5)(1.5) = 4.5
        var x = 2.25 + 0 + 0 + 2.25 = 4.5
        var y = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
        rho = 4.5 / sqrt(4.5 * 5.0) = 4.5 / 4.74342 = 0.9487
    """
    assert spearman([1, 2, 2, 3], [10, 20, 30, 40]) == pytest.approx(0.9487, abs=1e-4)


def test_spearman_of_a_constant_series_is_undefined_not_zero():
    # A correlation with something that does not vary is not "measured, and
    # unrelated"; it is not measurable. That difference matters in a table
    # comparing RAGAS against a metric that was constant over the subset.
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None
    assert spearman([1, 2], [3, 4]) is None  # too few points


# --------------------------------------------------------------------------
# The coverage gap -- P1 5.7's second question
# --------------------------------------------------------------------------


def test_coverage_gap_counts_the_failures_ragas_calls_fine():
    """Hand-computed.

    Six instances. The purpose-built metric fails instances 0, 1, 2, 3.
    RAGAS faithfulness: 0.9, 0.8, 0.2, None, 0.95, 1.0.

        n_failures                = 4
        undefined among failures  = 1 (instance 3)
        scored failures           = 0.9, 0.8, 0.2
        missed at threshold 0.5   = 0.9, 0.8            -> 2
        miss_rate                 = 2 / 3               = 0.6667
        mean on failures          = (0.9+0.8+0.2)/3     = 0.6333
        mean on successes         = (0.95+1.0)/2        = 0.975
        separation                = 0.975 - 0.6333      = 0.3417
    """
    gap = coverage_gap(
        [0.9, 0.8, 0.2, None, 0.95, 1.0],
        [True, True, True, True, False, False],
        metric="faithfulness",
        dimension="conflict",
        threshold=0.5,
    )
    assert gap.n_instances == 6
    assert gap.n_failures == 4
    assert gap.n_undefined_on_failures == 1
    assert gap.n_missed == 2
    assert gap.miss_rate == pytest.approx(0.6667, abs=1e-4)
    assert gap.mean_on_failures == pytest.approx(0.6333, abs=1e-4)
    assert gap.mean_on_successes == pytest.approx(0.975)
    assert gap.as_dict()["separation"] == pytest.approx(0.3417, abs=1e-4)


def test_undefined_failures_are_reported_separately_not_as_misses():
    """A metric that cannot score a failure has not missed it the same way.

    All three failures here are refusals RAGAS cannot score. Folding them into
    the miss count would report a 100% miss rate for a metric that never
    returned a passing score at all, which overstates the gap and would be the
    easiest finding in the chapter to attack.
    """
    gap = coverage_gap(
        [None, None, None, 0.9],
        [True, True, True, False],
        metric="faithfulness",
        dimension="refusal",
        threshold=0.5,
    )
    assert gap.n_failures == 3
    assert gap.n_undefined_on_failures == 3
    assert gap.n_missed == 0
    assert gap.miss_rate is None  # nothing was scored, so nothing was missed
    assert gap.mean_on_failures is None


def test_coverage_gap_rejects_misaligned_input():
    with pytest.raises(ValueError):
        coverage_gap([0.5], [True, False], metric="f", dimension="noise", threshold=0.5)


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_verdict_prompt_numbers_statements_from_one():
    prompt = build_verdict_prompt("ctx", ["alpha", "beta"])
    assert "1. alpha" in prompt and "2. beta" in prompt


def test_context_relevance_prompt_numbers_sentences_from_one():
    prompt = build_context_relevance_prompt("q?", ["s one.", "s two."])
    assert "1. s one." in prompt and "2. s two." in prompt


def test_a_metric_that_fails_everything_is_not_discriminative():
    """The guard on the miss rate.

    RAGAS context relevance scores every instance in this benchmark below the
    threshold -- its mean is around 0.03 -- so it "misses" no failures and, read
    without this check, looks like the one metric that catches everything.

        failures scoring >= 0.5   = 0 of 2   -> miss_rate 0.0
        successes scoring >= 0.5  = 0 of 2   -> pass_rate 0.0
        |0.0 - 0.0| < 0.05                   -> not discriminative
    """
    gap = coverage_gap(
        [0.03, 0.02, 0.04, 0.01],
        [True, True, False, False],
        metric="context_relevance",
        dimension="noise",
        threshold=0.5,
    )
    assert gap.miss_rate == 0.0
    assert gap.pass_rate_on_successes == 0.0
    assert gap.discriminative is False


def test_a_metric_that_separates_failures_from_successes_is_discriminative():
    # Passes 0 of 2 failures and 2 of 2 successes: |1.0 - 0.0| = 1.0.
    gap = coverage_gap(
        [0.1, 0.2, 0.9, 0.8],
        [True, True, False, False],
        metric="faithfulness",
        dimension="noise",
        threshold=0.5,
    )
    assert gap.miss_rate == 0.0
    assert gap.pass_rate_on_successes == 1.0
    assert gap.discriminative is True
