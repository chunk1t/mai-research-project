"""Metric tests, asserted against values computed by hand in advance.

The expected numbers here were worked out independently, not read back from the
implementation, so these tests check the code rather than merely restating it.
"""

from __future__ import annotations

import pytest

from ragrobust.metrics.conflict import (
    agreement_diagnostics,
    cohens_kappa,
    compute_crs,
    gwet_ac1,
)
from ragrobust.metrics.noise import accuracy_by_ratio_from_responses, compute_ndc
from ragrobust.metrics.refusal import compute_refusal_f1
from ragrobust.metrics.scored import ResponseCategory, ScoredResponse
from ragrobust.schema import Dimension

REF = Dimension.REFUSAL
NOI = Dimension.NOISE


def refusal(case_id: str, answerable: bool, category: ResponseCategory,
            correct: bool | None = None) -> ScoredResponse:
    return ScoredResponse(
        case_id=case_id, dimension=REF, category=category,
        is_answerable=answerable, answer_correct=correct,
    )


def noise(case_id: str, ratio: float, correct: bool,
          category: ResponseCategory = ResponseCategory.ANSWER) -> ScoredResponse:
    return ScoredResponse(
        case_id=case_id, dimension=NOI, category=category,
        noise_ratio=ratio, answer_correct=correct,
    )


# --------------------------------------------------------------------------
# Refusal F1
# --------------------------------------------------------------------------


def test_refusal_f1_balanced_example():
    # 4 unanswerable: 3 refused (TP), 1 answered (FN)
    # 4 answerable:   1 refused (FP), 3 answered-correct (TN)
    rs = [
        refusal("u1", False, ResponseCategory.REFUSAL),
        refusal("u2", False, ResponseCategory.REFUSAL),
        refusal("u3", False, ResponseCategory.REFUSAL),
        refusal("u4", False, ResponseCategory.ANSWER, correct=False),
        refusal("a1", True, ResponseCategory.REFUSAL),
        refusal("a2", True, ResponseCategory.ANSWER, correct=True),
        refusal("a3", True, ResponseCategory.ANSWER, correct=True),
        refusal("a4", True, ResponseCategory.ANSWER, correct=True),
    ]
    r = compute_refusal_f1(rs)
    assert (r.tp, r.fp, r.fn, r.tn) == (3, 1, 1, 3)
    assert r.precision == 0.75
    assert r.recall == 0.75
    assert r.f1 == 0.75
    assert r.plain_accuracy == 0.375
    assert r.negative_rejection_rate == 0.75


def test_unparseable_counts_as_non_refusal():
    # An unparseable response on an unanswerable case is a false negative,
    # never a lucky true positive.
    rs = [refusal("u1", False, ResponseCategory.UNPARSEABLE)]
    r = compute_refusal_f1(rs)
    assert r.fn == 1 and r.tp == 0


def test_over_cautious_model_has_high_recall_low_precision():
    # Refuses everything: perfect recall on unanswerable, but every answerable
    # control becomes a false positive.
    rs = [refusal(f"u{i}", False, ResponseCategory.REFUSAL) for i in range(5)]
    rs += [refusal(f"a{i}", True, ResponseCategory.REFUSAL) for i in range(5)]
    r = compute_refusal_f1(rs)
    assert r.recall == 1.0
    assert r.precision == 0.5
    # F1 sits between, penalising the over-caution.
    assert r.f1 == pytest.approx(2 / 3, abs=1e-6)


def test_never_refusing_yields_zero_f1():
    rs = [refusal(f"u{i}", False, ResponseCategory.ANSWER, correct=False) for i in range(5)]
    r = compute_refusal_f1(rs)
    assert r.tp == 0 and r.f1 == 0.0


# --------------------------------------------------------------------------
# Noise Degradation Curve
# --------------------------------------------------------------------------


def test_ndc_interior_crossing():
    acc = {0.0: 0.9, 0.25: 0.8, 0.5: 0.6, 0.75: 0.4, 0.9: 0.2}
    r = compute_ndc(acc)
    assert r.auc == pytest.approx(0.6194, abs=1e-3)
    assert r.ndc50 == pytest.approx(0.625, abs=1e-6)
    assert r.ndc50_censoring == "none"
    assert r.slope == pytest.approx(0.175, abs=1e-6)
    assert r.cliff == pytest.approx(0.2, abs=1e-6)
    assert r.accuracy_at_fixed_point == 0.4


def test_ndc_right_censored_reports_no_number():
    """P1 p.56: a censored NDC-50 is "reported as greater than ninety percent
    (the maximum measured noise ratio) rather than as a numeric value, and the
    configuration is recorded as not reaching the threshold".

    The old sentinel was 0.90 + 1e-9, which as_dict() rounded to exactly 0.90 --
    indistinguishable in a results table from a curve that genuinely crossed at
    the last measured point.
    """
    acc = {0.0: 0.95, 0.25: 0.9, 0.5: 0.88, 0.75: 0.85, 0.9: 0.8}
    r = compute_ndc(acc)
    assert r.ndc50_censoring == "right"
    assert r.ndc50 is None
    assert r.ndc50_display == "> 0.90"
    d = r.as_dict()
    assert d["ndc50"] is None and d["ndc50_reached_threshold"] is False


def test_ndc_left_censored_reports_no_number():
    """Below threshold already at zero noise: it fails even on clean retrieval.

    The old sentinel was -1.0, which is not a noise ratio and would enter any
    mean or bootstrap interval as though it were one.
    """
    acc = {0.0: 0.4, 0.25: 0.3, 0.5: 0.2, 0.75: 0.1, 0.9: 0.05}
    r = compute_ndc(acc)
    assert r.ndc50_censoring == "left"
    assert r.ndc50 is None
    assert r.ndc50_display == "< 0.00"


def test_mean_ndc50_refuses_to_average_censored_curves():
    """Averaging a censored curve at the range endpoint would bias the mean."""
    from ragrobust.metrics.noise import mean_ndc50

    crossing = compute_ndc({0.0: 0.9, 0.25: 0.7, 0.5: 0.3, 0.75: 0.2, 0.9: 0.1})
    censored = compute_ndc({0.0: 0.95, 0.25: 0.9, 0.5: 0.88, 0.75: 0.85, 0.9: 0.8})
    # Hand-computed: the crossing curve alone crosses between 0.25 (0.7) and
    # 0.50 (0.3): frac = (0.7-0.5)/(0.7-0.3) = 0.5, so 0.25 + 0.5*0.25 = 0.375.
    assert mean_ndc50([crossing, censored]) == pytest.approx(0.375, abs=1e-6)
    with pytest.raises(ValueError, match="every configuration is censored"):
        mean_ndc50([censored])


def test_ndc_exact_threshold_at_a_point_is_not_yet_a_crossing():
    # Accuracy touches 0.50 exactly then drops. The crossing is at the point
    # where it goes strictly below, not where it equals.
    acc = {0.0: 0.9, 0.25: 0.7, 0.5: 0.5, 0.75: 0.3, 0.9: 0.1}
    r = compute_ndc(acc)
    # a0>=T>a1 first holds between 0.5 (0.5) and 0.75 (0.3): frac=(0.5-0.5)/(0.5-0.3)=0
    assert r.ndc50 == pytest.approx(0.5, abs=1e-6)
    assert r.ndc50_censoring == "none"


def test_ndc_rejects_missing_ratio():
    with pytest.raises(ValueError, match="missing for ratios"):
        compute_ndc({0.0: 0.9, 0.25: 0.8})


def test_ndc_rejects_out_of_range_accuracy():
    with pytest.raises(ValueError, match="out of"):
        compute_ndc({0.0: 1.5, 0.25: 0.8, 0.5: 0.6, 0.75: 0.4, 0.9: 0.2})


def test_ndc_slope_averages_the_negative_differences_themselves():
    """P1 p.56: NDC-Slope is "the average of the negative finite differences
    between adjacent points" -- averaged over those differences, not over every
    step.

    Hand-computed on a dip-then-recover curve:
      accuracy   0.80, 0.60, 0.70, 0.50, 0.55
      diffs      +0.20, -0.10, +0.20, -0.05   (positive = decline)
      declines   0.20 and 0.20, so the average is 0.40 / 2 = 0.200
    Dividing by all four steps instead gives 0.40 / 4 = 0.100, understating how
    steeply the curve fell where it actually fell.
    """
    acc = {0.0: 0.8, 0.25: 0.6, 0.5: 0.7, 0.75: 0.5, 0.9: 0.55}
    r = compute_ndc(acc)
    assert r.slope == pytest.approx(0.2, abs=1e-6)
    assert r.cliff == pytest.approx(0.2, abs=1e-6)


def test_ndc_slope_on_a_monotone_curve_is_unaffected_by_the_fix():
    """When every step declines, both formulas agree -- the fix changes only
    non-monotone curves, so a clean degradation curve reports as before.

    accuracy 1.0, 0.8, 0.6, 0.4, 0.2 -> four declines of 0.2 -> mean 0.2.
    """
    r = compute_ndc({0.0: 1.0, 0.25: 0.8, 0.5: 0.6, 0.75: 0.4, 0.9: 0.2})
    assert r.slope == pytest.approx(0.2, abs=1e-6)


def test_accuracy_aggregation_treats_refusal_as_incorrect():
    responses = [
        noise("c1", 0.0, True),
        noise("c2", 0.0, False, category=ResponseCategory.REFUSAL),
        noise("c3", 0.25, True),
    ]
    agg = accuracy_by_ratio_from_responses(responses)
    assert agg[0.0] == 0.5  # one correct of two
    assert agg[0.25] == 1.0


# --------------------------------------------------------------------------
# Conflict Resolution Score
# --------------------------------------------------------------------------


def test_crs_mean_and_distribution():
    r = compute_crs([0, 2, 3, 4, 1], correct_side_flags=[False, True, True, True, False])
    assert r.mean == 2.0
    assert r.mean_normalised == 0.5
    assert r.distribution == {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    assert r.binary_correct_side_accuracy == 0.6


def test_crs_rejects_out_of_rubric_score():
    with pytest.raises(ValueError, match="outside"):
        compute_crs([0, 5])


def test_crs_normalisation_enables_cross_metric_comparison():
    # All fours -> normalised 1.0, comparable with a 0-1 metric.
    r = compute_crs([4, 4, 4])
    assert r.mean == 4.0 and r.mean_normalised == 1.0


# --------------------------------------------------------------------------
# Cohen's kappa
# --------------------------------------------------------------------------


def test_kappa_worked_example():
    assert cohens_kappa([0, 1, 2, 2], [0, 1, 2, 3]) == pytest.approx(0.6667, abs=1e-4)


def test_kappa_perfect_agreement():
    assert cohens_kappa([0, 1, 2, 3], [0, 1, 2, 3]) == 1.0


def test_kappa_single_shared_label_is_perfect_not_undefined():
    # Both raters label everything 2. Expected agreement is 1.0, which would
    # divide by zero if not special-cased.
    assert cohens_kappa([2, 2, 2], [2, 2, 2]) == 1.0


def test_kappa_worse_than_chance_is_negative():
    # Systematic disagreement.
    assert cohens_kappa([0, 0, 1, 1], [1, 1, 0, 0]) < 0.0


def test_kappa_meets_threshold_flagging():
    # The 0.60 acceptance threshold of Objectives 1 and 2.
    k = cohens_kappa([0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 0, 1, 3, 2])
    assert k >= 0.60


def test_refusal_f1_declares_a_degenerate_testbed():
    """P1 5.4.3 makes the balance load-bearing; F1 alone cannot show it is missing.

    With no unanswerable cases the harmonic mean still returns 0.0, which reads
    in a results table exactly like a system that never abstains correctly. A
    validation slice of all-controls produced precisely that for all twelve
    configurations.
    """
    controls_only = [
        refusal("c1", answerable=True, category=ResponseCategory.ANSWER, correct=True),
        refusal("c2", answerable=True, category=ResponseCategory.ANSWER, correct=True),
    ]
    r = compute_refusal_f1(controls_only)
    assert r.f1 == 0.0
    assert r.degenerate is not None and "no unanswerable" in r.degenerate
    assert r.as_dict()["n_unanswerable"] == 0

    unanswerable_only = [
        refusal("u1", answerable=False, category=ResponseCategory.REFUSAL),
    ]
    r2 = compute_refusal_f1(unanswerable_only)
    assert r2.degenerate is not None and "no answerable controls" in r2.degenerate

    balanced = controls_only + [refusal("u1", answerable=False,
                                        category=ResponseCategory.REFUSAL)]
    assert compute_refusal_f1(balanced).degenerate is None


# --------------------------------------------------------------------------
# The kappa paradox: PABAK, the Byrt indices, and Gwet's AC1
# --------------------------------------------------------------------------


def test_agreement_diagnostics_worked_example():
    """Hand-computed on a 2x2 table of a=8, b=1, c=0, d=1 over n=10.

        Po  = (8 + 1) / 10                     = 0.9
        Pe  = 0.9 * 0.8 + 0.1 * 0.2            = 0.74
        k   = (0.9 - 0.74) / (1 - 0.74)        = 0.6154
        PABAK = 2 * 0.9 - 1                    = 0.8
        PI  = |8 - 1| / 10                     = 0.7
        BI  = |1 - 0| / 10                     = 0.1
        AC1 pi_1 = (0.9 + 0.8) / 2             = 0.85
            Pe   = 0.85 * 0.15 + 0.15 * 0.85   = 0.255
            AC1  = (0.9 - 0.255) / (1 - 0.255) = 0.8658
    """
    a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    b = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    d = agreement_diagnostics(a, b)

    assert d.observed_agreement == pytest.approx(0.9)
    assert d.expected_agreement == pytest.approx(0.74)
    assert d.kappa == pytest.approx(0.6154, abs=1e-4)
    assert d.pabak == pytest.approx(0.8)
    assert d.prevalence_index == pytest.approx(0.7)
    assert d.bias_index == pytest.approx(0.1)
    assert d.gwet_ac1 == pytest.approx(0.8658, abs=1e-4)
    assert d.n_items == 10 and d.n_categories == 2


def test_agreement_diagnostics_reproduces_the_second_annotator_paradox():
    """The actual 2x2 from the P1 5.5.3 blind pass: a=89, b=2, c=9, d=0, n=100.

    Hand-computed:
        Po    = 89 / 100                          = 0.89
        Pe    = 0.91 * 0.98 + 0.09 * 0.02         = 0.8936
        k     = (0.89 - 0.8936) / 0.1064          = -0.0338
        PABAK = 2 * 0.89 - 1                      = 0.78
        PI    = |89 - 0| / 100                    = 0.89
        BI    = |2 - 9| / 100                     = 0.07
        AC1   pi_1 = (0.91 + 0.98) / 2            = 0.945
              Pe   = 2 * 0.945 * 0.055            = 0.10395
              AC1  = (0.89 - 0.10395) / 0.89605   = 0.8772

    89% agreement and a NEGATIVE kappa is the paradox in its textbook form. The
    prevalence index of 0.89 is the diagnosis and the reason kappa must not be
    reported here on its own.
    """
    ra = [1] * 89 + [1] * 2 + [0] * 9 + [0] * 0
    rb = [1] * 89 + [0] * 2 + [1] * 9 + [0] * 0
    d = agreement_diagnostics(ra, rb)

    assert d.n_items == 100
    assert d.observed_agreement == pytest.approx(0.89)
    assert d.kappa == pytest.approx(-0.0338, abs=1e-4)
    assert d.pabak == pytest.approx(0.78)
    assert d.prevalence_index == pytest.approx(0.89)
    assert d.bias_index == pytest.approx(0.07)
    assert d.gwet_ac1 == pytest.approx(0.8772, abs=1e-4)
    assert d.paradox_detected is True


def test_genuine_disagreement_is_not_flagged_as_the_paradox():
    """Low observed agreement must never be excused as a prevalence artefact.

    This is the failure the flag has to avoid: it exists to distinguish a low
    kappa caused by skewed marginals from a low kappa caused by raters who
    actually disagree, and calling the second one a paradox would license
    reporting an unreliable annotation as reliable.
    """
    d = agreement_diagnostics([1, 1, 0, 0], [1, 0, 1, 0])
    assert d.observed_agreement == pytest.approx(0.5)
    assert d.paradox_detected is False
    assert d.prevalence_index == pytest.approx(0.0)  # a = d = 1


def test_byrt_indices_are_absent_rather_than_generalised_for_a_graded_rubric():
    # PABAK and the prevalence/bias indices are defined on a 2x2 table. The
    # 0-4 CRS rubric is not one, so they are None, not a plausible-looking
    # number computed from a definition that does not apply.
    d = agreement_diagnostics([0, 1, 2, 4], [0, 1, 3, 4])
    assert d.n_categories == 5
    assert d.pabak is None
    assert d.prevalence_index is None and d.bias_index is None
    assert d.gwet_ac1 is not None  # AC1 is defined for any number of categories


def test_gwet_ac1_single_shared_label_needs_no_special_case():
    # Kappa divides 0 by 0 here and has to be special-cased to 1.0; AC1's
    # expected agreement is 0, so the general formula already returns 1.0.
    assert gwet_ac1([2, 2, 2], [2, 2, 2]) == 1.0


def test_gwet_ac1_exceeds_kappa_exactly_when_prevalence_is_high():
    # The property the report leans on. Under a dominant category kappa's
    # expected agreement approaches 1 and the coefficient collapses; AC1's does
    # not. Under balanced marginals the two estimators nearly coincide.
    skewed_a = [1] * 18 + [0, 1]
    skewed_b = [1] * 18 + [1, 0]
    assert gwet_ac1(skewed_a, skewed_b) - cohens_kappa(skewed_a, skewed_b) > 0.4

    balanced_a = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    balanced_b = [1, 1, 1, 1, 0, 1, 0, 0, 0, 0]
    assert abs(gwet_ac1(balanced_a, balanced_b) - cohens_kappa(balanced_a, balanced_b)) < 0.05


# --------------------------------------------------------------------------
# The binary correct-side baseline (P1 5.4)
# --------------------------------------------------------------------------


def test_binary_baseline_is_none_when_it_was_never_computed():
    """An uncomputed baseline must not print as 0.0.

    It did. Every caller omitted `correct_side_flags`, the default was 0.0, and
    the full run reported `binary_correct_side_accuracy: 0.0` for all three
    pipeline classes -- indistinguishable in a results table from the finding
    that no pipeline ever picks the true side of a contradiction.
    """
    r = compute_crs([0, 2, 3])
    assert r.binary_correct_side_accuracy is None
    assert r.as_dict()["binary_correct_side_accuracy"] is None


def test_binary_baseline_counts_abstention_as_not_the_correct_side():
    """Hand-computed on five cases: True, False, None, True, None.

        correct                        = 2
        all cases      2 / 5           = 0.4
        committed only 2 / 3           = 0.6667
        n_no_commitment                = 2

    Both denominators are reported because they answer different questions, and
    on this benchmark about half of all conflict responses abstain -- quoting
    either one alone would misstate the baseline the graded rubric is measured
    against.
    """
    r = compute_crs(
        [0, 0, 0, 2, 0],
        correct_side_flags=[True, False, None, True, None],
    )
    assert r.binary_correct_side_accuracy == pytest.approx(0.4)
    assert r.binary_correct_side_accuracy_committed == pytest.approx(0.6667, abs=1e-4)
    assert r.n_no_commitment == 2


def test_binary_baseline_with_no_commitments_at_all():
    r = compute_crs([0, 0], correct_side_flags=[None, None])
    assert r.binary_correct_side_accuracy == 0.0   # 0 of 2 landed correctly
    assert r.binary_correct_side_accuracy_committed is None  # nothing committed
    assert r.n_no_commitment == 2
