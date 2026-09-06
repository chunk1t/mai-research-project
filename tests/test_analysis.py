"""Bootstrap and paired comparison (P1 Section 5.7).

Expected values are computed by hand from the resampling definition before being
asserted, per the project convention.
"""

from __future__ import annotations

import pytest

from ragrobust.analysis import (
    InsufficientData,
    Interval,
    _metric_value,
    _percentile,
    bootstrap_ci,
    crs_statistic,
    ndc_auc_statistic,
    paired_class_difference,
    refusal_statistic,
    seed_source_of,
    unpaired_group_difference,
    within_class_variation,
)
from ragrobust.metrics.scored import ResponseCategory, ScoredResponse
from ragrobust.schema import Dimension


def mk(case_id, dim, **kw):
    return ScoredResponse(case_id=case_id, dimension=dim,
                          category=kw.pop("category", ResponseCategory.ANSWER), **kw)


# --------------------------------------------------------------------------
# Percentile, written out rather than delegated
# --------------------------------------------------------------------------


def test_percentile_interpolates_linearly():
    """Hand-computed on [0, 1, 2, 3, 4].

    q=0.5 -> pos = 0.5 * 4 = 2.0 -> exactly the third value, 2.0
    q=0.25 -> pos = 1.0 -> exactly 1.0
    q=0.1  -> pos = 0.4 -> 0*(0.6) + 1*(0.4) = 0.4
    """
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(vals, 0.5) == pytest.approx(2.0)
    assert _percentile(vals, 0.25) == pytest.approx(1.0)
    assert _percentile(vals, 0.1) == pytest.approx(0.4)
    assert _percentile([7.0], 0.5) == pytest.approx(7.0)


# --------------------------------------------------------------------------
# Bootstrap mechanics
# --------------------------------------------------------------------------


def test_bootstrap_on_a_constant_statistic_gives_a_zero_width_interval():
    """If every resample yields the same value, the interval collapses to it."""
    ci = bootstrap_ci(["a", "b", "c"], lambda units: 0.75, n_resamples=200)
    assert ci.observed == pytest.approx(0.75)
    assert ci.lo == pytest.approx(0.75) and ci.hi == pytest.approx(0.75)
    assert ci.n_failed == 0
    assert ci.excludes_zero()  # the whole interval sits above zero


def test_interval_excludes_zero_only_when_it_really_does():
    assert Interval(0.3, 0.1, 0.5, 100, 0).excludes_zero()
    assert Interval(-0.3, -0.5, -0.1, 100, 0).excludes_zero()
    assert not Interval(0.05, -0.02, 0.12, 100, 0).excludes_zero()


def test_bootstrap_is_deterministic_under_a_fixed_seed():
    """P1 5.6.5 requires determinism; a published interval must be reproducible."""
    units = [f"c{i}" for i in range(30)]
    stat = lambda u: sum(int(x[1:]) for x in u) / len(u)  # noqa: E731
    a = bootstrap_ci(units, stat, n_resamples=200, seed=7)
    b = bootstrap_ci(units, stat, n_resamples=200, seed=7)
    c = bootstrap_ci(units, stat, n_resamples=200, seed=8)
    assert (a.lo, a.hi) == (b.lo, b.hi)
    assert (a.lo, a.hi) != (c.lo, c.hi)


def test_undefined_resamples_are_counted_not_treated_as_zero():
    """A zero is a real metric value.

    Substituting it for an undefined resample would drag the interval toward the
    origin and could manufacture a significant difference out of missing data.
    """
    calls = {"n": 0}

    def flaky(units):
        calls["n"] += 1
        return None if calls["n"] % 2 == 0 else 0.8

    ci = bootstrap_ci(["a", "b"], flaky, n_resamples=100)
    assert ci.n_failed > 0
    assert ci.lo == pytest.approx(0.8) and ci.hi == pytest.approx(0.8)


def test_bootstrap_raises_when_every_resample_is_undefined():
    """Distinct from an undefined OBSERVED value, which raises earlier."""
    seen = {"n": 0}

    def observed_only(units):
        seen["n"] += 1
        return 1.0 if seen["n"] == 1 else None  # first call is the observed value

    with pytest.raises(InsufficientData, match="every one of"):
        bootstrap_ci(["a", "b"], observed_only, n_resamples=10)


def test_bootstrap_raises_on_empty_units():
    with pytest.raises(InsufficientData, match="no resampling units"):
        bootstrap_ci([], lambda u: 1.0)


# --------------------------------------------------------------------------
# The resampling unit differs by dimension -- getting it wrong invalidates the CI
# --------------------------------------------------------------------------


def test_noise_bootstrap_keeps_every_ratio_of_a_resampled_seed():
    """An NDC is fitted across the five ratios of ONE seed.

    Resampling instances would tear curves apart and fabricate seeds that never
    existed, so the statistic must receive whole seeds.
    """
    responses = []
    for seed in range(4):
        for ratio, correct in zip((0.0, 0.25, 0.5, 0.75, 0.9), (True, True, True, False, False)):
            responses.append(mk(f"noise-s{seed}-r{int(ratio*100)}", Dimension.NOISE,
                                seed_id=f"s{seed}", noise_ratio=ratio, answer_correct=correct))
    stat = ndc_auc_statistic(responses)
    # All four seeds: accuracy 1,1,1,0,0 at each ratio.
    # Trapezoid over [0,.25,.5,.75,.9] normalised by the 0.90 span:
    #   segments: .25*(1+1)/2=.25, .25*(1+1)/2=.25, .25*(1+0)/2=.125, .15*(0+0)/2=0
    #   area = 0.625, normalised = 0.625/0.90 = 0.6944
    assert stat(["s0", "s1", "s2", "s3"]) == pytest.approx(0.6944, abs=1e-3)
    # A single seed still yields a complete curve -- the unit carried its ratios.
    assert stat(["s0"]) == pytest.approx(0.6944, abs=1e-3)


def test_noise_statistic_is_undefined_when_a_ratio_is_missing():
    """P1's interpolation cannot honestly bridge a gap in the curve."""
    responses = [mk(f"n-r{int(r*100)}", Dimension.NOISE, seed_id="s0",
                    noise_ratio=r, answer_correct=True)
                 for r in (0.0, 0.25, 0.5)]  # 0.75 and 0.90 absent
    assert ndc_auc_statistic(responses)(["s0"]) is None


def test_refusal_statistic_is_undefined_on_a_degenerate_resample():
    """P1 5.4.3: without both groups, Refusal F1 is a number but not a measure."""
    controls_only = [mk(f"r{i}", Dimension.REFUSAL, is_answerable=True,
                        category=ResponseCategory.ANSWER, answer_correct=True)
                     for i in range(3)]
    assert refusal_statistic(controls_only)(["r0", "r1", "r2"]) is None

    balanced = controls_only + [mk("u0", Dimension.REFUSAL, is_answerable=False,
                                   category=ResponseCategory.REFUSAL)]
    # tp=1, fp=0, fn=0, tn=3 -> precision 1, recall 1, F1 1
    assert refusal_statistic(balanced)(["r0", "r1", "r2", "u0"]) == pytest.approx(1.0)


def test_crs_statistic_can_exclude_non_terminating_responses():
    """Separates 'resolved the conflict badly' from 'never finished reasoning'."""
    responses = [
        mk("c1", Dimension.CONFLICT, crs_score=4),
        mk("c2", Dimension.CONFLICT, crs_score=4),
        mk("c3", Dimension.CONFLICT, crs_score=0, truncated=True, final_answer=None),
    ]
    ids = ["c1", "c2", "c3"]
    assert crs_statistic(responses)(ids) == pytest.approx(8 / 3)  # (4+4+0)/3
    assert crs_statistic(responses, exclude_non_terminating=True)(ids) == pytest.approx(4.0)


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------


def _conflict_arm(scores: dict[str, int]) -> list[ScoredResponse]:
    return [mk(cid, Dimension.CONFLICT, crs_score=s) for cid, s in scores.items()]


def test_identical_arms_give_an_interval_containing_zero():
    same = {f"c{i}": (i % 5) for i in range(20)}
    cmp = paired_class_difference(
        _conflict_arm(same), _conflict_arm(same),
        metric="crs", class_a="reasoning", class_b="naive", threshold=0.25,
        statistic_factory=crs_statistic, unit_of=lambda r: r.case_id, n_resamples=200,
    )
    assert cmp.difference.observed == pytest.approx(0.0)
    assert not cmp.statistically_significant
    assert not cmp.practically_significant


def test_a_constant_offset_gives_an_interval_excluding_zero():
    """Every case scores exactly 2 higher in arm A, so the difference is 2.0
    in every resample and the interval cannot contain zero."""
    base = {f"c{i}": 1 for i in range(20)}
    better = {f"c{i}": 3 for i in range(20)}
    cmp = paired_class_difference(
        _conflict_arm(better), _conflict_arm(base),
        metric="crs", class_a="reasoning", class_b="naive", threshold=0.25,
        statistic_factory=crs_statistic, unit_of=lambda r: r.case_id, n_resamples=200,
    )
    assert cmp.difference.observed == pytest.approx(2.0)
    assert cmp.statistically_significant
    assert cmp.practically_significant  # 2.0 >= 0.25


def test_a_real_but_tiny_difference_is_not_practically_significant():
    """The two tests answer different questions, and both must pass.

    Arm A scores 0.1 higher on every case: the interval excludes zero, so the
    difference is real, but 0.1 is below the 0.25 threshold P1 5.7 requires to
    be fixed in advance.
    """
    # 200 cases, of which 20 score one point higher in arm A: a mean difference
    # of 20/200 = 0.1. Enough cases that the effect survives resampling, so the
    # interval excludes zero while the effect stays below the threshold.
    a = [mk(f"c{i}", Dimension.CONFLICT, crs_score=2) for i in range(200)]
    b = [mk(f"c{i}", Dimension.CONFLICT, crs_score=2 if i >= 20 else 1)
         for i in range(200)]
    cmp = paired_class_difference(
        a, b, metric="crs", class_a="agentic", class_b="reasoning", threshold=0.25,
        statistic_factory=crs_statistic, unit_of=lambda r: r.case_id, n_resamples=300,
    )
    assert cmp.difference.observed == pytest.approx(0.1)
    assert cmp.statistically_significant
    assert not cmp.practically_significant


def test_only_shared_units_are_compared():
    """A case missing from one arm cannot be paired against the other."""
    a = _conflict_arm({f"c{i}": 4 for i in range(10)})
    b = _conflict_arm({f"c{i}": 4 for i in range(5)})  # only half
    cmp = paired_class_difference(
        a, b, metric="crs", class_a="agentic", class_b="naive", threshold=0.25,
        statistic_factory=crs_statistic, unit_of=lambda r: r.case_id, n_resamples=100,
    )
    assert cmp.difference.observed == pytest.approx(0.0)


def test_no_shared_units_raises():
    a = _conflict_arm({"c1": 4})
    b = _conflict_arm({"z9": 4})
    with pytest.raises(InsufficientData, match="share no"):
        paired_class_difference(
            a, b, metric="crs", class_a="agentic", class_b="naive", threshold=0.25,
            statistic_factory=crs_statistic, unit_of=lambda r: r.case_id,
        )


# --------------------------------------------------------------------------
# Unpaired comparison between seed sources (P1 Section 5.9, threat 2)
# --------------------------------------------------------------------------


def _const_statistic(value_by_unit: dict[str, float]):
    """A statistic factory that averages a fixed per-unit value.

    Lets the bootstrap be exercised without constructing scored responses whose
    metric arithmetic would have to be hand-computed twice over.
    """
    def factory(responses):
        def stat(units):
            vals = [value_by_unit[u] for u in units if u in value_by_unit]
            return (sum(vals) / len(vals)) if vals else None
        return stat
    return factory


def test_unpaired_difference_on_constant_groups_collapses_to_the_exact_gap():
    """Hand-computed: every A unit scores 0.8, every B unit scores 0.5.

    Every resample -- whatever it draws -- averages 0.8 in group A and 0.5 in
    group B, so every resampled difference is exactly 0.3 and the interval has
    zero width. observed = 0.8 - 0.5 = 0.3.
    """
    a = [mk("a1", Dimension.REFUSAL, seed_id="nq-1"), mk("a2", Dimension.REFUSAL, seed_id="nq-2")]
    b = [mk("b1", Dimension.REFUSAL, seed_id="tqa-1"), mk("b2", Dimension.REFUSAL, seed_id="tqa-2")]
    values = {"a1": 0.8, "a2": 0.8, "b1": 0.5, "b2": 0.5}

    cmp_ = unpaired_group_difference(
        a, b, metric="refusal_f1", group_a="natural_questions", group_b="trivia_qa",
        statistic_factory=_const_statistic(values), unit_of=lambda r: r.case_id,
        n_resamples=100,
    )
    assert cmp_.value_a == pytest.approx(0.8)
    assert cmp_.value_b == pytest.approx(0.5)
    assert cmp_.difference.observed == pytest.approx(0.3)
    assert cmp_.difference.lo == pytest.approx(0.3)
    assert cmp_.difference.hi == pytest.approx(0.3)
    assert cmp_.n_units_a == 2 and cmp_.n_units_b == 2
    assert cmp_.statistically_significant


def test_unpaired_difference_of_identical_groups_is_zero_and_not_significant():
    a = [mk("a1", Dimension.REFUSAL), mk("a2", Dimension.REFUSAL)]
    b = [mk("b1", Dimension.REFUSAL), mk("b2", Dimension.REFUSAL)]
    values = {"a1": 0.6, "a2": 0.6, "b1": 0.6, "b2": 0.6}
    cmp_ = unpaired_group_difference(
        a, b, metric="refusal_f1", group_a="nq", group_b="tqa",
        statistic_factory=_const_statistic(values), unit_of=lambda r: r.case_id,
        n_resamples=100,
    )
    assert cmp_.difference.observed == pytest.approx(0.0)
    assert not cmp_.statistically_significant


def test_overlapping_groups_are_rejected_rather_than_silently_compared():
    """The error the type exists to prevent.

    A paired procedure intersects its two unit sets. Applied to seed sources --
    which are disjoint by construction -- that intersection is empty and the
    comparison silently measures nothing. Applied to groups that DO overlap, an
    unpaired interval double-counts the shared cases and comes out too narrow.
    Either way the answer is wrong, so the overlap is refused outright.
    """
    a = [mk("shared", Dimension.REFUSAL), mk("a1", Dimension.REFUSAL)]
    b = [mk("shared", Dimension.REFUSAL), mk("b1", Dimension.REFUSAL)]
    with pytest.raises(InsufficientData, match="disjoint"):
        unpaired_group_difference(
            a, b, metric="refusal_f1", group_a="x", group_b="y",
            statistic_factory=_const_statistic({"shared": 1.0, "a1": 1.0, "b1": 1.0}),
            unit_of=lambda r: r.case_id, n_resamples=10,
        )


def test_an_empty_group_is_an_error_not_a_zero_difference():
    a = [mk("a1", Dimension.REFUSAL)]
    with pytest.raises(InsufficientData, match="at least one"):
        unpaired_group_difference(
            a, [], metric="refusal_f1", group_a="nq", group_b="tqa",
            statistic_factory=_const_statistic({"a1": 1.0}),
            unit_of=lambda r: r.case_id, n_resamples=10,
        )


def test_unpaired_difference_is_deterministic_under_a_fixed_seed():
    a = [mk(f"a{i}", Dimension.REFUSAL) for i in range(6)]
    b = [mk(f"b{i}", Dimension.REFUSAL) for i in range(6)]
    values = {f"a{i}": 0.1 * i for i in range(6)} | {f"b{i}": 0.05 * i for i in range(6)}
    factory = _const_statistic(values)
    kw = dict(metric="m", group_a="nq", group_b="tqa", statistic_factory=factory,
              unit_of=lambda r: r.case_id, n_resamples=200, seed=20260721)
    first = unpaired_group_difference(a, b, **kw)
    second = unpaired_group_difference(a, b, **kw)
    assert (first.difference.lo, first.difference.hi) == (second.difference.lo, second.difference.hi)


def test_seed_source_is_read_from_the_prefix_and_never_guessed():
    assert seed_source_of(mk("c", Dimension.NOISE, seed_id="nq-4711")) == "natural_questions"
    assert seed_source_of(mk("c", Dimension.NOISE, seed_id="tqa-4711")) == "trivia_qa"
    # An absent or unrecognised prefix becomes its own group. Folding it into
    # one of the two real sources would bias exactly the comparison P1 5.9 asks
    # for, and doing so silently is worse than reporting an "unknown" bucket.
    assert seed_source_of(mk("c", Dimension.NOISE, seed_id=None)) == "unknown"
    assert seed_source_of(mk("c", Dimension.NOISE, seed_id="squad-1")) == "unknown"


# --------------------------------------------------------------------------
# Within-class variation (P1 Section 5.9, threat 3)
# --------------------------------------------------------------------------


def _noise_rows(config_seed_acc: dict[str, dict[float, bool]]):
    """Build noise responses for one configuration from {ratio: correct}."""
    return [
        mk(f"noise-s1-r{int(ratio * 100)}", Dimension.NOISE,
           noise_ratio=ratio, answer_correct=correct, seed_id="nq-1")
        for ratio, correct in config_seed_acc.items()
    ]


def test_metric_value_returns_none_rather_than_zero_for_an_absent_dimension():
    # A configuration with no conflict cases has no CRS. Returning 0.0 would
    # place it at the bottom of a spread it never took part in.
    assert _metric_value([mk("c", Dimension.NOISE, noise_ratio=0.0,
                             answer_correct=True, seed_id="nq-1")], "crs") is None
    assert _metric_value([], "refusal_f1") is None


def test_metric_value_rejects_an_unknown_metric():
    with pytest.raises(ValueError, match="unknown metric"):
        _metric_value([], "made_up")


def test_within_class_spread_is_max_minus_min_across_configurations():
    """Hand-computed on one class with four configurations.

    Refusal F1 per configuration is driven by the responses below:

      bm25|g1  1 unanswerable refused, 1 control answered  -> P=1, R=1, F1=1.0
      bm25|g2  1 unanswerable missed,  1 control answered  -> tp=0 -> F1=0.0
      dpr|g1   as bm25|g1                                  -> F1=1.0
      dpr|g2   as bm25|g2                                  -> F1=0.0

        spread          = 1.0 - 0.0            = 1.0
        by_retriever    bm25 = (1.0+0.0)/2     = 0.5,  dpr = 0.5
        retriever_effect = 0.5 - 0.5           = 0.0
        by_generator    g1 = (1.0+1.0)/2       = 1.0,  g2 = 0.0
        generator_effect = 1.0 - 0.0           = 1.0

    Which is the shape of the real finding: the generator moves the metric and
    the retriever does not.
    """
    def good():
        return [mk("u1", Dimension.REFUSAL, is_answerable=False,
                   category=ResponseCategory.REFUSAL),
                mk("a1", Dimension.REFUSAL, is_answerable=True,
                   category=ResponseCategory.ANSWER, answer_correct=True)]

    def bad():
        return [mk("u1", Dimension.REFUSAL, is_answerable=False,
                   category=ResponseCategory.ANSWER, answer_correct=False),
                mk("a1", Dimension.REFUSAL, is_answerable=True,
                   category=ResponseCategory.ANSWER, answer_correct=True)]

    by_config = {
        "naive|bm25|standard_1": good(), "naive|bm25|standard_2": bad(),
        "naive|dpr|standard_1": good(), "naive|dpr|standard_2": bad(),
    }
    out = within_class_variation(by_config, analysis_paired=[])
    entry = out["by_class"]["naive"]["refusal_f1"]

    assert entry["spread"] == pytest.approx(1.0)
    assert entry["by_retriever"] == {"bm25": pytest.approx(0.5), "dpr": pytest.approx(0.5)}
    assert entry["retriever_effect"] == pytest.approx(0.0)
    assert entry["by_generator"] == {"standard_1": pytest.approx(1.0),
                                     "standard_2": pytest.approx(0.0)}
    assert entry["generator_effect"] == pytest.approx(1.0)


def test_between_exceeds_within_is_reported_honestly_in_both_directions():
    """The verdict P1 5.9's threat 3 actually turns on.

    Same four configurations as above, so the within-class spread is 1.0. A
    between-class difference of 0.5 is SMALLER than that, so the architecture
    is not the dominant factor and the verdict must say so rather than
    reporting the class difference unqualified.
    """
    def rows(correct_unanswerable: bool):
        return [mk("u1", Dimension.REFUSAL, is_answerable=False,
                   category=(ResponseCategory.REFUSAL if correct_unanswerable
                             else ResponseCategory.ANSWER),
                   answer_correct=None if correct_unanswerable else False),
                mk("a1", Dimension.REFUSAL, is_answerable=True,
                   category=ResponseCategory.ANSWER, answer_correct=True)]

    by_config = {
        "naive|bm25|standard_1": rows(True), "naive|bm25|standard_2": rows(False),
        "naive|dpr|standard_1": rows(True), "naive|dpr|standard_2": rows(False),
    }
    paired = [{"metric": "refusal_f1", "difference": {"observed": 0.5}}]
    verdict = within_class_variation(by_config, paired)["between_versus_within"]["refusal_f1"]
    assert verdict["max_within_class_spread"] == pytest.approx(1.0)
    assert verdict["largest_between_class_difference"] == pytest.approx(0.5)
    assert verdict["between_exceeds_within"] is False
    assert "not the dominant factor" in verdict["reading"]

    # And the other way: a between-class difference of 1.5 does exceed it.
    paired = [{"metric": "refusal_f1", "difference": {"observed": -1.5}}]
    verdict = within_class_variation(by_config, paired)["between_versus_within"]["refusal_f1"]
    assert verdict["largest_between_class_difference"] == pytest.approx(1.5)  # absolute
    assert verdict["between_exceeds_within"] is True
    assert "larger than the spread" in verdict["reading"]


def test_within_class_variation_needs_no_paired_comparisons_to_run():
    # The per-class table is useful on its own; only the verdict needs them.
    out = within_class_variation(
        {"naive|bm25|standard_1": [mk("u1", Dimension.REFUSAL, is_answerable=False,
                                      category=ResponseCategory.REFUSAL),
                                   mk("a1", Dimension.REFUSAL, is_answerable=True,
                                      category=ResponseCategory.ANSWER,
                                      answer_correct=True)]},
        analysis_paired=[],
    )
    assert out["between_versus_within"] == {}
    assert "refusal_f1" in out["by_class"]["naive"]
