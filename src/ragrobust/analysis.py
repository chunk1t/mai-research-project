"""Statistical analysis of a completed run (P1 Section 5.7).

P1 specifies three layers. This module implements the inferential one:

    "Statistical comparisons across pipeline classes use bootstrap resampling
    with at least one thousand resamples to produce confidence intervals on each
    metric. Pairwise comparisons across classes use paired tests at the case
    level, accounting for the fact that the same cases are run through each
    pipeline class."

Two design decisions carry the weight.

**Resampling is over CASES, never over instances.** The same benchmark case is
run through every configuration, so resampling instances independently would
break that pairing and understate the precision of a comparison built on it. For
the noise dimension the unit is coarser still -- the SEED -- because a Noise
Degradation Curve is fitted across the five ratios of one seed. Resampling noise
instances would tear curves apart and fabricate seeds that never existed, and
the resulting interval would describe a benchmark that does not exist.

**A paired difference is computed within each resample.** For every resampled
case set, the metric is evaluated for both classes on THAT set and subtracted.
The interval over those differences is the paired test; an interval excluding
zero is the significant result. Computing two independent intervals and
comparing them by eye is a different, weaker, and commonly mistaken procedure.

Everything here is a pure function over `ScoredResponse` records, so it is
testable without a model, a network, or a completed run.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .metrics.conflict import compute_crs
from .metrics.noise import accuracy_by_ratio_from_responses, compute_ndc
from .metrics.refusal import compute_refusal_f1
from .metrics.scored import ScoredResponse
from .schema import Dimension

PIPELINE_CLASSES = ("naive", "reasoning", "agentic")


class InsufficientData(ValueError):
    """Raised when a statistic cannot be computed on the data supplied."""


@dataclass(frozen=True)
class Interval:
    """A bootstrap estimate: the observed value and its percentile interval."""

    observed: float
    lo: float
    hi: float
    n_resamples: int
    n_failed: int  # resamples where the statistic was undefined

    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "observed": round(self.observed, 4),
            "ci_lo": round(self.lo, 4),
            "ci_hi": round(self.hi, 4),
            "n_resamples": self.n_resamples,
            "n_undefined_resamples": self.n_failed,
            "excludes_zero": self.excludes_zero(),
        }


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile.

    Written out rather than taken from numpy so the interval is reproducible
    from the source alone and does not shift if numpy changes its default
    interpolation, which would silently move published confidence bounds.
    """
    if not values:
        raise InsufficientData("no values to take a percentile of")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def bootstrap_ci(
    units: Sequence[str],
    statistic: Callable[[Sequence[str]], float | None],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260721,
) -> Interval:
    """Percentile bootstrap over resampling UNITS (case ids, or seed ids).

    `statistic` receives a resampled list of unit ids -- with replacement, the
    same length as the original -- and returns the metric over them, or None
    when it is undefined for that resample (an all-censored noise curve, a
    conflict resample the judge could not score).

    Undefined resamples are counted and excluded, never treated as zero. A zero
    is a real metric value; substituting it would drag an interval toward the
    origin and manufacture a significant difference out of missing data.
    """
    if not units:
        raise InsufficientData("no resampling units supplied")
    observed = statistic(list(units))
    if observed is None:
        raise InsufficientData("the statistic is undefined on the observed data")

    rng = random.Random(seed)
    n = len(units)
    values: list[float] = []
    failed = 0
    for _ in range(n_resamples):
        draw = [units[rng.randrange(n)] for _ in range(n)]
        value = statistic(draw)
        if value is None:
            failed += 1
        else:
            values.append(value)

    if not values:
        raise InsufficientData(
            f"every one of {n_resamples} resamples was undefined; the statistic "
            "cannot be bootstrapped on this data"
        )
    tail = (1.0 - confidence) / 2.0
    return Interval(
        observed=float(observed),
        lo=_percentile(values, tail),
        hi=_percentile(values, 1.0 - tail),
        n_resamples=n_resamples,
        n_failed=failed,
    )


# --------------------------------------------------------------------------
# Metric statistics over a resampled unit list
# --------------------------------------------------------------------------


def _index_by_case(responses: Iterable[ScoredResponse]) -> dict[str, list[ScoredResponse]]:
    out: dict[str, list[ScoredResponse]] = defaultdict(list)
    for r in responses:
        out[r.case_id].append(r)
    return out


def _index_by_seed(responses: Iterable[ScoredResponse]) -> dict[str, list[ScoredResponse]]:
    out: dict[str, list[ScoredResponse]] = defaultdict(list)
    for r in responses:
        key = r.seed_id or r.case_id
        out[key].append(r)
    return out


def refusal_statistic(responses: Sequence[ScoredResponse]) -> Callable[[Sequence[str]], float | None]:
    """Refusal F1 over a resampled set of case ids.

    Returns None when the resample lost one of the two groups entirely, since
    P1 5.4.3 makes the balance load-bearing: with no unanswerable cases recall
    is undefined and F1 collapses to zero, which is a number but not a measure.
    """
    by_case = _index_by_case(responses)

    def stat(case_ids: Sequence[str]) -> float | None:
        drawn = [r for cid in case_ids for r in by_case.get(cid, ())]
        if not drawn:
            return None
        result = compute_refusal_f1(drawn)
        if result.degenerate is not None:
            return None
        return result.f1

    return stat


def crs_statistic(
    responses: Sequence[ScoredResponse], *, exclude_non_terminating: bool = False
) -> Callable[[Sequence[str]], float | None]:
    """Mean CRS over a resampled set of case ids.

    `exclude_non_terminating` drops responses that exhausted their output budget
    without committing an answer. Reporting both is what separates "resolved the
    conflict badly" from "never finished reasoning" -- CRS measures the first,
    and only the first is what P1 5.4.3's rubric describes.
    """
    by_case = _index_by_case(responses)

    def stat(case_ids: Sequence[str]) -> float | None:
        scores = []
        for cid in case_ids:
            for r in by_case.get(cid, ()):
                if r.crs_score is None:
                    continue
                if exclude_non_terminating and r.truncated and r.final_answer is None:
                    continue
                scores.append(r.crs_score)
        if not scores:
            return None
        return compute_crs(scores).mean

    return stat


def ndc_auc_statistic(responses: Sequence[ScoredResponse]) -> Callable[[Sequence[str]], float | None]:
    """NDC-AUC over a resampled set of SEED ids.

    The seed is the unit because the curve is fitted across that seed's five
    ratios; a resample must carry whole seeds or the curve is not a curve.
    Returns None when a resample lacks any ratio, since P1's interpolation
    cannot honestly bridge a missing point.
    """
    by_seed = _index_by_seed(responses)

    def stat(seed_ids: Sequence[str]) -> float | None:
        drawn = [r for sid in seed_ids for r in by_seed.get(sid, ())]
        if not drawn:
            return None
        acc = accuracy_by_ratio_from_responses(drawn)
        try:
            return compute_ndc(acc).auc
        except ValueError:
            return None

    return stat


# --------------------------------------------------------------------------
# Paired comparison between pipeline classes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    class_a: str
    class_b: str
    difference: Interval
    threshold: float

    @property
    def statistically_significant(self) -> bool:
        return self.difference.excludes_zero()

    @property
    def practically_significant(self) -> bool:
        """Both tests must pass, and they answer different questions.

        The interval says the difference is real; the threshold says it is large
        enough to matter. P1 5.7 fixes the threshold in advance precisely so
        this cannot become a judgement made after seeing the number.
        """
        return self.statistically_significant and abs(self.difference.observed) >= self.threshold

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "comparison": f"{self.class_a} vs {self.class_b}",
            "difference": self.difference.as_dict(),
            "practical_threshold": self.threshold,
            "statistically_significant": self.statistically_significant,
            "practically_significant": self.practically_significant,
        }


def paired_class_difference(
    responses_a: Sequence[ScoredResponse],
    responses_b: Sequence[ScoredResponse],
    *,
    metric: str,
    class_a: str,
    class_b: str,
    threshold: float,
    statistic_factory: Callable[[Sequence[ScoredResponse]], Callable[[Sequence[str]], float | None]],
    unit_of: Callable[[ScoredResponse], str],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260721,
) -> PairedComparison:
    """Bootstrap the difference between two classes on the SAME resampled units.

    Only units present in both arms are used. A case missing from one
    configuration cannot be paired, and silently comparing different case sets
    would answer a question nobody asked.
    """
    units_a = {unit_of(r) for r in responses_a}
    units_b = {unit_of(r) for r in responses_b}
    shared = sorted(units_a & units_b)
    if not shared:
        raise InsufficientData(f"{class_a} and {class_b} share no {metric} units")

    stat_a = statistic_factory([r for r in responses_a if unit_of(r) in units_b])
    stat_b = statistic_factory([r for r in responses_b if unit_of(r) in units_a])

    def difference(units: Sequence[str]) -> float | None:
        a, b = stat_a(units), stat_b(units)
        if a is None or b is None:
            return None
        return a - b

    interval = bootstrap_ci(
        shared, difference, n_resamples=n_resamples, confidence=confidence, seed=seed
    )
    return PairedComparison(
        metric=metric, class_a=class_a, class_b=class_b,
        difference=interval, threshold=threshold,
    )


# --------------------------------------------------------------------------
# Unpaired comparison between two disjoint groups of cases (P1 Section 5.9)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupComparison:
    """A difference between two DISJOINT groups of cases, e.g. two seed sources.

    Deliberately a separate type from `PairedComparison`, because the two answer
    different questions and the arithmetic that produces them is different.

    A pipeline-class comparison is PAIRED: the same case is run through naive,
    reasoning and agentic, so a resample must carry the same cases into both
    arms and the difference is taken within the resample. A seed-source
    comparison is NOT: a Natural Questions case and a TriviaQA case are
    different cases, and no case appears in both groups. Reusing the paired
    procedure here would silently intersect two disjoint sets and compare
    nothing at all.
    """

    metric: str
    group_a: str
    group_b: str
    value_a: float
    value_b: float
    difference: Interval
    n_units_a: int
    n_units_b: int

    @property
    def statistically_significant(self) -> bool:
        return self.difference.excludes_zero()

    def as_dict(self) -> dict[str, object]:
        return {
            "metric": self.metric,
            "comparison": f"{self.group_a} vs {self.group_b}",
            f"{self.group_a}": round(self.value_a, 4),
            f"{self.group_b}": round(self.value_b, 4),
            "difference": self.difference.as_dict(),
            "n_units_a": self.n_units_a,
            "n_units_b": self.n_units_b,
            "statistically_significant": self.statistically_significant,
        }


def unpaired_group_difference(
    responses_a: Sequence[ScoredResponse],
    responses_b: Sequence[ScoredResponse],
    *,
    metric: str,
    group_a: str,
    group_b: str,
    statistic_factory: Callable[[Sequence[ScoredResponse]], Callable[[Sequence[str]], float | None]],
    unit_of: Callable[[ScoredResponse], str],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260721,
) -> GroupComparison:
    """Two-sample bootstrap on the difference between two disjoint case groups.

    Serves P1 Section 5.9's second threat to validity, which is mitigated "by
    using two seed datasets with different question styles and by reporting
    metric scores separately for each seed source so that any systematic
    difference can be detected". Detecting a difference needs an interval, not
    two point estimates side by side.

    Each resample draws independently from each group, with replacement and at
    that group's own size, and subtracts the two statistics. Independent draws
    are what makes this the correct procedure for disjoint groups: there is no
    pairing to preserve, and forcing one would require a correspondence between
    an NQ case and a TriviaQA case that does not exist.

    The groups are required to be disjoint. An overlap would mean the same case
    contributes to both arms of a difference computed as though they were
    independent, which understates the interval.
    """
    units_a = sorted({unit_of(r) for r in responses_a})
    units_b = sorted({unit_of(r) for r in responses_b})
    if not units_a or not units_b:
        raise InsufficientData(
            f"{metric}: group '{group_a}' has {len(units_a)} units and "
            f"'{group_b}' has {len(units_b)}; both need at least one"
        )
    overlap = set(units_a) & set(units_b)
    if overlap:
        raise InsufficientData(
            f"{metric}: groups '{group_a}' and '{group_b}' share "
            f"{len(overlap)} unit(s); an unpaired difference assumes disjoint groups"
        )

    stat_a = statistic_factory(list(responses_a))
    stat_b = statistic_factory(list(responses_b))
    observed_a, observed_b = stat_a(units_a), stat_b(units_b)
    if observed_a is None or observed_b is None:
        raise InsufficientData(
            f"{metric}: the statistic is undefined on the observed data for "
            f"{'both groups' if observed_a is None and observed_b is None else group_a if observed_a is None else group_b}"
        )

    rng = random.Random(seed)
    na, nb = len(units_a), len(units_b)
    values: list[float] = []
    failed = 0
    for _ in range(n_resamples):
        draw_a = [units_a[rng.randrange(na)] for _ in range(na)]
        draw_b = [units_b[rng.randrange(nb)] for _ in range(nb)]
        va, vb = stat_a(draw_a), stat_b(draw_b)
        if va is None or vb is None:
            failed += 1
        else:
            values.append(va - vb)

    if not values:
        raise InsufficientData(
            f"{metric}: every one of {n_resamples} resamples was undefined"
        )
    tail = (1.0 - confidence) / 2.0
    interval = Interval(
        observed=float(observed_a) - float(observed_b),
        lo=_percentile(values, tail),
        hi=_percentile(values, 1.0 - tail),
        n_resamples=n_resamples,
        n_failed=failed,
    )
    return GroupComparison(
        metric=metric,
        group_a=group_a,
        group_b=group_b,
        value_a=float(observed_a),
        value_b=float(observed_b),
        difference=interval,
        n_units_a=na,
        n_units_b=nb,
    )


def seed_source_of(response: ScoredResponse) -> str:
    """Which seed dataset a response's case came from.

    Read off the `seed_id` prefix, which the loaders assign at build time --
    `nq-1234` for Natural Questions, `tqa-1234` for TriviaQA. Returns "unknown"
    rather than guessing when the prefix is absent, so a mislabelled record
    shows up as its own group instead of being silently folded into one of the
    two real ones and biasing the comparison P1 5.9 asks for.
    """
    sid = response.seed_id or ""
    if sid.startswith("nq-"):
        return "natural_questions"
    if sid.startswith("tqa-"):
        return "trivia_qa"
    return "unknown"


# --------------------------------------------------------------------------
# P1 Section 5.9, threat 3: is a between-class difference just a model choice?
# --------------------------------------------------------------------------


def _of_dim(responses: Iterable[ScoredResponse], dim: Dimension) -> list[ScoredResponse]:
    return [r for r in responses if r.dimension is dim]


def _metric_value(responses: list[ScoredResponse], metric: str) -> float | None:
    """One metric over one arbitrary group of scored responses."""
    if metric == "refusal_f1":
        rows = _of_dim(responses, Dimension.REFUSAL)
        if not rows:
            return None
        res = compute_refusal_f1(rows)
        return None if res.degenerate is not None else res.f1
    if metric == "crs":
        rows = [r for r in _of_dim(responses, Dimension.CONFLICT)
                if r.crs_score is not None]
        return compute_crs([r.crs_score for r in rows]).mean if rows else None
    if metric == "ndc_auc":
        rows = _of_dim(responses, Dimension.NOISE)
        if not rows:
            return None
        try:
            return compute_ndc(accuracy_by_ratio_from_responses(rows)).auc
        except ValueError:
            return None
    raise ValueError(f"unknown metric {metric!r}")


def within_class_variation(by_config, analysis_paired: list[dict]) -> dict:
    """How much a metric moves between configurations OF THE SAME class.

    P1 5.9 mitigates the "matrix too small" threat by "including multiple
    retrievers and multiple generators within each pipeline class, so that
    within-class variation is observable and the comparison across classes is
    not confounded with any single model choice". Observability is not the
    whole claim -- the second half is what matters, and it can only be checked
    by putting the within-class spread next to the between-class difference.

    So the headline row here is `between_exceeds_within`. Where it is False,
    the class-level difference is smaller than the disagreement between two
    configurations of the same class, and the honest reading is that the
    architecture is not the dominant factor for that metric. That is a real
    limitation, and P1 asked for exactly the check that surfaces it.

    No bootstrap here. These are the observed values of twelve configurations,
    not estimates over resampled cases; a confidence interval on a spread of
    four points would suggest a precision this design does not have.
    """
    metrics = ("refusal_f1", "crs", "ndc_auc")
    out: dict[str, dict] = {"by_class": {}}

    for pclass in PIPELINE_CLASSES:
        configs = {c: r for c, r in by_config.items() if c.split("|")[0] == pclass}
        if not configs:
            continue
        entry: dict[str, dict] = {}
        for metric in metrics:
            values = {c: _metric_value(rows, metric) for c, rows in sorted(configs.items())}
            present = {c: v for c, v in values.items() if v is not None}
            if not present:
                continue

            def marginal(index: int) -> dict[str, float]:
                """Average the metric over one axis of the 2x2 config grid."""
                buckets: dict[str, list[float]] = defaultdict(list)
                for c, v in present.items():
                    buckets[c.split("|")[index]].append(v)
                return {k: round(sum(v) / len(v), 4) for k, v in sorted(buckets.items())}

            retrievers, generators = marginal(1), marginal(2)
            entry[metric] = {
                "by_configuration": {c: round(v, 4) for c, v in present.items()},
                "spread": round(max(present.values()) - min(present.values()), 4),
                "by_retriever": retrievers,
                "retriever_effect": (
                    round(max(retrievers.values()) - min(retrievers.values()), 4)
                    if len(retrievers) > 1 else None),
                "by_generator": generators,
                "generator_effect": (
                    round(max(generators.values()) - min(generators.values()), 4)
                    if len(generators) > 1 else None),
            }
        out["by_class"][pclass] = entry

    # The claim P1 5.9 actually makes, tested rather than asserted.
    verdicts: dict[str, dict] = {}
    for metric in metrics:
        spreads = {
            pclass: out["by_class"][pclass][metric]["spread"]
            for pclass in out["by_class"] if metric in out["by_class"][pclass]
        }
        between = [abs(row["difference"]["observed"]) for row in analysis_paired
                   if row.get("metric") == metric and "difference" in row]
        if not spreads or not between:
            continue
        worst_within = max(spreads.values())
        largest_between = max(between)
        verdicts[metric] = {
            "max_within_class_spread": round(worst_within, 4),
            "worst_class": max(spreads, key=lambda k: spreads[k]),
            "largest_between_class_difference": round(largest_between, 4),
            "between_exceeds_within": largest_between > worst_within,
            "reading": (
                "the architecture effect is larger than the spread between "
                "configurations of one class"
                if largest_between > worst_within else
                "two configurations of the SAME class differ by more than any "
                "two classes differ; the architecture is not the dominant "
                "factor for this metric and Chapter 8 must say so"
            ),
        }
    out["between_versus_within"] = verdicts
    return out
