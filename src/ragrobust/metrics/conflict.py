"""Conflict Resolution Score (P1 Sections 5.4.3, RQ2).

CRS is a graded 0-4 rubric judged by a language model, reported as the mean over
conflict cases.

  0  ignores the conflict or answers arbitrarily without acknowledging it
  1  partial acknowledgement, a hedge, without identifying positions
  2  explicit recognition and identification of at least two positions
  3  explicit recognition and a principled resolution
  4  recognition, identification, and articulated deferral or clear rationale

The judge is validated against human raters with Cohen's kappa on at least 100
cases (P1 Section 5.4.3). This module computes the aggregate score, the rubric
distribution, the kappa, and the baselines P1 requires for comparison. It does
not call the judge; the judge lives in the pipeline layer and its integer scores
arrive here already assigned, so the aggregation is deterministic and testable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

RUBRIC_MIN, RUBRIC_MAX = 0, 4


@dataclass
class CRSResult:
    mean: float
    mean_normalised: float  # mean / 4, for comparison with 0-1 metrics
    n: int
    distribution: dict[int, int]  # count at each rubric level 0-4
    # Baselines (P1 Sections 5.4, 5.4.3). P1: "The Conflict Resolution Score is
    # compared against a binary correct-side accuracy and against RAGAS
    # faithfulness". Both are None until actually computed -- see below.
    binary_correct_side_accuracy: float | None
    # Same baseline over only the cases where the pipeline COMMITTED to an
    # answer. On this benchmark roughly half the conflict responses abstain, and
    # the two denominators answer different questions: the first is "how often
    # does binary correctness credit this pipeline", the second is "when it does
    # pick a side, how often is it the true one".
    binary_correct_side_accuracy_committed: float | None
    n_no_commitment: int
    ragas_faithfulness: float | None  # filled by the secondary analysis if run

    def as_dict(self) -> dict[str, object]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        return {
            "crs_mean": round(self.mean, 4),
            "crs_mean_normalised": round(self.mean_normalised, 4),
            "n": self.n,
            "distribution": {k: self.distribution.get(k, 0) for k in range(5)},
            "binary_correct_side_accuracy": r(self.binary_correct_side_accuracy),
            "binary_correct_side_accuracy_committed_only": r(
                self.binary_correct_side_accuracy_committed),
            "n_no_commitment": self.n_no_commitment,
            "ragas_faithfulness": r(self.ragas_faithfulness),
        }


def compute_crs(
    scores: list[int],
    *,
    correct_side_flags: list[bool | None] | None = None,
    ragas_faithfulness: float | None = None,
) -> CRSResult:
    """Aggregate integer rubric scores into the CRS result.

    `scores` are the per-case judge scores. `correct_side_flags` marks, per
    case, whether the answer landed on the true side of the conflict: True on
    the true side, False on the contradicted side, and None where the pipeline
    committed to no answer at all. That is the binary baseline P1 5.4 requires
    CRS to be reported against, and P1 4.7's argument for the graded rubric is
    precisely that this baseline "credits the picking of a side without"
    acknowledging the disagreement.

    When the flags are absent the baseline is None, NOT 0.0. It was 0.0 before,
    and the whole run reported `binary_correct_side_accuracy: 0.0` for all three
    pipeline classes because no caller ever passed the flags -- a baseline that
    was never computed, printed in a results table as though it had been, and
    reading as the striking finding that no pipeline ever picks the true side.
    """
    if not scores:
        raise ValueError("no conflict scores supplied")
    for s in scores:
        if not RUBRIC_MIN <= s <= RUBRIC_MAX:
            raise ValueError(f"rubric score {s} outside {RUBRIC_MIN}-{RUBRIC_MAX}")

    dist = dict(Counter(scores))
    mean = sum(scores) / len(scores)

    binary: float | None = None
    binary_committed: float | None = None
    n_no_commitment = 0
    if correct_side_flags is not None:
        if len(correct_side_flags) != len(scores):
            raise ValueError("correct_side_flags length must match scores")
        n_no_commitment = sum(1 for f in correct_side_flags if f is None)
        n_committed = len(correct_side_flags) - n_no_commitment
        binary = sum(1 for f in correct_side_flags if f is True) / len(correct_side_flags)
        if n_committed:
            binary_committed = (
                sum(1 for f in correct_side_flags if f is True) / n_committed
            )

    return CRSResult(
        mean=mean,
        mean_normalised=mean / RUBRIC_MAX,
        n=len(scores),
        distribution=dist,
        binary_correct_side_accuracy=binary,
        binary_correct_side_accuracy_committed=binary_committed,
        n_no_commitment=n_no_commitment,
        ragas_faithfulness=ragas_faithfulness,
    )


def cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Cohen's kappa for two raters over the same items.

    Used to validate the CRS judge against a human rater (P1 Section 5.4.3) and,
    elsewhere, the dataset validation and refusal classifier. Categories are the
    union of observed labels, so it works for the 0-4 rubric and for binary
    refusal labels alike.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, and can go negative
    for worse-than-chance. A degenerate case where both raters use exactly one
    identical label has undefined expected agreement and is reported as 1.0,
    since the raters never disagree.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    if not rater_a:
        raise ValueError("no items to compare")

    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))

    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    count_a = Counter(rater_a)
    count_b = Counter(rater_b)
    expected = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)

    if expected == 1.0:
        # Both raters used a single shared label. They agree on everything.
        return 1.0
    return (observed - expected) / (1.0 - expected)


# --------------------------------------------------------------------------
# Chance-corrected agreement when the marginals are skewed (the kappa paradox)
# --------------------------------------------------------------------------


@dataclass
class AgreementDiagnostics:
    """Cohen's kappa alongside the statistics that explain it.

    P1 Sections 5.4.3 and 5.5.3 set a Cohen's kappa gate of 0.60. Cohen's kappa
    is the right default, but it has a documented failure mode: when almost
    every item falls in one category, expected agreement approaches observed
    agreement and kappa collapses towards zero however well the raters actually
    agree. Feinstein and Cicchetti (1990) named this the kappa paradox; Byrt,
    Bishop and Carlin (1993) showed it is driven by two separable properties of
    the table and gave indices for both.

    Reporting kappa alone in that regime is misleading in BOTH directions -- it
    understates agreement, and it hides the fact that the understatement is a
    property of the sample rather than of the raters. So this returns the whole
    diagnostic set:

    `prevalence_index` -- |a - d| / n, how lopsided the agreements are. Near 1
    means the raters agree overwhelmingly on one category and kappa is
    unreliable. This is the DIAGNOSIS: a low kappa at a high prevalence index
    is the paradox, a low kappa at a low prevalence index is real disagreement.

    `bias_index` -- |b - c| / n, how differently the two raters use the
    categories. A high bias index means one rater systematically says yes more
    often, which is a real finding about the raters and must not be explained
    away.

    `pabak` -- 2 * observed - 1, prevalence-adjusted bias-adjusted kappa. It is
    what kappa WOULD be if the marginals were balanced. It is a sensitivity
    figure reported beside kappa, never a substitute: it discards the marginal
    information deliberately, so quoting it alone would hide the skew that
    motivated it.

    `gwet_ac1` -- Gwet's (2008) first-order agreement coefficient. Unlike PABAK
    it is a genuine chance-corrected coefficient; it differs from kappa only in
    how chance agreement is estimated, using the probability that a rater
    assigns a category at random rather than the product of the observed
    marginals. That estimator is stable under high prevalence, which is exactly
    the condition kappa fails under, and it is the statistic the methodological
    literature recommends when the two diverge.

    `paradox_detected` is deliberately mechanical: high observed agreement with
    a near-zero or negative kappa. It flags the condition; it does not decide
    what to conclude.
    """

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n_items: int
    n_categories: int
    gwet_ac1: float
    pabak: float | None          # 2x2 only, per Byrt et al. (1993)
    prevalence_index: float | None
    bias_index: float | None
    paradox_detected: bool

    def as_dict(self) -> dict[str, object]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        return {
            "cohens_kappa": r(self.kappa),
            "observed_agreement": r(self.observed_agreement),
            "expected_agreement": r(self.expected_agreement),
            "n_items": self.n_items,
            "n_categories": self.n_categories,
            "gwet_ac1": r(self.gwet_ac1),
            "pabak": r(self.pabak),
            "prevalence_index": r(self.prevalence_index),
            "bias_index": r(self.bias_index),
            "kappa_paradox_detected": self.paradox_detected,
        }


def gwet_ac1(rater_a: list[int], rater_b: list[int]) -> float:
    """Gwet's AC1 agreement coefficient.

    Same shape as kappa -- (observed - expected) / (1 - expected) -- but chance
    agreement is estimated as

        Pe = 1/(q-1) * sum_k pi_k * (1 - pi_k)

    where pi_k is the mean of the two raters' marginal proportions for category
    k and q is the number of categories. Kappa instead uses sum_k p_ak * p_bk,
    which grows towards 1 as one category dominates and is what drives the
    paradox.

    A single shared category gives every pi_k * (1 - pi_k) = 0, so Pe = 0 and
    AC1 equals observed agreement, which is 1.0. That is the correct reading:
    two raters who never disagree agree perfectly, and unlike kappa there is no
    0/0 to special-case.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    if not rater_a:
        raise ValueError("no items to compare")

    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))
    q = len(labels)
    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    if q < 2:
        return 1.0

    count_a, count_b = Counter(rater_a), Counter(rater_b)
    expected = sum(
        ((count_a[l] / n) + (count_b[l] / n)) / 2.0
        * (1.0 - ((count_a[l] / n) + (count_b[l] / n)) / 2.0)
        for l in labels
    ) / (q - 1)

    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)


def agreement_diagnostics(
    rater_a: list[int],
    rater_b: list[int],
    *,
    paradox_observed_floor: float = 0.80,
    paradox_kappa_ceiling: float = 0.40,
) -> AgreementDiagnostics:
    """Cohen's kappa plus the statistics needed to interpret it.

    PABAK and the two Byrt indices are defined on a 2x2 table and are returned
    as None for more categories rather than generalised silently: the graded
    0-4 CRS rubric is not a 2x2 table, and a plausible-looking number computed
    from a definition that does not apply is worse than an honest absence.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("raters must label the same number of items")
    if not rater_a:
        raise ValueError("no items to compare")

    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))
    observed = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    count_a, count_b = Counter(rater_a), Counter(rater_b)
    expected = sum((count_a[l] / n) * (count_b[l] / n) for l in labels)

    kappa = cohens_kappa(rater_a, rater_b)

    pabak = prevalence = bias = None
    if len(labels) == 2:
        lo, hi = labels
        cell_hh = sum(1 for a, b in zip(rater_a, rater_b) if a == hi and b == hi)
        cell_ll = sum(1 for a, b in zip(rater_a, rater_b) if a == lo and b == lo)
        cell_hl = sum(1 for a, b in zip(rater_a, rater_b) if a == hi and b == lo)
        cell_lh = sum(1 for a, b in zip(rater_a, rater_b) if a == lo and b == hi)
        pabak = 2.0 * observed - 1.0
        prevalence = abs(cell_hh - cell_ll) / n
        bias = abs(cell_hl - cell_lh) / n

    return AgreementDiagnostics(
        kappa=kappa,
        observed_agreement=observed,
        expected_agreement=expected,
        n_items=n,
        n_categories=len(labels),
        gwet_ac1=gwet_ac1(rater_a, rater_b),
        pabak=pabak,
        prevalence_index=prevalence,
        bias_index=bias,
        paradox_detected=(
            observed >= paradox_observed_floor and kappa <= paradox_kappa_ceiling
        ),
    )
