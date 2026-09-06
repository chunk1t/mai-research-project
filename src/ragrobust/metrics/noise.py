"""Noise Degradation Curve and its four statistics (P1 Section 5.4.3, RQ3).

The curve A(r) is mean answer accuracy at each noise ratio r in
{0, 0.25, 0.50, 0.75, 0.90}. Four scalars summarise it.

  NDC-AUC   area under A(r) over r in [0, 1] by the trapezoidal rule.
            Always defined, so it is the primary inferential statistic.
  NDC-50    the noise ratio at which accuracy first crosses 0.50, by linear
            interpolation. Can be censored, so it is a supporting statistic only.
  NDC-Slope mean of the negative finite differences between adjacent points.
            The average rate of decline.
  NDC-Cliff the single steepest one-step drop. Flags abrupt rather than
            graceful failure.

Directionality (P1 Section 5.4.3): NDC-AUC and NDC-50 are better when higher,
NDC-Slope and NDC-Cliff are better when lower.

NDC-50 censoring is the subtle part and is handled explicitly.

  Right-censored: accuracy never falls to 0.50 across the measured range. No
  crossing exists. Reported as > 0.90 (the maximum measured ratio), not as a
  number, and flagged as not reaching the threshold.

  Left-censored: accuracy is already below 0.50 at zero noise. Reported as < 0.0,
  meaning the configuration fails even on clean retrieval.

Getting either boundary wrong silently biases the headline noise result, so both
are covered by dedicated tests.
"""

from __future__ import annotations

from dataclasses import dataclass

# The five measured ratios. NDC-AUC integrates over [0, 1], and 0.90 is the
# largest measured point, so the trapezoidal integral is taken over the measured
# span [0, 0.90] and reported as such rather than extrapolated to 1.0.
DEFAULT_RATIOS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 0.90)
THRESHOLD = 0.50


@dataclass
class NDCResult:
    ratios: tuple[float, ...]
    accuracy: tuple[float, ...]
    auc: float
    # None whenever the crossing is CENSORED, never a sentinel number. P1 p.56:
    # a censored value is "reported as greater than ninety percent (the maximum
    # measured noise ratio) rather than as a numeric value, and the
    # configuration is recorded as not reaching the threshold". The previous
    # sentinels were -1.0 and 0.90+1e-9; rounding sent the latter to exactly
    # 0.90, indistinguishable from a genuine crossing at the last measured
    # point, and -1.0 would enter any mean or bootstrap interval as if it were
    # a ratio.
    ndc50: float | None
    ndc50_censoring: str  # "none", "right", or "left"
    slope: float
    cliff: float
    # Baseline P1 requires: single-point accuracy at a fixed noise level (RGB style).
    accuracy_at_fixed_point: float

    @property
    def ndc50_display(self) -> str:
        """How NDC-50 should appear in a table (P1 p.56).

        A censored configuration is reported relative to the measured range
        rather than as a number, so a reader cannot mistake it for a crossing
        that was actually observed.
        """
        if self.ndc50 is not None:
            return f"{self.ndc50:.3f}"
        if self.ndc50_censoring == "right":
            return f"> {self.ratios[-1]:.2f}"
        return f"< {self.ratios[0]:.2f}"

    def as_dict(self) -> dict[str, object]:
        return {
            "ndc_auc": round(self.auc, 4),
            "ndc50": None if self.ndc50 is None else round(self.ndc50, 4),
            "ndc50_censoring": self.ndc50_censoring,
            "ndc50_display": self.ndc50_display,
            "ndc50_reached_threshold": self.ndc50 is not None,
            "ndc_slope": round(self.slope, 4),
            "ndc_cliff": round(self.cliff, 4),
            "curve": {round(r, 2): round(a, 4) for r, a in zip(self.ratios, self.accuracy)},
            "accuracy_at_fixed_point": round(self.accuracy_at_fixed_point, 4),
        }


def _trapezoid(xs: tuple[float, ...], ys: tuple[float, ...]) -> float:
    """Trapezoidal integral of ys over xs, normalised by the x span.

    Normalising by the span makes NDC-AUC a span-independent mean height in
    [0, 1], so it is comparable even if the ratio grid changes. Without this a
    wider grid would inflate the raw area.
    """
    if len(xs) < 2:
        return float(ys[0]) if ys else 0.0
    area = 0.0
    for i in range(len(xs) - 1):
        area += (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2.0
    span = xs[-1] - xs[0]
    return area / span if span > 0 else float(ys[0])


def _find_ndc50(
    ratios: tuple[float, ...], acc: tuple[float, ...]
) -> tuple[float | None, str]:
    """Locate the 0.50 crossing with explicit censoring.

    Returns (value, censoring). The value is a ratio in the measured range for an
    interior crossing and None when censored, because a censored curve has no
    numeric crossing to report (P1 p.56).
    """
    # Left-censored: already below threshold at zero noise, so the crossing lies
    # outside the measured range on the low side -- the configuration fails even
    # on clean retrieval.
    if acc[0] < THRESHOLD:
        return None, "left"

    # Walk the curve looking for the first downward crossing.
    for i in range(len(ratios) - 1):
        a0, a1 = acc[i], acc[i + 1]
        if a0 >= THRESHOLD > a1:
            # Linear interpolation between the bracketing points.
            r0, r1 = ratios[i], ratios[i + 1]
            frac = (a0 - THRESHOLD) / (a0 - a1)  # a0 != a1 since a0 >= T > a1
            return r0 + frac * (r1 - r0), "none"

    # Right-censored: never dropped to the threshold across the measured range.
    return None, "right"


def compute_ndc(
    accuracy_by_ratio: dict[float, float],
    *,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    fixed_point: float = 0.75,
) -> NDCResult:
    """Compute the curve and its four statistics from per-ratio accuracy.

    `accuracy_by_ratio` maps each measured ratio to mean accuracy at that ratio.
    Every ratio in `ratios` must be present, because a missing point would leave
    a gap the interpolation cannot honestly bridge.
    """
    missing = [r for r in ratios if r not in accuracy_by_ratio]
    if missing:
        raise ValueError(f"accuracy missing for ratios {missing}")
    for r, a in accuracy_by_ratio.items():
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"accuracy {a} at ratio {r} out of [0, 1]")

    acc = tuple(accuracy_by_ratio[r] for r in ratios)

    auc = _trapezoid(ratios, acc)
    ndc50_raw, censoring = _find_ndc50(ratios, acc)

    # Finite differences. Only declines contribute to slope and cliff, because an
    # accuracy rise under more noise is noise in the estimate, not robustness to
    # be rewarded.
    drops = [acc[i] - acc[i + 1] for i in range(len(acc) - 1)]  # positive = decline
    negative_drops = [d for d in drops if d > 0]
    # P1 p.56: NDC-Slope is "the average of the negative finite differences
    # between adjacent points". Averaged over the negative differences
    # THEMSELVES, not over every step: dividing the sum of declines by the total
    # step count reports a dip-then-recover curve as 0.100 where P1's formula
    # gives 0.0625, understating how steeply it actually fell where it fell.
    slope = sum(negative_drops) / len(negative_drops) if negative_drops else 0.0
    cliff = max(drops) if drops else 0.0
    cliff = max(cliff, 0.0)  # if accuracy only ever rises, cliff is 0

    return NDCResult(
        ratios=ratios,
        accuracy=acc,
        auc=auc,
        ndc50=ndc50_raw,
        ndc50_censoring=censoring,
        slope=slope,
        cliff=cliff,
        accuracy_at_fixed_point=accuracy_by_ratio[fixed_point],
    )


def accuracy_by_ratio_from_responses(responses: list) -> dict[float, float]:
    """Aggregate scored noise responses into mean accuracy per ratio.

    Accepts ScoredResponse-like objects with `noise_ratio` and `answer_correct`.
    A refusal or unparseable response counts as incorrect for accuracy, which is
    the conservative treatment of P1 Section 5.6.4.
    """
    buckets: dict[float, list[int]] = {}
    for r in responses:
        if r.noise_ratio is None:
            continue
        buckets.setdefault(r.noise_ratio, []).append(1 if r.answer_correct else 0)
    return {ratio: sum(hits) / len(hits) for ratio, hits in buckets.items() if hits}


def mean_ndc50(results: list[NDCResult]) -> float:
    """Mean NDC-50 over configurations that actually reached the threshold.

    Refuses to average a censored curve. P1 p.56 records a censored
    configuration as "not reaching the threshold", which is not a number and
    cannot be summarised as one; silently substituting the range endpoint would
    bias the mean toward whichever side the censoring fell on. Callers that need
    to report on censored configurations should report the COUNT of them
    alongside this mean.
    """
    values = [r.ndc50 for r in results if r.ndc50 is not None]
    if not values:
        raise ValueError(
            "every configuration is censored; NDC-50 has no numeric mean "
            "(report the censoring counts instead)"
        )
    return sum(values) / len(values)
