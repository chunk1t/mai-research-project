"""Throughput measurement and run projection.

P1 Section 5.8 makes the run's feasibility an explicit risk: the workload is
"on the order of seventy thousand generator calls" and the budget is fixed. The
project plan calls this the Week 0 de-risking — measure the real serving rate
early, then decide whether the full matrix fits the window, rather than
discovering it four days in.

The arithmetic lives here, separate from the script that gathers the samples, so
it can be tested against hand-computed values with no GPU and no network.

Percentiles use the nearest-rank method: for n samples the p-th percentile is
the value at index ceil(p * n) - 1 of the sorted list. It needs no interpolation
convention, so a reported p95 is always an observed latency rather than a
number that never occurred.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Sample:
    """One generator call observed during the spike."""

    arm: str  # which generator configuration produced it
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    ok: bool = True


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. `p` is a fraction in (0, 1]."""
    if not values:
        return 0.0
    if not 0.0 < p <= 1.0:
        raise ValueError("percentile fraction must be in (0, 1]")
    ordered = sorted(values)
    index = math.ceil(p * len(ordered)) - 1
    return ordered[index]


@dataclass
class ArmStats:
    """Per-configuration observation. Reasoning arms emit far more output."""

    arm: str
    n_ok: int
    n_failed: int
    mean_prompt_tokens: float
    mean_completion_tokens: float
    p50_latency_s: float
    p95_latency_s: float


@dataclass
class Projection:
    """What the observed rate implies for the full run."""

    observed_rps: float
    total_calls: int
    projected_hours: float
    projected_gpu_usd: float
    budget_hours: float
    verdict: str  # "comfortable", "fits", "tight", or "does not fit"
    arms: list[ArmStats] = field(default_factory=list)
    n_failed: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "observed_rps": round(self.observed_rps, 4),
            "total_calls": self.total_calls,
            "projected_hours": round(self.projected_hours, 3),
            "projected_gpu_usd": round(self.projected_gpu_usd, 2),
            "budget_hours": self.budget_hours,
            "verdict": self.verdict,
            "n_failed": self.n_failed,
            "arms": [
                {
                    "arm": a.arm,
                    "n_ok": a.n_ok,
                    "n_failed": a.n_failed,
                    "mean_prompt_tokens": round(a.mean_prompt_tokens, 1),
                    "mean_completion_tokens": round(a.mean_completion_tokens, 1),
                    "p50_latency_s": round(a.p50_latency_s, 3),
                    "p95_latency_s": round(a.p95_latency_s, 3),
                }
                for a in self.arms
            ],
        }


def _verdict(projected_hours: float, budget_hours: float) -> str:
    """Grade the projection against the time available.

    Deliberately conservative. A run projected to consume the entire remaining
    window is called "tight", not "fits", because the projection excludes
    dataset construction, judging, analysis, and the report — and because a
    single failed batch costs a re-run.
    """
    if budget_hours <= 0:
        raise ValueError("budget_hours must be positive")
    if projected_hours <= budget_hours * 0.5:
        return "comfortable"
    if projected_hours <= budget_hours * 0.8:
        return "fits"
    if projected_hours <= budget_hours * 1.5:
        return "tight"
    return "does not fit"


def summarise_arms(samples: list[Sample]) -> list[ArmStats]:
    by_arm: dict[str, list[Sample]] = {}
    for s in samples:
        by_arm.setdefault(s.arm, []).append(s)

    out: list[ArmStats] = []
    for arm in sorted(by_arm):
        rows = by_arm[arm]
        ok = [r for r in rows if r.ok]
        latencies = [r.latency_s for r in ok]
        out.append(
            ArmStats(
                arm=arm,
                n_ok=len(ok),
                n_failed=len(rows) - len(ok),
                mean_prompt_tokens=(
                    sum(r.prompt_tokens for r in ok) / len(ok) if ok else 0.0
                ),
                mean_completion_tokens=(
                    sum(r.completion_tokens for r in ok) / len(ok) if ok else 0.0
                ),
                p50_latency_s=percentile(latencies, 0.5),
                p95_latency_s=percentile(latencies, 0.95),
            )
        )
    return out


def project_run(
    samples: list[Sample],
    *,
    wall_clock_s: float,
    total_calls: int,
    gpu_hourly_usd: float,
    budget_hours: float,
) -> Projection:
    """Extrapolate the observed serving rate to the full experimental matrix.

    The rate is measured as completed calls over wall-clock time at the spike's
    concurrency, not as the reciprocal of mean latency. With vLLM's continuous
    batching those two differ by roughly the batch size, and using latency would
    understate throughput by an order of magnitude.

    Failed calls are excluded from the rate but reported, because a failure rate
    that is quietly averaged into a throughput figure hides the very problem the
    spike exists to find.
    """
    if wall_clock_s <= 0:
        raise ValueError("wall_clock_s must be positive")
    if total_calls <= 0:
        raise ValueError("total_calls must be positive")

    ok = [s for s in samples if s.ok]
    n_failed = len(samples) - len(ok)
    if not ok:
        raise ValueError("no successful samples; cannot project a run")

    observed_rps = len(ok) / wall_clock_s
    projected_hours = (total_calls / observed_rps) / 3600.0

    return Projection(
        observed_rps=observed_rps,
        total_calls=total_calls,
        projected_hours=projected_hours,
        projected_gpu_usd=projected_hours * gpu_hourly_usd,
        budget_hours=budget_hours,
        verdict=_verdict(projected_hours, budget_hours),
        arms=summarise_arms(samples),
        n_failed=n_failed,
    )
