"""Throughput projection tests, asserted against hand-computed values.

The spike is the Week 0 de-risking of P1 Section 5.8, so its arithmetic decides
whether the full matrix is attempted. A projection that flatters the run is
worse than no projection, which is why the boundaries are pinned here.
"""

from __future__ import annotations

import pytest

from ragrobust.throughput import Sample, percentile, project_run, summarise_arms


def samples(n: int, *, arm: str = "standard_1", ok: bool = True, **kw) -> list[Sample]:
    return [
        Sample(
            arm=arm,
            latency_s=kw.get("latency_s", 2.0),
            prompt_tokens=kw.get("prompt_tokens", 100),
            completion_tokens=kw.get("completion_tokens", 50),
            ok=ok,
        )
        for _ in range(n)
    ]


def test_percentile_uses_nearest_rank():
    values = [float(i) for i in range(1, 11)]  # 1..10
    # p50: ceil(0.5 * 10) - 1 = index 4 -> 5.0
    assert percentile(values, 0.5) == 5.0
    # p95: ceil(0.95 * 10) - 1 = index 9 -> 10.0
    assert percentile(values, 0.95) == 10.0
    assert percentile(values, 1.0) == 10.0
    # Every reported percentile is an observed latency, never an interpolation.
    assert percentile(values, 0.5) in values


def test_percentile_edges():
    assert percentile([], 0.5) == 0.0
    assert percentile([3.0], 0.95) == 3.0
    with pytest.raises(ValueError):
        percentile([1.0], 0.0)


def test_projection_worked_example():
    # 12 successful calls in 6.0 s wall clock -> 2.0 requests/second.
    # 69,600 / 2.0 = 34,800 s = 9.6667 h. At $1.10/h that is $10.63.
    p = project_run(
        samples(12),
        wall_clock_s=6.0,
        total_calls=69_600,
        gpu_hourly_usd=1.10,
        budget_hours=48.0,
    )
    assert p.observed_rps == pytest.approx(2.0)
    assert p.projected_hours == pytest.approx(9.6667, abs=1e-4)
    assert p.projected_gpu_usd == pytest.approx(10.6333, abs=1e-4)


def test_rate_is_wall_clock_not_reciprocal_latency():
    # Continuous batching means 8 concurrent 2-second calls complete in about
    # 2 seconds, not 16. Using mean latency would understate throughput by
    # roughly the batch size and wrongly declare the run infeasible.
    p = project_run(
        samples(8, latency_s=2.0),
        wall_clock_s=2.0,
        total_calls=3600,
        gpu_hourly_usd=1.0,
        budget_hours=48.0,
    )
    assert p.observed_rps == pytest.approx(4.0)  # 8 / 2.0, not 1 / 2.0
    assert p.projected_hours == pytest.approx(0.25)


@pytest.mark.parametrize(
    "total_calls,expected",
    [
        (86_400, "comfortable"),  # 24.0 h, exactly half the 48 h budget
        (108_000, "fits"),        # 30.0 h, inside 0.8 * 48 = 38.4
        (180_000, "tight"),       # 50.0 h, over budget but inside 1.5x
        (360_000, "does not fit"),  # 100.0 h
    ],
)
def test_verdict_boundaries(total_calls, expected):
    # 10 calls in 10 s -> 1.0 rps, so projected_hours = total_calls / 3600.
    p = project_run(
        samples(10),
        wall_clock_s=10.0,
        total_calls=total_calls,
        gpu_hourly_usd=1.0,
        budget_hours=48.0,
    )
    assert p.verdict == expected


def test_a_run_consuming_the_whole_window_is_called_tight_not_fits():
    # Conservative by design: the projection covers generation only, not dataset
    # construction, judging, analysis, or the report -- and one failed batch
    # costs a re-run.
    p = project_run(
        samples(10),
        wall_clock_s=10.0,
        total_calls=int(47.0 * 3600),  # 47 h against a 48 h budget
        gpu_hourly_usd=1.0,
        budget_hours=48.0,
    )
    assert p.projected_hours == pytest.approx(47.0)
    assert p.verdict == "tight"


def test_failures_are_excluded_from_the_rate_but_reported():
    # A failure rate averaged into throughput would hide the very problem the
    # spike exists to find.
    rows = samples(8) + samples(2, ok=False)
    p = project_run(
        rows,
        wall_clock_s=4.0,
        total_calls=7200,
        gpu_hourly_usd=1.0,
        budget_hours=48.0,
    )
    assert p.observed_rps == pytest.approx(2.0)  # 8 successes / 4 s, not 10 / 4
    assert p.n_failed == 2


def test_arms_are_reported_separately():
    # Reasoning arms emit far more output than standard ones (P1 5.8), so a
    # single blended figure would hide the cost driver.
    rows = samples(4, arm="standard_1", completion_tokens=60)
    rows += samples(4, arm="reasoning_1", completion_tokens=400)
    stats = {a.arm: a for a in summarise_arms(rows)}
    assert stats["standard_1"].mean_completion_tokens == 60.0
    assert stats["reasoning_1"].mean_completion_tokens == 400.0
    assert stats["reasoning_1"].n_failed == 0


def test_projection_refuses_to_guess_without_successes():
    with pytest.raises(ValueError, match="no successful samples"):
        project_run(
            samples(3, ok=False),
            wall_clock_s=1.0,
            total_calls=100,
            gpu_hourly_usd=1.0,
            budget_hours=48.0,
        )


def test_projection_rejects_impossible_inputs():
    with pytest.raises(ValueError, match="wall_clock_s"):
        project_run(samples(1), wall_clock_s=0.0, total_calls=10, gpu_hourly_usd=1.0, budget_hours=1.0)
    with pytest.raises(ValueError, match="total_calls"):
        project_run(samples(1), wall_clock_s=1.0, total_calls=0, gpu_hourly_usd=1.0, budget_hours=1.0)
    with pytest.raises(ValueError, match="budget_hours"):
        project_run(samples(1), wall_clock_s=1.0, total_calls=10, gpu_hourly_usd=1.0, budget_hours=0.0)


def test_glm_serving_command_passes_trust_remote_code():
    """GLM-4 ships modelling code in its weights repo; vLLM refuses without it.

    Without the flag the server exits during startup, so no request ever gets an
    error -- the arm just looks like a cold start that never finishes. That hid
    the failure behind a routing bug for an entire session.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "deploy" / "modal_vllm.py").read_text()
    assert '"glm" in model_id.lower()' in src
    assert '"--trust-remote-code"' in src
    # Scoped, not global: the flag executes repository code, so it should apply
    # only to the models that require it.
    assert src.index('"qwen" in model_id.lower()') < src.index('"--trust-remote-code"')
