#!/usr/bin/env python3
"""Measure the real serving rate and project the full run (P1 Section 5.8).

The Week 0 de-risking. Runs a representative slice of the benchmark through the
actual pipeline code against the actual endpoint, then extrapolates to the full
matrix. Using the real code path matters: a synthetic ping would miss prefill
cost, which dominates at high noise ratios, and would miss the reasoning arms'
output length, which dominates everything else.

The sample deliberately over-weights the expensive end. Noise cases at ratio
0.90 carry ten times their signal, so a spike drawn uniformly would flatter the
projection.

    # No GPU, no network: proves the harness works before you spend credits.
    python scripts/throughput_spike.py --dry-run

    # The real thing, once MODAL_VLLM_URL and MODAL_VLLM_KEY are in .env.
    python scripts/throughput_spike.py --requests 40 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ragrobust.corpus import SandboxedCorpus  # noqa: E402
from ragrobust.pipelines import NaivePipeline, ReasoningPipeline, RetrievalPolicy  # noqa: E402
from ragrobust.providers.base import (  # noqa: E402
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    RateLimiter,
)
from ragrobust.providers.factory import build_provider, load_config  # noqa: E402
from ragrobust.schema import (  # noqa: E402
    Dimension,
    Passage,
    PerturbationType,
    SeedSource,
    TestCase,
)
from ragrobust.throughput import Sample, project_run  # noqa: E402

# The full matrix from P1 Section 5.6.5.
TOTAL_CALLS = 69_600

# Mix mirroring the benchmark's cost profile rather than its case counts: the
# r=0.90 noise cases are a fifth of the noise testbed but carry ten times the
# tokens, so they are sampled heavily here.
CASE_MIX: tuple[tuple[str, float], ...] = (
    ("refusal", 0.15),
    ("conflict", 0.15),
    ("noise-0.25", 0.15),
    ("noise-0.75", 0.25),
    ("noise-0.90", 0.30),
)

SIGNAL_TOKENS = 700  # median of the real seeds in data/seeds/seeds.json


class DryRunProvider(LLMProvider):
    """Stands in for the endpoint so the harness can be validated offline."""

    def __init__(self, family: str = "alibaba"):
        super().__init__(
            name="dry-run", model="dry-run-model", family=family, limiter=RateLimiter(16)
        )

    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        # A crude stand-in for prefill plus decode, enough to prove the harness
        # measures and projects. The numbers it produces are not evidence.
        await asyncio.sleep(0.01 + len(req.prompt) / 400_000)
        completion = 400 if req.thinking else 60
        return GenerationResponse(
            text="<final_answer>1889</final_answer>",
            model=self.model,
            family=self.family,
            reasoning_trace="..." if req.thinking else None,
            prompt_tokens=len(req.prompt.split()),
            completion_tokens=completion,
        )


def _passage(i: int, tokens: int, *, answer: bool = False, injected: bool = False) -> Passage:
    body = ("the tower district construction record entry " * ((tokens // 6) + 1)).split()
    text = " ".join(body[:tokens])
    if answer:
        text = "The tower was completed in 1889. " + text
    return Passage(
        passage_id=f"p{i}", text=text, is_answer_bearing=answer, is_injected=injected
    )


def synthetic_case(kind: str, index: int) -> TestCase:
    """Build a case with the token profile the real benchmark will have."""
    query = "In what year was the tower completed?"
    if kind == "refusal":
        return TestCase(
            case_id=f"spike-refusal-{index}",
            query=query,
            retrieved_passages=[_passage(0, SIGNAL_TOKENS, answer=True)],
            answer="1889",
            dimension=Dimension.REFUSAL,
            perturbation_type=PerturbationType.ANSWER_PASSAGE_RETAINED,
            seed_source=SeedSource.TRIVIA_QA,
            seed_id=f"s{index}",
        )
    if kind == "conflict":
        return TestCase(
            case_id=f"spike-conflict-{index}",
            query=query,
            retrieved_passages=[
                _passage(0, SIGNAL_TOKENS, answer=True),
                _passage(1, SIGNAL_TOKENS, injected=True),
            ],
            answer="1889",
            dimension=Dimension.CONFLICT,
            perturbation_type=PerturbationType.CONTRADICTION_INJECTED,
            seed_source=SeedSource.TRIVIA_QA,
            seed_id=f"s{index}",
        )

    ratio = float(kind.split("-")[1])
    # Total tokens are signal/(1 - ratio); the balance is distractors.
    distractor_budget = int(SIGNAL_TOKENS * ratio / (1 - ratio))
    per_distractor = 200
    n = max(1, distractor_budget // per_distractor)
    passages = [_passage(0, SIGNAL_TOKENS, answer=True)]
    passages += [
        _passage(i + 1, per_distractor, injected=True) for i in range(n)
    ]
    return TestCase(
        case_id=f"spike-noise{int(ratio * 100)}-{index}",
        query=query,
        retrieved_passages=passages,
        answer="1889",
        dimension=Dimension.NOISE,
        perturbation_type=PerturbationType.DISTRACTORS_ADDED,
        noise_ratio=ratio,
        seed_source=SeedSource.TRIVIA_QA,
        seed_id=f"s{index}",
    )


def build_cases(n: int, rng: random.Random) -> list[TestCase]:
    kinds = [k for k, _ in CASE_MIX]
    weights = [w for _, w in CASE_MIX]
    return [
        synthetic_case(kind, i)
        for i, kind in enumerate(rng.choices(kinds, weights=weights, k=n))
    ]


async def run_one(pipeline, case: TestCase, arm: str) -> Sample:
    corpus = SandboxedCorpus(case, distractor_pool=[])
    started = time.perf_counter()
    result = await pipeline.run(corpus)
    elapsed = time.perf_counter() - started
    return Sample(
        arm=arm,
        latency_s=elapsed,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        ok=result.ok,
    )


async def spike(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    cases = build_cases(args.requests, rng)

    if args.dry_run:
        provider = DryRunProvider()
        arms = [
            ("standard (dry-run)", NaivePipeline(provider, RetrievalPolicy("bm25"))),
            (
                "reasoning (dry-run)",
                ReasoningPipeline(provider, RetrievalPolicy("bm25"), thinking=True),
            ),
        ]
    else:
        cfg = load_config(args.config)
        gens = cfg["generators"]
        arms = []
        for name in args.arms.split(","):
            spec = gens[name.strip()]
            provider = build_provider(name.strip(), spec, cfg)
            policy = RetrievalPolicy(
                "bm25", max_context_tokens=cfg["controls"]["max_context_tokens"]
            )
            await provider.open()
            if spec.get("thinking") or spec.get("cot_prompt"):
                arms.append(
                    (
                        name.strip(),
                        ReasoningPipeline(
                            provider,
                            policy,
                            thinking=bool(spec.get("thinking")),
                            cot_prompt=bool(spec.get("cot_prompt")),
                        ),
                    )
                )
            else:
                arms.append((name.strip(), NaivePipeline(provider, policy)))

    semaphore = asyncio.Semaphore(args.concurrency)

    async def guarded(pipeline, case, arm):
        async with semaphore:
            return await run_one(pipeline, case, arm)

    tasks = [
        guarded(pipeline, case, arm)
        for arm, pipeline in arms
        for case in cases
    ]

    print(
        f"issuing {len(tasks)} calls across {len(arms)} arm(s) "
        f"at concurrency {args.concurrency}...",
        flush=True,
    )
    started = time.perf_counter()
    samples = await asyncio.gather(*tasks)
    wall_clock = time.perf_counter() - started

    projection = project_run(
        list(samples),
        wall_clock_s=wall_clock,
        total_calls=args.total_calls,
        gpu_hourly_usd=args.gpu_hourly_usd,
        budget_hours=args.budget_hours,
    )

    report = projection.as_dict()
    report["wall_clock_s"] = round(wall_clock, 2)
    report["concurrency"] = args.concurrency
    report["dry_run"] = args.dry_run

    print("\n=== THROUGHPUT SPIKE ===")
    print(json.dumps(report, indent=2))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwritten to {args.out}")

    if args.dry_run:
        print(
            "\nDRY RUN: the numbers above come from a simulated provider and are "
            "NOT evidence about the real run. They prove only that the harness "
            "measures, projects, and reports."
        )
    else:
        print(f"\nVERDICT: {projection.verdict} "
              f"({projection.projected_hours:.1f} h projected against a "
              f"{args.budget_hours:.0f} h budget)")
        if projection.n_failed:
            print(f"WARNING: {projection.n_failed} call(s) failed; investigate "
                  "before trusting the rate")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true", help="simulate; no GPU, no network")
    p.add_argument("--requests", type=int, default=20, help="cases per arm")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--arms", default="standard_1,reasoning_1")
    p.add_argument("--config", default="configs/models.yaml")
    p.add_argument("--total-calls", type=int, default=TOTAL_CALLS, dest="total_calls")
    p.add_argument("--gpu-hourly-usd", type=float, default=1.10, dest="gpu_hourly_usd")
    p.add_argument("--budget-hours", type=float, default=48.0, dest="budget_hours")
    p.add_argument("--seed", type=int, default=20260721)
    p.add_argument("--out", default="runs/throughput_spike.json")
    args = p.parse_args()

    if not args.dry_run and not os.environ.get("MODAL_VLLM_URL"):
        print(
            "MODAL_VLLM_URL is unset. Either deploy first and put it in .env, "
            "or run with --dry-run to validate the harness.",
            file=sys.stderr,
        )
        return 2
    return asyncio.run(spike(args))


if __name__ == "__main__":
    raise SystemExit(main())
