"""Noise testbed construction (P1 Section 5.5.2).

The answer-bearing passage is preserved and distractors are added at five noise
ratios: 0, 25, 50, 75, and 90 percent. Correct behaviour is an accurate answer
that is not derailed by the distractors.

Three implementation details carry experimental weight.

The ratio is over tokens, not passage counts. P1 Section 5.5.2 specifies that the
proportion of distractor tokens is matched to the target ratio, so a handful of
long distractors and many short ones represent the same amount of noise.

Distractors must be topically related. Obviously off-topic distractors are
trivially ignorable and would measure topic filtering rather than robustness.

Distractor positions are randomised. P1 holds positional sensitivity constant and
studies the ratio rather than the position, so position must not covary with the
ratio and leak a second variable into the comparison.
"""

from __future__ import annotations

import random
import re

import numpy as np

from ..schema import NOISE_RATIOS, Dimension, PerturbationType, Passage, TestCase
from .seeds import Seed, distractor_states_answer


MIN_DISTRACTOR_TOKENS = 15


def select_distractors_for_ratio(
    answer_tokens: int,
    candidates: list[Passage],
    target_ratio: float,
    rng: random.Random,
    *,
    min_distractor_tokens: int = MIN_DISTRACTOR_TOKENS,
) -> list[Passage]:
    """Choose distractors so distractor tokens hit the target proportion.

    For a target ratio r, distractor tokens d and signal tokens s must satisfy
    d / (d + s) = r, so d = s * r / (1 - r).

    At r = 0.90 this means nine distractor tokens for every signal token, which
    is why the pool has to be large enough at the top of the range.

    Whole passages are too coarse to hit low ratios accurately. At r = 0.25 with
    roughly 90 signal tokens the budget is only about 30 tokens, while a typical
    distractor runs to 100, so appending whole passages overshoots badly. The
    final distractor is therefore truncated to land on the budget exactly. A
    truncated passage is still a realistic retrieval unit, and precise ratio
    control matters more here because the ratio is the x axis of the Noise
    Degradation Curve. A case that misses its target sits at the wrong point on
    the curve and biases NDC-AUC and NDC-50.

    Truncation is skipped when the remaining budget is below
    `min_distractor_tokens`, since a three-word fragment is not a plausible
    passage and would look like a generation artefact during manual validation.
    """
    if target_ratio <= 0.0:
        return []
    if target_ratio >= 1.0:
        raise ValueError("noise ratio must be below 1.0")

    budget = answer_tokens * target_ratio / (1.0 - target_ratio)
    # Nearest first, in the order select_distractor_pool() ranked them.
    #
    # This used to shuffle. That silently undid the ranking: at r=0.25 a case
    # needs roughly one distractor, so a shuffle drew a random member of the
    # 12-passage pool rather than the closest one. Measured across the 2,000
    # non-zero-ratio instances, only 36% of chosen distractors came from the
    # seed's own article -- exactly the rate a uniform draw predicts -- and 314
    # of 500 cases at r=0.25 ended up with no same-article distractor at all,
    # which is what left them failing P1 5.5.3's "topically related" review.
    #
    # Distractor POSITION is still randomised, by build_noise_cases_for_seed
    # below; P1 5.5.2 holds position constant and studies the ratio, and that
    # control is unaffected by which passages are chosen.
    pool = list(candidates)

    chosen: list[Passage] = []
    used = 0.0
    for p in pool:
        remaining = budget - used
        if remaining < min_distractor_tokens:
            break
        tok = p.token_estimate()
        if tok <= remaining:
            chosen.append(p)
            used += tok
        else:
            words = p.text.split()[: int(remaining)]
            chosen.append(
                Passage(
                    passage_id=f"{p.passage_id}-trunc",
                    text=" ".join(words),
                    is_injected=True,
                )
            )
            used += len(words)
            break
    return chosen


def achieved_ratio(passages: list[Passage]) -> float:
    """Actual distractor token proportion. Reported for validation."""
    total = sum(p.token_estimate() for p in passages)
    if total == 0:
        return 0.0
    noise = sum(p.token_estimate() for p in passages if p.is_injected)
    return noise / total


def build_noise_cases_for_seed(
    seed: Seed,
    distractor_pool: list[Passage],
    *,
    rng_seed: int,
    ratios: tuple[float, ...] = NOISE_RATIOS,
    tolerance: float = 0.08,
) -> tuple[list[TestCase], list[str]]:
    """Emit one case per noise ratio for a single seed.

    Each seed therefore contributes five evaluation instances, which is how 500
    noise cases become 2,500 instances in P1 Section 5.6.5.
    """
    rng = random.Random(f"{rng_seed}-{seed.seed_id}")
    signal = list(seed.passages)
    signal_tokens = sum(p.token_estimate() for p in signal)

    cases: list[TestCase] = []
    warnings: list[str] = []

    for r in ratios:
        distractors = select_distractors_for_ratio(signal_tokens, distractor_pool, r, rng)
        passages = signal + distractors
        # Randomise position so that ratio and position do not covary.
        rng.shuffle(passages)

        got = achieved_ratio(passages)
        if abs(got - r) > tolerance:
            warnings.append(
                f"{seed.seed_id} ratio {r:.2f}: achieved {got:.2f}, "
                f"outside tolerance {tolerance}"
            )

        cases.append(
            TestCase(
                case_id=f"noise-{seed.seed_id}-r{int(r * 100):02d}",
                query=seed.query,
                retrieved_passages=passages,
                answer=seed.answer,
                dimension=Dimension.NOISE,
                perturbation_type=PerturbationType.DISTRACTORS_ADDED,
                noise_ratio=r,
                seed_source=seed.source,
                seed_id=seed.seed_id,
                gen_params={
                    "rng_seed": rng_seed,
                    "target_ratio": r,
                    "achieved_ratio": round(got, 4),
                    "n_distractors": len(distractors),
                },
            )
        )

    return cases, warnings




def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / max(na * nb, 1e-9))


def select_distractor_pool(
    seed: Seed,
    distractor_source: list[Passage],
    embeddings: dict[str, np.ndarray] | None,
    *,
    similarity_floor: float = 0.30,
    similarity_ceiling: float = 0.95,
    max_pool: int | None = None,
) -> tuple[list[Passage], str | None]:
    """Filter candidates down to topically related, answer-free distractors.

    P1 Section 5.5.2 requires distractors to be topically related, so that the
    testbed does not degenerate into the trivial case where distractors are
    obviously off-topic and easily ignored. Without this filter the noise
    dimension would partly measure topic detection rather than robustness under
    noise, and would inflate scores for every pipeline class equally.

    The anchor is the seed's answer-bearing passage, which is the same anchor
    the refusal testbed uses, so "topically related" means the same thing in
    both testbeds.

    The ceiling excludes near-duplicates of the answer passage, which are the
    candidates most likely to restate the answer through paraphrase and so slip
    past the lexical answer check.

    `max_pool` keeps only the nearest neighbours. P1 Section 5.5.2 selects by
    "retrieving topically nearest neighbours", and a floor alone does not do
    that: `select_distractors_for_ratio` shuffles the pool, so an unbounded pool
    of several hundred admissible passages draws mostly from its weakest tail.
    Ranking and truncating makes the selection nearest-neighbour in fact rather
    than merely above-threshold.
    """
    needle = seed.answer.strip().lower()
    own = {p.passage_id for p in seed.passages}

    base = [
        p
        for p in distractor_source
        if p.passage_id not in own
        and not p.is_answer_bearing
        and not distractor_states_answer(p.text, seed.answer)
    ]

    if embeddings is None:
        # Degrade loudly, never silently. A pool built without the topical
        # filter is not the pool P1 specifies.
        return (
            [
                Passage(passage_id=f"dist-{p.passage_id}", text=p.text, is_injected=True)
                for p in base
            ],
            "no embeddings supplied, topical similarity filter SKIPPED",
        )

    anchors = [p for p in seed.answer_passages if p.passage_id in embeddings]
    if not anchors:
        return [], "answer passage has no embedding, cannot apply topical filter"
    anchor = embeddings[anchors[0].passage_id]

    scored: list[tuple[float, Passage]] = []
    for p in base:
        if p.passage_id not in embeddings:
            continue
        sim = _cosine(anchor, embeddings[p.passage_id])
        if similarity_floor <= sim <= similarity_ceiling:
            scored.append((sim, p))

    # Nearest first, with the passage id breaking ties so the pool is stable
    # across runs regardless of the order candidates arrived in.
    scored.sort(key=lambda x: (-x[0], x[1].passage_id))
    if max_pool is not None:
        scored = scored[:max_pool]

    return [
        Passage(passage_id=f"dist-{p.passage_id}", text=p.text, is_injected=True)
        for _, p in scored
    ], None


def build_noise_testbed(
    seeds: list[Seed],
    distractor_source: list[Passage],
    embeddings: dict[str, np.ndarray] | None = None,
    *,
    n_cases: int = 500,
    rng_seed: int = 20260721,
    similarity_floor: float = 0.30,
    similarity_ceiling: float = 0.95,
    max_pool: int | None = None,
) -> tuple[list[TestCase], dict[str, object]]:
    """Construct the full noise testbed.

    Note that `n_cases` counts seeds, not instances. 500 seeds across 5 ratios
    yields 2,500 evaluation instances.
    """
    chosen = seeds[:n_cases]

    all_cases: list[TestCase] = []
    all_warnings: list[str] = []

    # A silent shortfall would quietly shrink the noise testbed and change the
    # instance count the analysis assumes. Surface it loudly instead.
    if len(chosen) < n_cases:
        all_warnings.append(
            f"SHORTFALL: requested {n_cases} noise seeds, only {len(chosen)} available"
        )
    if embeddings is None:
        all_warnings.append(
            "SPEC DEVIATION: no embeddings supplied, distractors are not "
            "topically filtered (P1 Section 5.5.2)"
        )

    for s in chosen:
        pool, warn = select_distractor_pool(
            s,
            distractor_source,
            embeddings,
            similarity_floor=similarity_floor,
            similarity_ceiling=similarity_ceiling,
            max_pool=max_pool,
        )
        if warn:
            all_warnings.append(f"{s.seed_id}: {warn}")
        if not pool:
            all_warnings.append(f"{s.seed_id}: empty distractor pool, skipped")
            continue

        cases, warns = build_noise_cases_for_seed(s, pool, rng_seed=rng_seed)
        all_cases.extend(cases)
        all_warnings.extend(warns)

    stats = {
        "n_seeds": len({c.seed_id for c in all_cases}),
        "n_instances": len(all_cases),
        "ratios": list(NOISE_RATIOS),
        "n_warnings": len(all_warnings),
        "warnings": all_warnings[:20],
    }
    return all_cases, stats
