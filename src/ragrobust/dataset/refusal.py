"""Refusal testbed construction (P1 Section 5.5.2).

Two case types in equal proportion.

Unanswerable cases replace the answer-bearing passage with a topically related
passage from the same corpus that does not support the answer. Correct behaviour
is an explicit refusal.

Answerable controls leave the answer-bearing passage in place. Correct behaviour
is to answer. These controls are not optional decoration. Without them a refusal
could never be wrong, refusal precision would equal one by construction, and
Refusal F1 would collapse into recall alone (P1 Section 5.4.3).
"""

from __future__ import annotations

import random

import numpy as np

from ..schema import NO_ANSWER, Dimension, PerturbationType, Passage, TestCase
from .seeds import Seed, distractor_states_answer


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / max(na * nb, 1e-9))


def select_replacement_passage(
    target: Passage,
    answer: str,
    candidates: list[Passage],
    embeddings: dict[str, np.ndarray],
    *,
    similarity_floor: float = 0.45,
    similarity_ceiling: float = 0.80,
) -> Passage | None:
    """Pick a passage that is topically near the original but answer free.

    The two thresholds encode the requirement of P1 Section 5.5.2 that context
    remains plausible while no longer supporting the answer.

    The floor keeps the replacement on topic. A wildly unrelated passage would
    make refusal trivially easy and would test topic detection rather than
    evidence sufficiency.

    The ceiling discards near-duplicates of the original passage, which are the
    candidates most likely to still carry the answer through paraphrase.

    A lexical check on the answer string runs in addition to the embedding
    thresholds, because embedding similarity alone does not guarantee that a
    surface form of the answer is absent.
    """
    if target.passage_id not in embeddings:
        return None
    t_emb = embeddings[target.passage_id]
    needle = answer.strip().lower()

    scored: list[tuple[float, Passage]] = []
    for c in candidates:
        if c.passage_id == target.passage_id or c.is_answer_bearing:
            continue
        if needle and distractor_states_answer(c.text, answer):
            continue
        if c.passage_id not in embeddings:
            continue
        sim = _cosine(t_emb, embeddings[c.passage_id])
        if similarity_floor <= sim <= similarity_ceiling:
            scored.append((sim, c))

    if not scored:
        return None
    # Take the most similar acceptable candidate, which is the hardest case that
    # still satisfies the constraints.
    scored.sort(key=lambda x: -x[0])
    best_sim, best = scored[0]
    return Passage(
        passage_id=best.passage_id,
        text=best.text,
        is_answer_bearing=False,
        is_injected=True,
    )


def build_unanswerable_case(
    seed: Seed,
    replacement: Passage,
    *,
    gen_params: dict[str, object],
) -> TestCase:
    """Replace every answer-bearing passage with the answer-free replacement."""
    passages = [p for p in seed.passages if not p.is_answer_bearing]
    passages.append(replacement)

    return TestCase(
        case_id=f"refusal-unans-{seed.seed_id}",
        query=seed.query,
        retrieved_passages=passages,
        answer=NO_ANSWER,
        dimension=Dimension.REFUSAL,
        perturbation_type=PerturbationType.ANSWER_PASSAGE_REMOVED,
        seed_source=seed.source,
        seed_id=seed.seed_id,
        gen_params={**gen_params, "replaced_passage_id": replacement.passage_id},
    )


def build_answerable_control(seed: Seed, *, gen_params: dict[str, object]) -> TestCase:
    """Leave the evidence intact. A refusal here is a false positive."""
    return TestCase(
        case_id=f"refusal-ctrl-{seed.seed_id}",
        query=seed.query,
        retrieved_passages=list(seed.passages),
        answer=seed.answer,
        dimension=Dimension.REFUSAL,
        perturbation_type=PerturbationType.ANSWER_PASSAGE_RETAINED,
        seed_source=seed.source,
        seed_id=seed.seed_id,
        gen_params=dict(gen_params),
    )


def build_refusal_testbed(
    seeds: list[Seed],
    embeddings: dict[str, np.ndarray],
    *,
    n_unanswerable: int = 100,
    n_answerable: int = 100,
    similarity_floor: float = 0.45,
    similarity_ceiling: float = 0.80,
    rng_seed: int = 20260721,
) -> tuple[list[TestCase], dict[str, int]]:
    """Construct a balanced refusal testbed.

    Balance matters. P1 Section 5.5.2 splits the testbed roughly evenly so that
    precision and recall are both well defined and neither dominates F1.
    """
    rng = random.Random(rng_seed)
    pool = list(seeds)
    rng.shuffle(pool)

    all_passages = [p for s in pool for p in s.passages]
    gen_params = {
        "rng_seed": rng_seed,
        "similarity_floor": similarity_floor,
        "similarity_ceiling": similarity_ceiling,
    }

    cases: list[TestCase] = []
    stats = {"unanswerable": 0, "answerable": 0, "no_replacement_found": 0}

    # Disjoint seed partitions. A seed must not appear as both an unanswerable
    # case and its own control, or the two would not be independent instances.
    # Index-based rather than iterator-based: consuming the shared iterator to
    # test the loop condition discards one seed at the phase boundary, which
    # silently unbalances the testbed by one case.
    i = 0
    while i < len(pool) and stats["unanswerable"] < n_unanswerable:
        s = pool[i]
        i += 1
        target = s.answer_passages[0]
        repl = select_replacement_passage(
            target,
            s.answer,
            all_passages,
            embeddings,
            similarity_floor=similarity_floor,
            similarity_ceiling=similarity_ceiling,
        )
        if repl is None:
            stats["no_replacement_found"] += 1
            continue
        cases.append(build_unanswerable_case(s, repl, gen_params=gen_params))
        stats["unanswerable"] += 1

    while i < len(pool) and stats["answerable"] < n_answerable:
        cases.append(build_answerable_control(pool[i], gen_params=gen_params))
        stats["answerable"] += 1
        i += 1

    # Balance is what makes Refusal F1 precision and recall comparable
    # (P1 Section 5.4.3), so a shortfall must not pass silently.
    if stats["unanswerable"] < n_unanswerable or stats["answerable"] < n_answerable:
        stats["SHORTFALL"] = 1

    return cases, stats
