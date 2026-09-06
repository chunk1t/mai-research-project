"""Conflict testbed construction (P1 Section 5.5.2).

A second passage is generated that contradicts the original answer in a single
identifiable claim while remaining plausible. Generation is performed by the
case-generator model and then filtered through a natural language inference
check. Cases where a clean contradiction cannot be produced are discarded rather
than repaired, because a case whose contradiction is ambiguous cannot be scored
against the graded CRS rubric.

The anti-circularity requirement of P1 Section 5.9 applies with particular force
here. The model that writes the contradiction must not belong to the same family
as any evaluated generator or the CRS judge, or a generator could exploit
artefacts left by a sibling and a judge could reward them.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Protocol

from ..schema import Dimension, PerturbationType, Passage, TestCase
from .seeds import Seed

CONTRADICTION_PROMPT = """\
You are constructing test data for a question answering benchmark.

Below is a passage and a factual question it answers. Rewrite the passage so that
it gives a DIFFERENT but PLAUSIBLE answer to the question. Follow these rules:

1. Change only the single specific claim that answers the question.
2. Keep every other detail, the topic, the tone, and the length unchanged.
3. The new answer must be plausible, not absurd. If the true answer is a year,
   shift it by a few years, not by centuries.
4. Do not mention that anything was changed. Do not add hedging language.
5. Return only the rewritten passage, with no preamble.

Question: {query}
True answer: {answer}
Passage:
{passage}
"""


class NLIModel(Protocol):
    """Entailment checker. Satisfied by a DeBERTa MNLI pipeline wrapper."""

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return label probabilities including a 'contradiction' key."""


@dataclass
class ConflictCandidate:
    seed: Seed
    contradictory_text: str
    contradiction_score: float
    accepted: bool
    reject_reason: str | None = None


def _extract_answer_sentence(passage: str, answer: str) -> str | None:
    """Find the sentence carrying the answer, used as the NLI premise.

    Running NLI over a whole passage dilutes the signal, because most sentences
    are unchanged and entail one another. Restricting the check to the sentence
    that actually carries the claim is what makes the contradiction score
    meaningful.
    """
    needle = answer.strip().lower()
    for sent in re.split(r"(?<=[.!?])\s+", passage):
        if needle and needle in sent.lower():
            return sent.strip()
    return None


def _aligned_sentence(generated: str, premise: str) -> str:
    """Find the rewritten counterpart of the premise sentence.

    The generator is told to change only the single claim that answers the
    question and to leave everything else intact, so the rewritten sentence is
    the one that still overlaps the premise most heavily. Token overlap locates
    it without needing to know the substituted answer, which by construction is
    unknown.

    This replaces a first-400-characters fallback. On the short passages used
    in testing that happened to include the changed claim, but real evidence
    runs to ~700 tokens, so the prefix would routinely have missed the
    contradiction entirely and rejected good candidates -- burning free-tier
    quota and biasing the testbed toward whichever passages happen to state
    their claim early.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", generated) if s.strip()]
    if not sentences:
        return generated[:400]

    premise_tokens = set(premise.lower().split())
    if not premise_tokens:
        return sentences[0]

    def overlap(sentence: str) -> float:
        tokens = set(sentence.lower().split())
        if not tokens:
            return 0.0
        return len(tokens & premise_tokens) / len(tokens | premise_tokens)

    return max(sentences, key=overlap)


def verify_contradiction(
    original_passage: str,
    generated_passage: str,
    answer: str,
    nli: NLIModel,
    *,
    threshold: float = 0.70,
) -> tuple[bool, float, str | None]:
    """Confirm the generated passage genuinely contradicts the original claim.

    Three ways a candidate is rejected.

    The generator returned the passage essentially unchanged, meaning no
    contradiction was introduced at all.

    The generated passage still contains the true answer, so it asserts both
    positions and the case would be incoherent.

    The NLI model does not score the pair as a contradiction above threshold,
    meaning the disagreement is not clean enough to grade against the rubric.
    """
    if generated_passage.strip() == original_passage.strip():
        return False, 0.0, "unchanged"

    if answer.strip().lower() in generated_passage.lower():
        return False, 0.0, "true_answer_still_present"

    premise = _extract_answer_sentence(original_passage, answer)
    if premise is None:
        return False, 0.0, "answer_sentence_not_locatable"

    hypothesis = _aligned_sentence(generated_passage, premise)
    scores = nli.predict(premise, hypothesis)
    contra = float(scores.get("contradiction", 0.0))
    if contra < threshold:
        return False, contra, "below_contradiction_threshold"
    return True, contra, None


def build_conflict_case(
    seed: Seed,
    contradictory_text: str,
    contradiction_score: float,
    *,
    gen_params: dict[str, object],
    rng_seed: int = 20260721,
) -> TestCase:
    """Pair the original answer-bearing passage with its contradiction.

    Both are present in the retrieved set, which is what makes the case a
    conflict rather than a substitution. The contradictory passage is flagged as
    injected but not as answer bearing, since it does not support the ground
    truth.

    Presentation order is randomised, which P1 Section 5.9 requires explicitly:
    the Zheng et al. (2023) judge-bias mitigations are applied "including
    randomized presentation order". Storing the true passage first in every case
    would give the CRS judge a positional shortcut perfectly correlated with the
    correct answer, so a judge with a position bias would score well on the
    rubric without resolving the conflict at all -- and the kappa against human
    raters could not distinguish that from genuine competence.

    Randomised per case rather than globally, and derived from the seed id, so
    the assignment is reproducible from the recorded seed (P1 5.6.5 requires the
    random seed to be recorded) and stable across rebuilds. `random.Random` is
    seeded with a string here deliberately: the built-in `hash()` is salted per
    process, so a hash-derived order would differ between runs.
    """
    original = seed.answer_passages[0]
    contradiction = Passage(
        passage_id=f"{original.passage_id}-contra",
        text=contradictory_text,
        is_answer_bearing=False,
        is_injected=True,
    )
    passages = [original, contradiction]
    random.Random(f"{rng_seed}:{seed.seed_id}").shuffle(passages)

    return TestCase(
        case_id=f"conflict-{seed.seed_id}",
        query=seed.query,
        retrieved_passages=passages,
        answer=seed.answer,
        dimension=Dimension.CONFLICT,
        perturbation_type=PerturbationType.CONTRADICTION_INJECTED,
        seed_source=seed.source,
        seed_id=seed.seed_id,
        gen_params={
            **gen_params,
            "contradiction_score": round(contradiction_score, 4),
            "contradicted_passage_id": original.passage_id,
            # Recorded so the randomisation is auditable after the fact and so
            # any residual position effect can be tested for in Chapter 7.
            "true_passage_position": passages.index(original),
            "presentation_order_seed": rng_seed,
        },
    )


def summarise_candidates(candidates: list[ConflictCandidate]) -> dict[str, object]:
    """Acceptance breakdown, reported with the benchmark release."""
    reasons: dict[str, int] = {}
    for c in candidates:
        if not c.accepted and c.reject_reason:
            reasons[c.reject_reason] = reasons.get(c.reject_reason, 0) + 1
    accepted = sum(1 for c in candidates if c.accepted)
    return {
        "n_candidates": len(candidates),
        "n_accepted": accepted,
        "acceptance_rate": round(accepted / len(candidates), 4) if candidates else 0.0,
        "rejections_by_reason": reasons,
    }
