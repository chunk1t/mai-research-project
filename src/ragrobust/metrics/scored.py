"""Scored response records.

A scored response is what a metric consumes: the case identity, the classified
response category, and the correctness of any substantive answer. Keeping this a
plain, serialisable record means the metrics never touch a live model and can be
unit tested against hand-built inputs.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from ..schema import Dimension


class ResponseCategory(str, Enum):
    """Output of the response parser (P1 Section 5.6.4)."""

    ANSWER = "answer"  # a substantive answer was committed
    REFUSAL = "refusal"  # explicit assertion that evidence is insufficient
    UNPARSEABLE = "unparseable"  # no usable final-answer field


class ScoredResponse(BaseModel):
    """One pipeline response reduced to its scoreable form."""

    case_id: str
    dimension: Dimension
    category: ResponseCategory

    # For answer cases, whether the committed answer matched the ground truth.
    # None when the response was a refusal or unparseable.
    answer_correct: bool | None = None

    # Ground-truth answerability of the case, copied from the benchmark so a
    # metric never has to reach back into the dataset.
    is_answerable: bool = True

    # Noise ratio for noise-dimension responses, else None.
    noise_ratio: float | None = None

    # CRS rubric score 0-4 for conflict responses, filled by the judge.
    crs_score: int | None = None

    # Audit trail from the response parser (P1 Section 5.6.4). The committed
    # field is kept so a disputed classification can be re-examined without
    # re-running the model, and the method records whether the rules were
    # sufficient or the language-model fallback was needed.
    final_answer: str | None = None
    classification_method: str | None = None

    # Correctness components, retained so Chapter 7 can report the stricter
    # exact-match figure alongside the headline number without a re-run.
    exact_match: bool | None = None
    token_f1: float | None = None

    # Carried for the analysis layer (P1 Section 5.7).
    #
    # `seed_id` is the noise bootstrap's resampling unit: a Noise Degradation
    # Curve is fitted across the five ratios of one seed, so resampling
    # instances would tear curves apart and fabricate seeds that never existed.
    #
    # `truncated` marks a response that exhausted its output budget without
    # committing an answer. That is neither a wrong answer nor a refusal, and
    # keeping it visible is what lets CRS be reported both with and without
    # non-terminating responses.
    seed_id: str | None = None
    truncated: bool = False
    recovery_method: str | None = None
