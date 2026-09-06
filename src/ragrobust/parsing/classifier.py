"""Three-way response classification with a language-model fallback.

P1 Section 5.6.4: "The final-answer field is classified into one of three
categories by a rule-based parser backed by a language-model fallback: a
substantive answer, an explicit refusal [...], or unparseable."

The protocol is uniform across all three pipeline classes, which is what stops
trace verbosity from confounding the comparison. A naive one-line answer and a
five-thousand-token agentic transcript are reduced by identical rules, and in
both cases only the delimited field is read.

Two conservative treatments are inherited from P1 and must not drift:

  An unparseable response is a non-refusal for Refusal F1 and incorrect for
  accuracy, "because it penalises rather than rewards ambiguous output".

  A refusal carries no correctness judgement. Whether abstaining was right is
  the business of Refusal F1, not of answer accuracy.

The fallback model must not belong to an evaluated generator's family: it
decides whether an evaluated generator refused, so a same-family classifier
could systematically read its sibling's phrasing as commitment. It is configured
as `refusal_classifier` in `configs/models.yaml` and audited alongside the other
roles (P1 Section 5.9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..metrics.scored import ResponseCategory, ScoredResponse
from ..pipelines.base import PipelineResult
from ..providers.base import GenerationRequest, LLMProvider
from ..schema import TestCase
from .rules import AnswerScore, RuleVerdict, rule_classify, score_answer

CLASSIFIER_PROMPT = """\
You are labelling the output of a question answering system.

Below is the system's final answer field, exactly as it committed it. Decide
which single label applies:

ANSWER      it commits to a substantive answer to the question
REFUSAL     it declines to answer, asserting the evidence is insufficient
UNPARSEABLE it is empty, or contains no usable answer and no clear refusal

A hedged or qualified answer that still commits is ANSWER, not REFUSAL.

Question: {query}

Final answer field:
\"\"\"
{field}
\"\"\"

Reply with exactly one word: ANSWER, REFUSAL, or UNPARSEABLE."""

_LABELS = {
    "ANSWER": ResponseCategory.ANSWER,
    "REFUSAL": ResponseCategory.REFUSAL,
    "UNPARSEABLE": ResponseCategory.UNPARSEABLE,
}


@dataclass
class Classification:
    category: ResponseCategory
    method: str  # "rule", "llm_fallback", or "rule_unconfident_no_fallback"
    reason: str
    warnings: list[str] = field(default_factory=list)


class ResponseParser:
    """Reduces a pipeline response to a scoreable record."""

    def __init__(
        self,
        fallback_provider: LLMProvider | None = None,
        *,
        max_tokens: int = 16,
        temperature: float = 0.0,
        seed: int | None = None,
    ):
        self.fallback_provider = fallback_provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed
        # Counted so Chapter 6 can report how often the rules were sufficient.
        self.counts: dict[str, int] = {}

    def _tally(self, method: str) -> None:
        self.counts[method] = self.counts.get(method, 0) + 1

    async def classify(self, query: str, final_answer: str | None) -> Classification:
        verdict: RuleVerdict = rule_classify(final_answer)
        if verdict.confident:
            self._tally("rule")
            return Classification(verdict.category, "rule", verdict.reason)

        if self.fallback_provider is None:
            # Degrade loudly, never silently. Falling back to the unconfident
            # rule verdict is defensible, but it must be visible in the run
            # record and in the kappa validation rather than assumed away.
            self._tally("rule_unconfident_no_fallback")
            return Classification(
                verdict.category,
                "rule_unconfident_no_fallback",
                verdict.reason,
                warnings=[
                    "SPEC DEVIATION: ambiguous final-answer field and no "
                    "language-model fallback configured (P1 5.6.4); kept the "
                    f"unconfident rule verdict '{verdict.category.value}'"
                ],
            )

        resp = await self.fallback_provider.generate(
            GenerationRequest(
                prompt=CLASSIFIER_PROMPT.format(query=query, field=final_answer or ""),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                thinking=False,
            )
        )
        if not resp.ok:
            self._tally("rule_unconfident_no_fallback")
            return Classification(
                verdict.category,
                "rule_unconfident_no_fallback",
                verdict.reason,
                warnings=[f"fallback classifier failed ({resp.error}); kept rule verdict"],
            )

        label = _parse_label(resp.text)
        if label is None:
            self._tally("rule_unconfident_no_fallback")
            return Classification(
                verdict.category,
                "rule_unconfident_no_fallback",
                verdict.reason,
                warnings=[
                    f"fallback classifier returned no usable label ({resp.text!r}); "
                    "kept rule verdict"
                ],
            )

        self._tally("llm_fallback")
        return Classification(label, "llm_fallback", "language-model fallback")

    async def score(self, case: TestCase, result: PipelineResult) -> ScoredResponse:
        """Reduce one pipeline result to the record the metrics consume."""
        if not result.ok:
            # A provider-level failure is not a model behaviour and must not be
            # scored as one. It is unparseable, which P1 treats conservatively.
            return ScoredResponse(
                case_id=case.case_id,
                dimension=case.dimension,
                category=ResponseCategory.UNPARSEABLE,
                answer_correct=None,
                is_answerable=case.is_answerable,
                noise_ratio=case.noise_ratio,
                final_answer=None,
                classification_method="provider_error",
            )

        classification = await self.classify(case.query, result.final_answer)

        answer_correct: bool | None = None
        answer_score: AnswerScore | None = None
        if classification.category is ResponseCategory.ANSWER and case.is_answerable:
            # Correctness applies only to a committed answer on a case that has
            # one. A refusal's correctness is Refusal F1's business, and an
            # unanswerable case has no answer to match.
            answer_score = score_answer(result.final_answer or "", case.answer)
            answer_correct = answer_score.correct

        return ScoredResponse(
            case_id=case.case_id,
            dimension=case.dimension,
            category=classification.category,
            answer_correct=answer_correct,
            is_answerable=case.is_answerable,
            noise_ratio=case.noise_ratio,
            final_answer=result.final_answer,
            classification_method=classification.method,
            exact_match=None if answer_score is None else answer_score.exact_match,
            token_f1=None if answer_score is None else answer_score.token_f1,
        )


def _parse_label(text: str) -> ResponseCategory | None:
    """Read a label out of the fallback model's reply, tolerating stray prose."""
    upper = text.strip().upper()
    for name, category in _LABELS.items():
        if name in upper:
            return category
    return None


def validation_sample(
    scored: list[ScoredResponse], *, n: int = 100, seed: int = 20260721
) -> list[dict[str, object]]:
    """Draw a sample for human labelling of the classifier (P1 Section 5.6.4).

    "The classifier is validated against human labels on the same
    manual-validation sample used for the dataset, with agreement reported using
    Cohen's kappa, so that any systematic parsing error is detectable rather
    than silent."

    The machine label is included so a human can be shown the field first and
    the label afterwards; feed both columns to `metrics.conflict.cohens_kappa`.
    Sampling is stratified by category so that refusals and unparseable
    responses, which are rarer than answers but where parsing errors matter
    most, are not crowded out of the sample.
    """
    import random

    rng = random.Random(seed)
    by_category: dict[str, list[ScoredResponse]] = {}
    for s in scored:
        by_category.setdefault(s.category.value, []).append(s)

    out: list[ScoredResponse] = []
    categories = sorted(by_category)
    if categories:
        per_category = max(1, n // len(categories))
        for category in categories:
            bucket = by_category[category]
            rng.shuffle(bucket)
            out.extend(bucket[:per_category])
    # Top up from whatever remains if a category was too small to fill its share.
    if len(out) < n:
        chosen = {id(s) for s in out}
        remainder = [s for s in scored if id(s) not in chosen]
        rng.shuffle(remainder)
        out.extend(remainder[: n - len(out)])

    return [
        {
            "case_id": s.case_id,
            "dimension": s.dimension.value,
            "final_answer": s.final_answer,
            "machine_label": s.category.value,
            "classification_method": s.classification_method,
            "human_label": "",
        }
        for s in out[:n]
    ]
