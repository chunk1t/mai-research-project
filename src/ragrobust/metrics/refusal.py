"""Refusal F1 (P1 Sections 5.4.3, RQ1).

Refusal F1 is the harmonic mean of refusal precision and refusal recall over the
balanced refusal testbed.

The confusion matrix is defined on the abstention decision, not on answer
correctness.

  A case REQUIRES a refusal when it is unanswerable.
  A case FORBIDS a refusal when it is an answerable control.

  true positive   unanswerable case, system refused
  false positive  answerable case, system refused    (over-caution)
  false negative  unanswerable case, system answered (hallucination risk)
  true negative   answerable case, system answered

  refusal precision = TP / (TP + FP)   fraction of refusals that were warranted
  refusal recall    = TP / (TP + FN)   fraction of warranted refusals achieved

This is why the testbed must contain answerable controls. Without them FP is
structurally zero, precision is one by construction, and F1 collapses into recall
(P1 Section 5.4.3).

An unparseable response is treated as a non-refusal, the conservative choice of
P1 Section 5.6.4, because it must not be rewarded as a lucky abstention.
"""

from __future__ import annotations

from dataclasses import dataclass

from .scored import ResponseCategory, ScoredResponse


@dataclass
class RefusalResult:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int
    n: int
    # Baselines P1 requires for the "does the metric add signal" comparison.
    plain_accuracy: float  # counts a correct refusal as wrong
    negative_rejection_rate: float  # RGB-style: recall on unanswerable cases only

    @property
    def degenerate(self) -> str | None:
        """Why this Refusal F1 is not interpretable, if it is not.

        P1 5.4.3 makes the balance load-bearing: without answerable controls a
        refusal could never be wrong, so precision is 1 by construction; without
        unanswerable cases there is nothing to recall. Either way the harmonic
        mean still returns a number -- 0.0 in the second case -- which reads in
        a results table exactly like a system that never abstains correctly.
        Reporting the reason is what stops that being mistaken for a finding.
        """
        if self.tp + self.fn == 0:
            return "no unanswerable cases: recall is undefined and F1 collapses to 0"
        if self.fp + self.tn == 0:
            return "no answerable controls: precision is 1 by construction (P1 5.4.3)"
        return None

    def as_dict(self) -> dict[str, float | int]:
        return {
            "refusal_precision": round(self.precision, 4),
            "refusal_recall": round(self.recall, 4),
            "refusal_f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "n": self.n,
            "plain_accuracy": round(self.plain_accuracy, 4),
            "negative_rejection_rate": round(self.negative_rejection_rate, 4),
            "n_unanswerable": self.tp + self.fn,
            "n_answerable": self.fp + self.tn,
            "degenerate": self.degenerate,
        }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_refusal_f1(responses: list[ScoredResponse]) -> RefusalResult:
    """Compute Refusal F1 and its diagnostic components.

    `responses` should contain only refusal-dimension cases. A mix of answerable
    controls and unanswerable cases is required for a meaningful precision.
    """
    tp = fp = fn = tn = 0
    plain_correct = 0

    for r in responses:
        refused = r.category is ResponseCategory.REFUSAL
        # Unparseable counts as a non-refusal (P1 Section 5.6.4).
        if r.is_answerable:
            if refused:
                fp += 1
            else:
                tn += 1
                # A correct answer on an answerable control is a plain-accuracy hit.
                if r.category is ResponseCategory.ANSWER and r.answer_correct:
                    plain_correct += 1
        else:
            if refused:
                tp += 1
            else:
                fn += 1
                # Plain accuracy counts a correct refusal as wrong, so an
                # unanswerable case can never score here. That is the point of
                # the baseline: it is blind to good abstention.

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = _f1(precision, recall)

    plain_accuracy = plain_correct / n if n else 0.0
    n_unanswerable = tp + fn
    negative_rejection_rate = tp / n_unanswerable if n_unanswerable else 0.0

    return RefusalResult(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n=n,
        plain_accuracy=plain_accuracy,
        negative_rejection_rate=negative_rejection_rate,
    )
