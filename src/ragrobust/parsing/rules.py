"""Rule-based half of the response protocol (P1 Section 5.6.4).

Two independent jobs live here, both deliberately pure so they can be tested
against hand-worked values with no model present.

Classification. The delimited final-answer field is sorted into a substantive
answer, an explicit refusal, or unparseable. P1 specifies "a rule-based parser
backed by a language-model fallback", so these rules also report how confident
they are; an ambiguous field is escalated rather than guessed at.

Correctness. Short-answer matching for the answerable cases, using the
normalisation convention of the SQuAD and TriviaQA evaluation scripts so the
numbers are comparable with published QA results rather than idiosyncratic.

The critical property, and the one P1 5.6.4 is most explicit about: only the
final-answer field is ever examined. "A trace that expresses uncertainty followed
by a committed answer is classified as a substantive answer, not a refusal."
Nothing in this module accepts a reasoning trace as an argument, so that cannot
be violated by accident.
"""

from __future__ import annotations

import re
import string
from collections import Counter

from ..matching import contains_on_word_boundary
from dataclasses import dataclass

from ..metrics.scored import ResponseCategory
from ..pipelines.prompts import REFUSAL_SENTINEL

# The curated abstention pattern set P1 Section 5.6.4 calls for. Every entry
# asserts that the evidence does not support an answer. Patterns are matched
# against the final-answer field only.
#
# Deliberately excluded: bare hedges such as "possibly" or "it seems". Those
# accompany a committed answer more often than they replace one, and treating
# them as refusals would inflate refusal recall on answerable controls, which is
# precisely the over-caution Refusal F1 exists to catch (P1 5.4.3).
REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        re.escape(REFUSAL_SENTINEL),
        r"\binsufficient\s+(?:evidence|information|context|data)\b",
        r"\bnot\s+enough\s+(?:evidence|information|context|data)\b",
        r"\b(?:evidence|information|context)\s+is\s+insufficient\b",
        r"\bcannot\s+be\s+(?:determined|answered|established)\b",
        r"\b(?:cannot|can't|could\s+not|unable\s+to)\s+(?:answer|determine|tell|say)\b",
        r"\b(?:passages?|context|documents?|sources?)\s+(?:do(?:es)?\s+not|don't|doesn't)"
        r"\s+(?:contain|provide|mention|state|specify|say|support)\b",
        r"\bno\s+(?:information|evidence|mention|answer)\b",
        r"\bnot\s+(?:stated|mentioned|specified|provided|given)\b",
        r"\b(?:i\s+)?(?:do\s+not|don't)\s+know\b",
        r"^\s*(?:unknown|n/?a|none)\s*$",
    )
)

# A refusal assertion inside a long field usually means the model refused about
# one part and answered another. That mixture is what the language-model
# fallback exists for, so it is escalated rather than resolved by keyword.
SHORT_FIELD_TOKENS = 15

_PUNCT = str.maketrans("", "", string.punctuation)
_ARTICLES = re.compile(r"\b(?:a|an|the)\b")


def normalise_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace.

    The SQuAD / TriviaQA convention, adopted so that answer accuracy here is
    comparable with published QA numbers instead of being a bespoke measure.
    """
    lowered = text.lower()
    depunctuated = lowered.translate(_PUNCT)
    without_articles = _ARTICLES.sub(" ", depunctuated)
    return " ".join(without_articles.split())


def exact_match(prediction: str, gold: str) -> bool:
    return normalise_answer(prediction) == normalise_answer(gold)


def contains_answer(prediction: str, gold: str) -> bool:
    """Whether the normalised gold answer appears in the normalised prediction.

    The convention short-answer QA benchmarks use, including RGB, which P1
    Section 5.4.3 names as a baseline. It forgives "It was completed in 1889"
    against a gold of "1889", which exact match would reject even though the
    model is plainly correct.

    Only ever applied to a field already classified as a substantive answer, so
    a negated form such as "not 1889" cannot slip through as correct: a response
    that declines to commit is classified as a refusal before it reaches here.
    """
    gold_norm = normalise_answer(gold)
    if not gold_norm:
        return False
    # Word-boundary containment, not bare substring. A bare substring grades a
    # prediction of "King George III" as correct against a gold of
    # "King George I", inflating accuracy in every configuration. The
    # convention's intent -- forgiving "it was completed in 1889" against
    # "1889" -- is unaffected, because that match sits on word boundaries.
    return contains_on_word_boundary(gold_norm, normalise_answer(prediction))


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 on normalised text, reported alongside exact match."""
    pred_tokens = normalise_answer(prediction).split()
    gold_tokens = normalise_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(overlap.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class AnswerScore:
    """Correctness of a committed answer, with the components kept visible."""

    correct: bool
    exact_match: bool
    contains: bool
    token_f1: float


def score_answer(prediction: str, gold: str) -> AnswerScore:
    """Grade a committed answer against the ground truth.

    `correct` is exact match OR containment, the short-answer convention. Both
    components are retained so Chapter 7 can report the stricter exact-match
    figure if a reviewer prefers it, without re-running anything.
    """
    em = exact_match(prediction, gold)
    contains = contains_answer(prediction, gold)
    return AnswerScore(
        correct=em or contains,
        exact_match=em,
        contains=contains,
        token_f1=token_f1(prediction, gold),
    )


def looks_like_refusal(field: str) -> bool:
    return any(p.search(field) for p in REFUSAL_PATTERNS)


@dataclass(frozen=True)
class RuleVerdict:
    """A rule-based classification and whether it should be trusted alone."""

    category: ResponseCategory
    confident: bool
    reason: str


def rule_classify(final_answer: str | None) -> RuleVerdict:
    """Classify the final-answer field by rule.

    `confident=False` marks the cases P1 Section 5.6.4 reserves for the
    language-model fallback. Guessing at them would put a systematic parsing
    error inside the headline metric, where the kappa validation is the only
    thing that would ever reveal it.
    """
    if final_answer is None:
        return RuleVerdict(
            ResponseCategory.UNPARSEABLE, True, "no delimited final-answer field"
        )

    field = final_answer.strip()
    if not field:
        return RuleVerdict(
            ResponseCategory.UNPARSEABLE, True, "final-answer field was empty"
        )

    refusal = looks_like_refusal(field)
    short = len(field.split()) <= SHORT_FIELD_TOKENS

    if refusal and short:
        return RuleVerdict(
            ResponseCategory.REFUSAL, True, "abstention pattern in a short field"
        )
    if refusal:
        # Long field asserting insufficiency and possibly also answering.
        return RuleVerdict(
            ResponseCategory.REFUSAL, False, "abstention pattern in a long field"
        )
    return RuleVerdict(ResponseCategory.ANSWER, True, "no abstention pattern")


# --------------------------------------------------------------------------
# Recovery of responses that answered but did not use the delimiter
# --------------------------------------------------------------------------

_SENTINEL = "INSUFFICIENT EVIDENCE"

# "<Tina Turner></final_answer>": the model substituted its answer into the
# OPENING tag position rather than between the tags. The closing tag is present
# and the pseudo-tag holds the whole answer, so the intent is unambiguous.
_MALFORMED_OPEN_RE = re.compile(r"^<([^<>]{1,120})></final_answer>\s*$", re.IGNORECASE)

# A trailing sentinel, with or without the angle brackets some models add:
# "... Therefore, the answer is:\n\n<INSUFFICIENT EVIDENCE>"
_TRAILING_SENTINEL_RE = re.compile(
    r"(?:^|\n|:)\s*<?\s*" + re.escape(_SENTINEL) + r"\s*>?\s*[.]?\s*$", re.IGNORECASE
)


def recover_unparsed_answer(raw_text: str) -> tuple[str | None, str | None]:
    """Recover a committed answer from a response that omitted the delimiter.

    Returns (answer, method), or (None, None) when nothing can be recovered
    safely.

    P1 Section 5.6.4 scores the delimited field, and the pipelines request it.
    GLM-4-9B largely ignores that instruction: on the live run it returned a
    bare "INSUFFICIENT EVIDENCE" for 49.3% of standard-arm responses and 29.7%
    of the chain-of-thought arm, against 0.4% and 2.1% for Qwen. Treating those
    as unparseable would score a *correct abstention* as a failure, and because
    the behaviour is model-specific it would report that GLM never abstains when
    the transcripts show it abstaining hundreds of times -- inverting the
    finding Refusal F1 exists to produce.

    Applied at SCORING time only, never inside the pipelines. `extract_final_answer`
    also decides when the agentic ReAct loop stops (agentic.py), so relaxing it
    mid-run would make later instances terminate on inputs earlier ones kept
    working on, breaking consistency within a single experiment.

    Deliberately narrow. Only three shapes are recovered, each unambiguous:

      * the response is exactly the refusal sentinel;
      * the response ENDS with the sentinel, optionally angle-bracketed, so the
        preceding text is the model's justification for abstaining;
      * a malformed opening tag of the form "<answer></final_answer>".

    Free prose with no delimiter and no sentinel is NOT recovered: choosing
    which clause was the answer would be the scorer inventing a commitment the
    model never marked, and a truncated reasoning trace looks exactly like it.
    """
    text = (raw_text or "").strip()
    if not text:
        return None, None

    if text.upper().strip(" .<>") == _SENTINEL:
        return _SENTINEL, "bare_sentinel"

    if _TRAILING_SENTINEL_RE.search(text):
        return _SENTINEL, "trailing_sentinel"

    m = _MALFORMED_OPEN_RE.match(text)
    if m:
        inner = m.group(1).strip()
        # Guard against recovering the literal tag name as an answer.
        if inner and inner.lower() != "final_answer":
            return inner, "malformed_open_tag"

    return None, None
