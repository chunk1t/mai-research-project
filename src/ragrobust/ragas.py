"""RAGAS secondary analysis (P1 Section 5.7).

P1 5.7: "A secondary analysis examines whether the RAGAS framework (Es et al.,
2024) detects the failures that the proposed metrics expose. For each case in a
held-out subset, the RAGAS scores for faithfulness, answer relevance, and
context relevance are computed and compared to the corresponding scores on the
proposed metrics."

The comparison answers two questions, both stated in P1: whether RAGAS scores
CORRELATE with the purpose-built metrics, and whether there are systematic
cases where RAGAS MISSES failures the purpose-built metrics catch. The second
is the one that carries the argument. P1 4.6.2 already predicts the answer --
"a model that fabricates an answer from insufficient context can still receive
a high faithfulness score", and "a model that picks one source from a
contradictory pair can receive high faithfulness and relevance scores even
though it fails to acknowledge the disagreement" -- so this module exists to
test those two predictions rather than to illustrate them.

DEVIATION FROM P1 5.8, and it is deliberate. P1 lists "the RAGAS library" in
the implementation stack. The three metrics are reimplemented here against the
definitions in Es et al. (2024) Section 3 instead, for three reasons that all
bear on validity:

  1. Determinism. P1 5.6.5 requires fixed seeds and temperature 0 on every
     model call, and 5.8 requires the run to be resumable from cache. Every
     call in this project goes through `providers/` for exactly that reason.
     The library drives its own LLM client and would put the study's most
     expensive control outside the layer that enforces it.
  2. Anti-circularity. `providers/factory.py` audits the model family behind
     every role (P1 5.9). A judge constructed inside a third-party library is
     not auditable by that check.
  3. Cost. Cached, an unchanged rerun issues no calls at all.

The metric DEFINITIONS are unchanged, and Chapter 6 should state the deviation
in these terms: same metrics, same prompting scheme, this project's provider
layer underneath.

Es et al. (2024), Section 3:

  Faithfulness      decompose the answer into atomic statements, verify each
                    against the retrieved context, score = supported / total.
  Answer relevance  generate candidate questions from the answer, score = mean
                    cosine similarity between them and the original question.
  Context relevance extract the context sentences needed to answer the
                    question, score = extracted / total sentences.

One property of the refusal testbed forces a decision the library does not
have to make. A response of "INSUFFICIENT EVIDENCE" contains no factual
statements, so faithfulness is 0/0 -- undefined, not zero. It is reported as
None with a reason rather than coerced, because coercing it either way
manufactures the study's own conclusion: coercing to 0.0 would make RAGAS look
like it detects refusal, and coercing to 1.0 would make it look like it rewards
refusal. That RAGAS is UNDEFINED on abstention is itself the P1 4.6.2 finding,
and it can only be reported if it is preserved.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Prompts. Phrased after Es et al. (2024) Section 3 and Appendix A.
# --------------------------------------------------------------------------

STATEMENT_PROMPT = """\
Break the answer below into atomic factual statements.

An atomic statement asserts exactly one fact and can be verified on its own. \
Resolve pronouns and references so each statement stands alone.

If the answer asserts no verifiable fact -- for example it declines to answer, \
says the evidence is insufficient, or is empty -- write exactly NONE and \
nothing else.

Question: {query}

Answer: {answer}

Write one statement per line, each beginning with "- ". Write nothing else.
"""

VERDICT_PROMPT = """\
Decide whether each statement can be inferred from the context.

Answer 1 if the context supports the statement, 0 if it does not. Judge only \
against the context: a statement that is true in the world but absent from the \
context is 0.

Context:
{context}

Statements:
{statements}

For each statement, on its own line, write the statement number, a colon, and \
the verdict. Example:
1: 1
2: 0
Write nothing else.
"""

QUESTION_GEN_PROMPT = """\
Write {n} different questions that the answer below would be a correct and \
complete reply to.

Base the questions ONLY on the answer. Do not use outside knowledge and do not \
try to guess the original question.

Answer: {answer}

Write one question per line. Write nothing else.
"""

CONTEXT_RELEVANCE_PROMPT = """\
Select the sentences from the context that are needed to answer the question.

Choose only sentences that contribute to the answer. If no sentence \
contributes, write exactly NONE.

Question: {query}

Context sentences:
{sentences}

Write the numbers of the selected sentences, separated by commas, and nothing \
else. Example: 2, 5, 6
"""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass
class RagasScores:
    """The three RAGAS metrics for one (case, configuration) instance.

    Each score is None when the metric is not defined for that instance, with
    the reason recorded in `notes`. None never means zero.
    """

    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_relevance: float | None = None
    n_statements: int = 0
    n_supported: int = 0
    n_context_sentences: int = 0
    n_relevant_sentences: int = 0
    n_generated_questions: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        return {
            "faithfulness": r(self.faithfulness),
            "answer_relevance": r(self.answer_relevance),
            "context_relevance": r(self.context_relevance),
            "n_statements": self.n_statements,
            "n_supported": self.n_supported,
            "n_context_sentences": self.n_context_sentences,
            "n_relevant_sentences": self.n_relevant_sentences,
            "n_generated_questions": self.n_generated_questions,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# Parsing. Pure, so every shape a judge can return is testable without a model.
# --------------------------------------------------------------------------

_NO_STATEMENTS = re.compile(r"^\s*none\s*\.?\s*$", re.IGNORECASE)


def parse_statements(text: str) -> list[str]:
    """Read the atomic statements out of the decomposition step.

    An empty list means "the answer asserts nothing verifiable", which is the
    correct reading of a refusal and is what makes faithfulness undefined
    rather than zero downstream.
    """
    if _NO_STATEMENTS.match(text.strip()):
        return []
    out: list[str] = []
    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^\s*(?:\d+\s*[.)]|[-*•])\s*", "", cleaned).strip()
        if not cleaned or _NO_STATEMENTS.match(cleaned):
            continue
        out.append(cleaned)
    return out


def parse_verdicts(text: str, n_statements: int) -> list[bool] | None:
    """Read one 0/1 verdict per statement.

    Returns None unless every statement got a verdict. A partial verdict list
    would silently change the denominator of faithfulness, turning a parsing
    failure into a score -- and on this data that score would land in the
    coverage-gap table as evidence.
    """
    verdicts: dict[int, bool] = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*([01])\b", line)
        if m:
            verdicts[int(m.group(1))] = m.group(2) == "1"
    if len(verdicts) != n_statements:
        return None
    if sorted(verdicts) != list(range(1, n_statements + 1)):
        return None
    return [verdicts[i] for i in range(1, n_statements + 1)]


def parse_generated_questions(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+\s*[.)]|[-*•])\s*", "", line.strip()).strip()
        cleaned = cleaned.strip('"').strip("'").strip()
        if cleaned:
            out.append(cleaned)
    return out


def parse_selected_sentences(text: str, n_sentences: int) -> list[int]:
    """Read the 1-based sentence numbers the judge selected.

    Numbers outside the range are dropped rather than raising: a judge that
    hallucinates sentence 12 of a 6-sentence context has made an error about
    that sentence, not about the other five, and discarding the whole instance
    would bias the sample towards short contexts.
    """
    if _NO_STATEMENTS.match(text.strip()):
        return []
    found = [int(x) for x in re.findall(r"\d+", text)]
    return sorted({i for i in found if 1 <= i <= n_sentences})


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences for the context-relevance denominator.

    Deliberately simple and deterministic. The metric is a RATIO of selected to
    total sentences, so what matters is that the same splitter produces the
    numerator and the denominator, not that it matches a linguist's judgement.
    A model-based splitter would make context relevance depend on a second
    model, and NQ passages here retain raw Wikipedia markup, which defeats most
    of them anyway.
    """
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text.strip()) if p.strip()]
    return parts


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def faithfulness_score(verdicts: list[bool]) -> float | None:
    """|V| / |S|, per Es et al. (2024).

    None for an empty statement list: 0/0 is undefined, and the whole point of
    the refusal comparison is that RAGAS has nothing to say there.
    """
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v) / len(verdicts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have the same dimension")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def answer_relevance_score(
    query_embedding: list[float], question_embeddings: list[list[float]]
) -> float | None:
    """Mean cosine similarity between the original question and the generated ones.

    None when no questions could be generated, which happens exactly when the
    answer carries no content -- a refusal, or an empty final-answer field.
    Same reasoning as faithfulness: absence is not a score of zero.
    """
    if not question_embeddings:
        return None
    sims = [cosine_similarity(query_embedding, q) for q in question_embeddings]
    return sum(sims) / len(sims)


def context_relevance_score(n_relevant: int, n_total: int) -> float | None:
    if n_total <= 0:
        return None
    return n_relevant / n_total


def build_statement_prompt(query: str, answer: str) -> str:
    return STATEMENT_PROMPT.format(query=query, answer=answer.strip() or "(empty)")


def build_verdict_prompt(context: str, statements: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
    return VERDICT_PROMPT.format(context=context, statements=numbered)


def build_question_prompt(answer: str, n: int = 3) -> str:
    return QUESTION_GEN_PROMPT.format(answer=answer.strip() or "(empty)", n=n)


def build_context_relevance_prompt(query: str, sentences: list[str]) -> str:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    return CONTEXT_RELEVANCE_PROMPT.format(query=query, sentences=numbered)


# --------------------------------------------------------------------------
# The comparison P1 5.7 actually asks for
# --------------------------------------------------------------------------


def _average_ranks(values: list[float]) -> list[float]:
    """Ranks with ties averaged, which is what makes Spearman correct on ties.

    The purpose-built metrics here are heavily tied by construction -- answer
    correctness is binary and CRS is 0 on most conflict cases -- so ordinal
    ranking without tie handling would fabricate an ordering that the data does
    not contain and inflate the correlation.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman's rank correlation, ties averaged.

    Rank-based rather than Pearson because one side of every comparison here is
    ordinal or binary: RAGAS scores are continuous on [0, 1] but answer
    correctness is a 0/1 indicator and CRS is a 0-4 rubric, and Pearson would
    assume interval spacing that neither has.

    None when either side is constant -- correlation with a variable that does
    not vary is undefined, and returning 0.0 would read in a results table as
    "measured, and unrelated" rather than "not measurable".
    """
    if len(xs) != len(ys):
        raise ValueError("both series must have the same length")
    if len(xs) < 3:
        return None
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0.0 or vy == 0.0:
        return None
    return cov / math.sqrt(vx * vy)


@dataclass
class CoverageGap:
    """How often RAGAS calls an instance fine that the purpose-built metric fails.

    This is the second of P1 5.7's two questions, and the one that carries the
    argument: "whether there are systematic cases where RAGAS misses failures
    that the proposed metrics catch, which would identify a coverage gap in
    existing reference-free tools".

    `n_undefined_on_failures` is reported separately from `n_missed` and is not
    folded into it. A metric that is undefined on a failure has not missed it in
    the same sense as one that scored it 0.9; conflating the two would overstate
    the gap. Both are coverage gaps, and they are different kinds.
    """

    metric: str
    dimension: str
    threshold: float
    n_instances: int
    n_failures: int
    n_missed: int              # our metric says failed, RAGAS scores >= threshold
    n_undefined_on_failures: int
    n_successes_passing: int   # our metric says fine, RAGAS scores >= threshold
    n_successes_scored: int
    mean_on_failures: float | None
    mean_on_successes: float | None

    @property
    def miss_rate(self) -> float | None:
        scored = self.n_failures - self.n_undefined_on_failures
        if scored <= 0:
            return None
        return self.n_missed / scored

    @property
    def pass_rate_on_successes(self) -> float | None:
        if self.n_successes_scored <= 0:
            return None
        return self.n_successes_passing / self.n_successes_scored

    @property
    def discriminative(self) -> bool:
        """Whether the miss rate carries any information at all.

        A metric that scores EVERYTHING below the threshold has a miss rate of
        0.0 while detecting nothing: it did not pass the failures, but it did
        not pass the successes either. Read without this guard, RAGAS context
        relevance scores 0.0 misses on all three dimensions of this benchmark
        and looks like the one metric that catches everything, when in fact its
        mean is 0.03 and it fails every instance indiscriminately.

        The test is that the metric's verdict differs between failures and
        successes by more than a rounding error in at least one direction.
        """
        pass_fail = self.miss_rate
        pass_ok = self.pass_rate_on_successes
        if pass_fail is None or pass_ok is None:
            return False
        return abs(pass_ok - pass_fail) >= 0.05

    def as_dict(self) -> dict[str, object]:
        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        return {
            "ragas_metric": self.metric,
            "dimension": self.dimension,
            "pass_threshold": self.threshold,
            "n_instances": self.n_instances,
            "n_failures_by_purpose_built_metric": self.n_failures,
            "n_failures_ragas_scores_as_passing": self.n_missed,
            "n_failures_ragas_cannot_score": self.n_undefined_on_failures,
            "miss_rate": r(self.miss_rate),
            # Without these two, a miss rate of 0.0 cannot be told apart from a
            # metric that rejects every instance it is shown.
            "n_successes_ragas_scores_as_passing": self.n_successes_passing,
            "pass_rate_on_successes": r(self.pass_rate_on_successes),
            "discriminative": self.discriminative,
            "mean_ragas_on_failures": r(self.mean_on_failures),
            "mean_ragas_on_successes": r(self.mean_on_successes),
            # The separation is what tells you whether RAGAS carries any signal
            # about this failure mode at all. Near zero means it does not.
            "separation": (
                None
                if self.mean_on_failures is None or self.mean_on_successes is None
                else round(self.mean_on_successes - self.mean_on_failures, 4)
            ),
        }


def coverage_gap(
    scores: list[float | None],
    failed: list[bool],
    *,
    metric: str,
    dimension: str,
    threshold: float,
) -> CoverageGap:
    """Compare one RAGAS metric against one purpose-built verdict.

    `failed[i]` is the purpose-built metric's verdict on instance i: True when
    the pipeline got it wrong. `scores[i]` is the RAGAS score, or None where
    RAGAS is undefined for that instance.
    """
    if len(scores) != len(failed):
        raise ValueError("scores and verdicts must be aligned")

    fail_scores = [s for s, f in zip(scores, failed) if f and s is not None]
    ok_scores = [s for s, f in zip(scores, failed) if not f and s is not None]
    n_failures = sum(1 for f in failed if f)
    undefined = sum(1 for s, f in zip(scores, failed) if f and s is None)

    return CoverageGap(
        metric=metric,
        dimension=dimension,
        threshold=threshold,
        n_instances=len(scores),
        n_failures=n_failures,
        n_missed=sum(1 for s in fail_scores if s >= threshold),
        n_undefined_on_failures=undefined,
        n_successes_passing=sum(1 for s in ok_scores if s >= threshold),
        n_successes_scored=len(ok_scores),
        mean_on_failures=(sum(fail_scores) / len(fail_scores)) if fail_scores else None,
        mean_on_successes=(sum(ok_scores) / len(ok_scores)) if ok_scores else None,
    )
