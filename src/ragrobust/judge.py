"""Conflict Resolution Score judge (P1 Sections 5.4.3, 5.6.4, 5.9).

`metrics/conflict.py` aggregates rubric scores; this module produces them. The
split matters: aggregation is pure arithmetic testable without a model, while
judging needs a live provider, so keeping them apart lets the metric be verified
independently of the judge that feeds it.

The rubric levels below are P1's own wording, not a paraphrase. The judge is
asked to apply the specified test, and Chapter 6 can quote the prompt as the
operationalisation of the metric.

Three controls ride on this module.

The judge is Claude Haiku, an `anthropic`-family model, disjoint from the
`google` case generator and the `alibaba`/`zhipu` evaluated generators
(P1 5.9). `providers/factory.py` enforces that at construction.

Only the delimited final answer is judged, never the reasoning trace
(P1 5.6.4). The trace is retained for qualitative analysis but a judge that saw
it could reward a model for reasoning it never actually committed to.

Presentation order of the two conflicting passages is already randomised per
case by `dataset/conflict.py` (P1 5.9, the Zheng et al. position-bias
mitigation), and the judge sees the case's stored order, so the mitigation
carries through to the judge rather than stopping at the generator.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from typing import TYPE_CHECKING

from .schema import TestCase

if TYPE_CHECKING:  # pragma: no cover
    from .metrics.conflict import AgreementDiagnostics

RUBRIC_MIN, RUBRIC_MAX = 0, 4

# P1 Section 5.4.3, verbatim.
RUBRIC_LEVELS: dict[int, str] = {
    0: "The response ignores the conflict entirely or produces an arbitrary "
       "answer without acknowledging disagreement.",
    1: "Partial acknowledgement, such as a hedge or a mention that sources may "
       "differ, without identification of positions.",
    2: "Explicit recognition of the conflict and identification of at least two "
       "positions.",
    3: "Explicit recognition and a principled resolution, such as selecting the "
       "more recent source.",
    4: "Explicit recognition, identification of positions, and an articulated "
       "deferral to the user or a clear rationale for the chosen resolution.",
}

CRS_PROMPT = """\
You are scoring ONE RESPONSE written by a question-answering system. The system \
was given evidence that contradicts itself. Your job is to judge what THE \
RESPONSE did about that contradiction.

Question:
{query}

THE RESPONSE YOU ARE SCORING:
{answer}

Apply this rubric exactly. Do not invent intermediate levels.

{rubric}

CRITICAL. Score the RESPONSE, not the evidence. The evidence below always \
contains two conflicting positions -- that is what makes this a conflict case -- \
so the presence of two positions there tells you nothing about the response. A \
response that does not itself mention any disagreement scores 0, however plainly \
the evidence contradicts itself.

Worked anchors:
  - "INSUFFICIENT EVIDENCE"  -> 0. It states nothing about a disagreement.
  - "1889"                   -> 0. A bare answer, picking one side silently.
  - "The sources disagree: one says 1889, the other 1887."  -> 2. The response \
    itself names the conflict and both positions.

The evidence the system was given, for reference only:
{passages}

Answer in exactly two lines.

First line, beginning "QUOTE: " -- copy the words FROM THE RESPONSE that mention \
the disagreement, or write "QUOTE: none" if the response never mentions one.
Second line, the rubric score inside score tags.

Example:
QUOTE: none
<score>0</score>
"""


@dataclass(frozen=True)
class CRSJudgement:
    case_id: str
    score: int | None
    raw: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.score is not None and self.error is None


def format_rubric() -> str:
    return "\n".join(f"Score {k}: {v}" for k, v in sorted(RUBRIC_LEVELS.items()))


def build_crs_prompt(case: TestCase, final_answer: str) -> str:
    """Render the judge prompt for one conflict case.

    Passages are presented in the case's stored order, which was randomised at
    construction, so the true passage is not systematically first.
    """
    passages = "\n\n".join(
        f"[{i}] {p.text}" for i, p in enumerate(case.retrieved_passages, 1)
    )
    return CRS_PROMPT.format(
        rubric=format_rubric(),
        query=case.query,
        passages=passages,
        answer=final_answer.strip() or "(the system produced no answer)",
    )


def parse_crs_score(text: str) -> int | None:
    """Extract the rubric score from a judge response.

    Requires the delimited field. A bare digit found anywhere in prose is not
    accepted: the judge sometimes restates the rubric, and "Score 3: explicit
    recognition..." would otherwise be mined for a 3 regardless of its verdict.
    Returning None when the field is absent keeps an unparseable judgement out
    of the aggregate rather than silently scoring it zero, which would drag the
    mean down and flatter the binary baseline it is compared against.
    """
    match = re.search(r"<score>\s*([0-4])\s*</score>", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# --------------------------------------------------------------------------
# Validating the judge against human raters (P1 Objective 2, Section 5.4.3)
# --------------------------------------------------------------------------

# P1: inter-rater agreement between judge and human "is reported using Cohen's
# kappa on a sample of at least one hundred cases, in line with the
# LLM-as-judge protocol of Zheng et al."
DEFAULT_VALIDATION_N = 100


@dataclass(frozen=True)
class JudgeAgreement:
    """Agreement between the CRS judge and a human rater."""

    kappa: float
    quadratic_kappa: float
    n: int
    observed_agreement: float
    exact_matches: int
    within_one: int
    judge_distribution: dict[int, int]
    human_distribution: dict[int, int]
    meets_gate: bool
    # Gwet's AC1 and the prevalence diagnostics, computed here for the SAME
    # reason they are computed for the dataset validation sheets: so the two
    # kappa gates in this study are interpreted by one rule. Applying a
    # prevalence correction only where kappa fails, and never where it passes,
    # would be choosing the statistic by its answer.
    diagnostics: "AgreementDiagnostics | None" = None

    def as_dict(self) -> dict[str, object]:
        return {
            "cohens_kappa": round(self.kappa, 4),
            "quadratic_weighted_kappa": round(self.quadratic_kappa, 4),
            "n_cases": self.n,
            "observed_agreement": round(self.observed_agreement, 4),
            "exact_matches": self.exact_matches,
            "within_one_level": self.within_one,
            "judge_distribution": dict(sorted(self.judge_distribution.items())),
            "human_distribution": dict(sorted(self.human_distribution.items())),
            "meets_p1_gate": self.meets_gate,
            "p1_threshold": 0.60,
            "paradox_diagnostics": (
                None if self.diagnostics is None else self.diagnostics.as_dict()
            ),
        }


def select_judge_validation_sample(
    judged: list[dict],
    *,
    n: int = DEFAULT_VALIDATION_N,
    rng_seed: int = 20260721,
) -> list[dict]:
    """Draw the responses a human will re-score, blind to the judge's verdict.

    Each row must carry `case_id`, `config_id`, `final_answer` and `crs_score`.

    Sampling is RANDOM over judged responses, not stratified by the judge's
    score. Stratifying would over-represent whichever levels the judge uses
    rarely and turn kappa into a statement about a sample that does not resemble
    the run -- and on this data it would be impossible anyway, since only a
    handful of responses sit outside the two levels the judge actually uses.

    It IS balanced across pipeline classes, which does not touch the score
    distribution but does stop the judge being validated only against one
    class's answering style: a naive one-line answer and an agentic synthesis
    are different objects to score.
    """
    if not judged:
        raise ValueError("no judged responses to sample from")

    rng = random.Random(rng_seed)
    by_class: dict[str, list[dict]] = {}
    for row in judged:
        by_class.setdefault(row["config_id"].split("|")[0], []).append(row)

    per_class = max(1, n // max(1, len(by_class)))
    picked: list[dict] = []
    for cls in sorted(by_class):
        pool = sorted(by_class[cls], key=lambda r: (r["config_id"], r["case_id"]))
        rng.shuffle(pool)
        picked.extend(pool[:per_class])

    # Top up to exactly n from whatever remains, so "at least one hundred" is met
    # even when the classes divide unevenly.
    if len(picked) < n:
        chosen = {(r["config_id"], r["case_id"]) for r in picked}
        rest = [r for r in judged if (r["config_id"], r["case_id"]) not in chosen]
        rng.shuffle(rest)
        picked.extend(rest[: n - len(picked)])

    picked.sort(key=lambda r: (r["config_id"], r["case_id"]))
    return picked[:n]


def _weighted_kappa(a: list[int], b: list[int], *, levels: int = 5) -> float:
    """Quadratic-weighted kappa for an ordinal rubric.

    P1 specifies plain Cohen's kappa, which is what the gate is measured on, and
    this is reported ALONGSIDE it as a diagnostic. On a zero-to-four ordinal
    scale unweighted kappa treats a 0-versus-4 disagreement exactly like a
    0-versus-1 one, so a judge that is consistently close but rarely exact reads
    as though it were random. Seeing both makes a near-miss on the gate
    diagnosable rather than simply fatal.
    """
    n = len(a)
    if n == 0:
        raise ValueError("no items to compare")
    obs = [[0] * levels for _ in range(levels)]
    for x, y in zip(a, b):
        obs[x][y] += 1
    row = [sum(obs[i]) for i in range(levels)]
    col = [sum(obs[i][j] for i in range(levels)) for j in range(levels)]

    denom = (levels - 1) ** 2
    num_o = num_e = 0.0
    for i in range(levels):
        for j in range(levels):
            w = ((i - j) ** 2) / denom
            num_o += w * obs[i][j]
            num_e += w * row[i] * col[j] / n
    if num_e == 0:
        return 1.0
    return 1.0 - (num_o / num_e)


def compute_judge_agreement(
    pairs: list[tuple[int, int]], *, threshold: float = 0.60
) -> JudgeAgreement:
    """Cohen's kappa between judge and human over the same responses.

    `pairs` is (judge_score, human_score). Rows the human left blank must be
    dropped by the caller, never defaulted -- an unrated row is absent data, and
    scoring it as agreement would inflate the very number the gate rests on.
    """
    from collections import Counter  # noqa: PLC0415

    from .metrics.conflict import agreement_diagnostics, cohens_kappa  # noqa: PLC0415

    if not pairs:
        raise ValueError("no rated pairs supplied")
    judge = [p[0] for p in pairs]
    human = [p[1] for p in pairs]
    for s in judge + human:
        if not RUBRIC_MIN <= s <= RUBRIC_MAX:
            raise ValueError(f"rubric score {s} outside {RUBRIC_MIN}-{RUBRIC_MAX}")

    kappa = cohens_kappa(judge, human)
    exact = sum(1 for j, h in pairs if j == h)
    within = sum(1 for j, h in pairs if abs(j - h) <= 1)
    return JudgeAgreement(
        diagnostics=agreement_diagnostics(judge, human),
        kappa=kappa,
        quadratic_kappa=_weighted_kappa(judge, human),
        n=len(pairs),
        observed_agreement=exact / len(pairs),
        exact_matches=exact,
        within_one=within,
        judge_distribution=dict(Counter(judge)),
        human_distribution=dict(Counter(human)),
        meets_gate=kappa >= threshold,
    )
