"""Benchmark data schema.

Implements Table 5.2 of the P1 report, one record per test case. The schema is
the contract between dataset construction, pipeline execution, and metric
computation, so it is validated strictly rather than duck-typed.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Sentinel used in the `answer` field when no answer is supported by the
# evidence. Chosen to be a string that cannot collide with a real short answer.
NO_ANSWER = "<<NO_ANSWER_SUPPORTED>>"

# The five noise ratios of P1 Section 5.4.2.
NOISE_RATIOS: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 0.90)


class Dimension(str, Enum):
    """Testbed label. One of the three evaluation dimensions."""

    REFUSAL = "refusal"
    CONFLICT = "conflict"
    NOISE = "noise"


class PerturbationType(str, Enum):
    """Which perturbation produced this case (P1 Table 5.2)."""

    ANSWER_PASSAGE_REMOVED = "answer_passage_removed"
    ANSWER_PASSAGE_RETAINED = "answer_passage_retained"  # answerable control
    CONTRADICTION_INJECTED = "contradiction_injected"
    DISTRACTORS_ADDED = "distractors_added"


class SeedSource(str, Enum):
    NATURAL_QUESTIONS = "natural_questions"
    TRIVIA_QA = "trivia_qa"


class Passage(BaseModel):
    """A single context passage inside a case's retrieved set."""

    passage_id: str
    text: str
    # True when this passage supports the ground-truth answer. Used by the
    # manual validation protocol and to assert perturbations behaved correctly.
    # Never exposed to a pipeline at inference time.
    is_answer_bearing: bool = False
    # True when the passage was injected as a distractor or contradiction.
    is_injected: bool = False

    def token_estimate(self) -> int:
        """Whitespace token count. Used for token-proportional noise ratios."""
        return len(self.text.split())


class TestCase(BaseModel):
    """One benchmark record. Mirrors P1 Table 5.2 field for field."""

    # Stops pytest trying to collect this as a test class.
    __test__ = False

    case_id: str
    query: str
    retrieved_passages: list[Passage]
    answer: str  # ground truth, or NO_ANSWER
    dimension: Dimension
    perturbation_type: PerturbationType
    noise_ratio: float | None = None  # noise cases only
    seed_source: SeedSource
    seed_id: str
    gen_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_invariants(self) -> TestCase:
        d, p = self.dimension, self.perturbation_type

        if not self.retrieved_passages:
            raise ValueError(f"{self.case_id}: retrieved_passages must not be empty")

        # noise_ratio is set if and only if this is a noise case.
        if d is Dimension.NOISE:
            if self.noise_ratio is None:
                raise ValueError(f"{self.case_id}: noise case requires noise_ratio")
            if self.noise_ratio not in NOISE_RATIOS:
                raise ValueError(
                    f"{self.case_id}: noise_ratio {self.noise_ratio} not in {NOISE_RATIOS}"
                )
        elif self.noise_ratio is not None:
            raise ValueError(f"{self.case_id}: noise_ratio only valid for noise cases")

        # Perturbation must match the dimension it claims to belong to.
        allowed = {
            Dimension.REFUSAL: {
                PerturbationType.ANSWER_PASSAGE_REMOVED,
                PerturbationType.ANSWER_PASSAGE_RETAINED,
            },
            Dimension.CONFLICT: {PerturbationType.CONTRADICTION_INJECTED},
            Dimension.NOISE: {PerturbationType.DISTRACTORS_ADDED},
        }[d]
        if p not in allowed:
            raise ValueError(f"{self.case_id}: perturbation {p} invalid for dimension {d}")

        # The central invariant of the refusal testbed. An unanswerable case
        # must carry the NO_ANSWER sentinel and must contain no answer-bearing
        # passage. An answerable control must do the opposite. Refusal F1
        # precision is undefined if these drift apart.
        has_answer_passage = any(x.is_answer_bearing for x in self.retrieved_passages)
        if p is PerturbationType.ANSWER_PASSAGE_REMOVED:
            if self.answer != NO_ANSWER:
                raise ValueError(f"{self.case_id}: unanswerable case must use NO_ANSWER")
            if has_answer_passage:
                raise ValueError(
                    f"{self.case_id}: unanswerable case retains an answer-bearing passage"
                )
        else:
            if self.answer == NO_ANSWER:
                raise ValueError(f"{self.case_id}: answerable case must have a real answer")
            if not has_answer_passage:
                raise ValueError(
                    f"{self.case_id}: answerable case has no answer-bearing passage"
                )

        return self

    @property
    def is_answerable(self) -> bool:
        """Answerable flag required by P1 Section 5.5.2 for Refusal F1."""
        return self.answer != NO_ANSWER

    def context_for_pipeline(self) -> list[dict[str, str]]:
        """Passages as a pipeline sees them, with provenance flags stripped."""
        return [{"passage_id": p.passage_id, "text": p.text} for p in self.retrieved_passages]

    def content_hash(self) -> str:
        """Stable hash over scoreable content. Detects post-freeze drift."""
        payload = json.dumps(
            {
                "query": self.query,
                "passages": [p.text for p in self.retrieved_passages],
                "answer": self.answer,
                "noise_ratio": self.noise_ratio,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Benchmark(BaseModel):
    """A full benchmark release with an integrity manifest."""

    version: str
    cases: list[TestCase]

    def counts_by_dimension(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cases:
            out[c.dimension.value] = out.get(c.dimension.value, 0) + 1
        return out

    def evaluation_instances(self) -> int:
        """Scoreable instances per configuration.

        Noise cases are stored once per ratio, so the stored case count already
        equals the instance count. P1 Section 5.6.5 expects roughly 2,900.
        """
        return len(self.cases)

    def manifest(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "n_cases": len(self.cases),
            "counts_by_dimension": self.counts_by_dimension(),
            "n_answerable": sum(1 for c in self.cases if c.is_answerable),
            "content_hash": hashlib.sha256(
                "".join(sorted(c.content_hash() for c in self.cases)).encode()
            ).hexdigest()[:16],
        }

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for c in self.cases:
                fh.write(c.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str, version: str) -> Benchmark:
        with open(path, encoding="utf-8") as fh:
            cases = [TestCase.model_validate_json(line) for line in fh if line.strip()]
        return cls(version=version, cases=cases)
