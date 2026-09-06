"""Per-case sandboxed corpus.

This is the single most important experimental control in the study. P1 Section
5.6.5 requires that retrieval operates over the case's own perturbed passage set
plus a controlled distractor pool, so that an agentic pipeline issuing extra
retrievals cannot recover evidence the perturbation deliberately withheld.

Without this sandbox the agentic class would be retrieving from a richer corpus
than the naive class, and any observed difference would be attributable to
retrieval quality rather than to orchestration. That would make the central
research question unanswerable.

A useful side effect: retrieval runs over tens of passages per case rather than a
full Wikipedia index, so BM25 and dense retrieval are computationally trivial and
the entire cost of the experiment sits in generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from .schema import Passage, TestCase


class Encoder(Protocol):
    """Minimal dense encoder interface, satisfied by SentenceTransformer."""

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray: ...


def _tokenize(text: str) -> list[str]:
    return [t for t in text.lower().split() if t]


@dataclass
class RetrievalResult:
    passages: list[Passage]
    scores: list[float]
    retriever: str


class SandboxedCorpus:
    """The retrievable universe for exactly one test case.

    Contains the case's perturbed passage set and nothing else. Any retrieval,
    including a re-retrieval issued mid-loop by an agent, is confined here.
    """

    def __init__(self, case: TestCase, distractor_pool: list[Passage] | None = None):
        self.case = case
        # The distractor pool is a controlled extension, not an escape hatch. It
        # never contains answer-bearing passages, which is asserted below.
        pool = list(distractor_pool or [])
        if any(p.is_answer_bearing for p in pool):
            raise ValueError(
                f"{case.case_id}: distractor pool contains an answer-bearing passage, "
                "which would defeat the perturbation"
            )
        self.passages: list[Passage] = list(case.retrieved_passages) + pool
        self._bm25 = BM25Okapi([_tokenize(p.text) for p in self.passages])
        self._dense: np.ndarray | None = None

    def index_dense(self, encoder: Encoder) -> None:
        emb = np.asarray(encoder.encode([p.text for p in self.passages]))
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self._dense = emb / np.clip(norms, 1e-9, None)

    def _score(
        self, query: str, retriever: str, encoder: Encoder | None = None
    ) -> np.ndarray:
        if retriever == "bm25":
            scores = np.asarray(self._bm25.get_scores(_tokenize(query)))
        elif retriever == "dpr":
            if self._dense is None:
                if encoder is None:
                    raise ValueError("dense retrieval requires an encoder or a built index")
                self.index_dense(encoder)
            assert self._dense is not None
            q = np.asarray(encoder.encode([query])) if encoder is not None else None
            if q is None:
                raise ValueError("dense retrieval requires an encoder for the query")
            q = q[0] / max(float(np.linalg.norm(q[0])), 1e-9)
            scores = self._dense @ q
        else:
            raise ValueError(f"unknown retriever: {retriever}")
        return scores

    def retrieve(
        self, query: str, k: int, retriever: str, encoder: Encoder | None = None
    ) -> RetrievalResult:
        """Top-k over the whole sandbox. Used for agentic re-retrieval.

        P1 Section 5.6.5 fixes k across configurations so that cost and latency
        stay comparable, and confines the results to the sandbox so that
        re-retrieval "cannot recover evidence that the perturbation has
        deliberately withheld".
        """
        scores = self._score(query, retriever, encoder)

        # Stable ordering. Ties break by passage index so that a fixed seed and a
        # fixed corpus always produce byte-identical retrieval across runs.
        order = sorted(range(len(self.passages)), key=lambda i: (-scores[i], i))[:k]
        return RetrievalResult(
            passages=[self.passages[i] for i in order],
            scores=[float(scores[i]) for i in order],
            retriever=retriever,
        )

    def rank_case_passages(
        self, query: str, retriever: str, encoder: Encoder | None = None
    ) -> RetrievalResult:
        """Order the case's own constructed passage set without changing membership.

        This is the initial retrieval every pipeline class performs. P1 Section
        5.4.1 defines the noise dimension over "the proportion of irrelevant
        passages in the retrieved context", so the ratio is a property of what
        the generator actually sees. A top-k cut would deliver 0.625 for both
        r=0.75 and r=0.90, collapsing the top of the Noise Degradation Curve and
        making A(r) measure something other than r.

        Retrieval still genuinely runs, as P1 Section 5.6.1 specifies: the
        retriever fixes the ORDER of the passages, so BM25 and DPR remain a real
        experimental variable and positional effects are attributable to the
        retriever rather than to construction order.

        Scores come from the sandbox-wide index so that term statistics are
        identical between this call and any agentic re-retrieval. Ranking
        against a second, case-only index would make the two incomparable.
        """
        scores = self._score(query, retriever, encoder)
        n_case = len(self.case.retrieved_passages)
        order = sorted(range(n_case), key=lambda i: (-scores[i], i))
        return RetrievalResult(
            passages=[self.passages[i] for i in order],
            scores=[float(scores[i]) for i in order],
            retriever=retriever,
        )

    def assert_perturbation_holds(self) -> None:
        """Verify the sandbox still enforces what the perturbation intended.

        For an unanswerable refusal case no reachable passage may be answer
        bearing. Called before a run so that a corpus construction bug fails
        loudly at setup rather than silently inflating refusal scores.
        """
        if not self.case.is_answerable:
            if any(p.is_answer_bearing for p in self.passages):
                raise AssertionError(
                    f"{self.case.case_id}: unanswerable case can reach an "
                    "answer-bearing passage through the sandbox"
                )


def build_distractor_pool(
    candidates: list[Passage], answer: str, max_size: int = 20
) -> list[Passage]:
    """Filter candidate passages down to a safe distractor pool.

    Any candidate containing the answer string is dropped. This is a deliberately
    conservative lexical check. It over-rejects, which is the correct direction:
    a distractor that leaks the answer corrupts the case, whereas a discarded
    safe distractor costs nothing.
    """
    needle = answer.strip().lower()
    out: list[Passage] = []
    for p in candidates:
        if p.is_answer_bearing:
            continue
        if needle and needle in p.text.lower():
            continue
        out.append(Passage(passage_id=p.passage_id, text=p.text, is_injected=True))
        if len(out) >= max_size:
            break
    return out
