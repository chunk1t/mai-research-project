"""Shared pipeline contract.

The three classes of P1 Section 5.6 differ only in orchestration. Everything
else — the retrieval policy, the context budget, the record written out — is
held constant here so that a difference in results is attributable to the
orchestration and not to an incidental difference in plumbing (P1 Section 5.6.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..corpus import Encoder, SandboxedCorpus
from ..schema import TestCase


@dataclass(frozen=True)
class RetrievalPolicy:
    """How each pipeline class is allowed to retrieve.

    Encodes the decision recorded in HANDOFF Section 2 item 5. The initial
    retrieval ranks the case's constructed passage set, fixing order but never
    membership, so the noise ratio delivered to the generator is the ratio the
    case was built to carry (P1 Section 5.4.1). `top_k` applies only to agentic
    re-retrieval, which is the case P1 Section 5.6.5 is written about.
    """

    retriever: str
    top_k: int = 5
    max_context_tokens: int = 8192
    encoder: Encoder | None = None

    def initial(self, corpus: SandboxedCorpus) -> list[Any]:
        return corpus.rank_case_passages(
            corpus.case.query, self.retriever, self.encoder
        ).passages

    def re_retrieve(self, corpus: SandboxedCorpus, query: str) -> list[Any]:
        return corpus.retrieve(
            query, k=self.top_k, retriever=self.retriever, encoder=self.encoder
        ).passages


@dataclass
class PipelineResult:
    """One pipeline run over one case, in the form the scorer consumes.

    `raw_text` is kept alongside `final_answer` because P1 Section 5.6.4 scores
    only the delimited field but retains the surrounding response for
    qualitative analysis. `reasoning_trace` is stored separately again, so that
    no consumer can score it by accident.
    """

    case_id: str
    pipeline_class: str
    final_answer: str | None
    raw_text: str
    reasoning_trace: str | None = None
    retrieved_passage_ids: list[str] = field(default_factory=list)
    generator_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Agentic only: the accumulated state P1 Section 5.6.3 requires to be
    # "inspected post hoc to diagnose failure modes".
    steps: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    # A generation that hit the output cap before emitting the delimited answer
    # field. Recorded explicitly because it is NOT the same outcome as a wrong
    # answer or a refusal: a reasoning model can exhaust the budget inside its
    # thinking block, so the answer it had already reached is never written out.
    # Left indistinguishable from a null answer, it would be scored as a failure
    # and would concentrate in the reasoning and agentic arms -- reading as
    # "reasoning does not help" for a purely mechanical reason.
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    def as_record(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pipeline_class": self.pipeline_class,
            "final_answer": self.final_answer,
            "raw_text": self.raw_text,
            "reasoning_trace": self.reasoning_trace,
            "retrieved_passage_ids": list(self.retrieved_passage_ids),
            "generator_calls": self.generator_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "n_steps": len(self.steps),
            "steps": list(self.steps),
            "warnings": list(self.warnings),
            "truncated": self.truncated,
            "error": self.error,
        }


def hit_output_cap(*responses: Any) -> bool:
    """Whether any response stopped because it ran out of output budget.

    vLLM reports this as finish_reason "length"; the OpenAI-compatible field is
    carried through in `GenerationResponse.meta` by the provider.
    """
    return any(
        (getattr(r, "meta", None) or {}).get("finish_reason") == "length"
        for r in responses
        if r is not None
    )


def estimate_tokens(passages: list[dict[str, str]]) -> int:
    """Whitespace token estimate, matching `Passage.token_estimate`.

    The same estimator is used for the noise ratio during construction, so the
    context budget and the ratio are measured on one scale rather than two.
    """
    return sum(len(p["text"].split()) for p in passages)


def check_context_budget(
    passages: list[dict[str, str]], case: TestCase, limit: int
) -> list[str]:
    """Warn when a case cannot fit the max_context_tokens control.

    Deliberately warns rather than truncates. A noise case holds
    signal/(1 - ratio) tokens, so r=0.90 carries ten times the signal; dropping
    passages to fit would change the delivered noise ratio and put the instance
    at the wrong point on the Noise Degradation Curve, which is precisely the
    failure the ratio machinery exists to prevent. An over-budget instance must
    be visible in the run record and excluded or re-sized deliberately, never
    quietly reshaped.
    """
    used = estimate_tokens(passages)
    if used <= limit:
        return []
    return [
        f"CONTEXT OVERFLOW: {case.case_id} needs ~{used} tokens against a "
        f"{limit} limit (dimension={case.dimension.value}, "
        f"noise_ratio={case.noise_ratio}); not truncated, because truncation "
        "would change the delivered noise ratio"
    ]
