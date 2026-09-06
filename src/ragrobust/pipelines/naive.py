"""Naive RAG: retrieve then generate (P1 Section 5.6.1).

"For each query, the retriever returns the top-k passages from the corpus [...]
The retrieved passages are concatenated with a fixed prompt template and passed
to the generator, which produces the answer in a single forward pass."

This is the reference class. Its cost is exactly one generator call per instance,
which is the baseline the other two classes are measured against.
"""

from __future__ import annotations

from ..corpus import SandboxedCorpus
from ..providers.base import GenerationRequest, LLMProvider
from .base import (
    PipelineResult,
    RetrievalPolicy,
    check_context_budget,
    hit_output_cap,
)
from .prompts import build_prompt, extract_final_answer

PIPELINE_CLASS = "naive"


class NaivePipeline:
    def __init__(
        self,
        provider: LLMProvider,
        policy: RetrievalPolicy,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ):
        self.provider = provider
        self.policy = policy
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed

    async def run(self, corpus: SandboxedCorpus) -> PipelineResult:
        case = corpus.case
        passages = self.policy.initial(corpus)
        context = [{"passage_id": p.passage_id, "text": p.text} for p in passages]
        warnings = check_context_budget(context, case, self.policy.max_context_tokens)

        prompt = build_prompt(case.query, context, reasoning=False)
        resp = await self.provider.generate(
            GenerationRequest(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                thinking=False,
            )
        )

        if not resp.ok:
            return PipelineResult(
                case_id=case.case_id,
                pipeline_class=PIPELINE_CLASS,
                final_answer=None,
                raw_text="",
                retrieved_passage_ids=[p.passage_id for p in passages],
                generator_calls=1,
                warnings=warnings,
                error=resp.error,
            )

        return PipelineResult(
            case_id=case.case_id,
            pipeline_class=PIPELINE_CLASS,
            final_answer=extract_final_answer(resp.text),
            raw_text=resp.text,
            reasoning_trace=resp.reasoning_trace,
            retrieved_passage_ids=[p.passage_id for p in passages],
            generator_calls=1,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            truncated=hit_output_cap(resp),
            warnings=warnings,
        )
