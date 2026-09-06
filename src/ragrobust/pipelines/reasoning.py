"""Reasoning-augmented RAG: retrieve, reason, generate (P1 Section 5.6.2).

P1 specifies two forms of reasoning and this class implements both behind one
interface, selected by configuration rather than by code:

  the reasoning-trained generator, "which produces extended reasoning traces by
  default at inference time" — selected through the provider's thinking flag;

  chain-of-thought prompting "in the style of Wei et al. (2022), applied to a
  standard generator" — selected through the reasoning block in the prompt.

A configuration uses exactly one of the two, never both: stacking a CoT prompt on
a thinking-mode model would confound the two reasoning forms P1 sets out to
compare.

Retrieval is identical to the naive class, which is what makes the comparison
clean: "the reasoning-augmented configuration uses the same retrieval step as the
naive configuration". The cost is still one generator call per instance, because
P1 Section 5.6.2 extracts the answer from a single response "by a post-processing
step" rather than issuing a second call. The provider layer performs that split,
so `raw_text` is answer-only and the trace is carried separately (P1 5.6.4).
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

PIPELINE_CLASS = "reasoning"


class ReasoningPipeline:
    def __init__(
        self,
        provider: LLMProvider,
        policy: RetrievalPolicy,
        *,
        thinking: bool = False,
        cot_prompt: bool = False,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        seed: int | None = None,
    ):
        if thinking and cot_prompt:
            raise ValueError(
                "a reasoning configuration uses either the reasoning-trained form "
                "or the chain-of-thought-prompted form, never both (P1 5.6.2)"
            )
        if not thinking and not cot_prompt:
            raise ValueError(
                "a reasoning configuration must select one reasoning form (P1 5.6.2)"
            )
        self.provider = provider
        self.policy = policy
        self.thinking = thinking
        self.cot_prompt = cot_prompt
        # Reasoning responses carry a trace as well as an answer, so the output
        # allowance is larger than the naive class. This is a budget, not a
        # decoding difference: temperature and seed stay constant (P1 5.6.5).
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed

    async def run(self, corpus: SandboxedCorpus) -> PipelineResult:
        case = corpus.case
        passages = self.policy.initial(corpus)
        context = [{"passage_id": p.passage_id, "text": p.text} for p in passages]
        warnings = check_context_budget(context, case, self.policy.max_context_tokens)

        prompt = build_prompt(case.query, context, reasoning=self.cot_prompt)
        resp = await self.provider.generate(
            GenerationRequest(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                thinking=self.thinking,
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
