"""Agentic RAG: decompose, multi-retrieve, synthesise (P1 Section 5.6.3).

P1 specifies a ReAct loop (Yao et al., 2023) in which "the language model
alternat[es] between a thought step that decides on the next action, an action
step that performs a retrieval or other tool call, and an observation step that
interprets the result. The loop terminates either when the model produces a final
answer or when a maximum number of steps is reached."

Two properties P1 names explicitly are load-bearing and are implemented as such.

The agent can issue multiple retrievals with different sub-queries derived from
the original query. Every one of them goes through `RetrievalPolicy.re_retrieve`,
which is confined to the per-case sandboxed corpus. This is the study's central
control (P1 Section 5.6.5): re-retrieval "cannot recover evidence that the
perturbation has deliberately withheld", so any agentic gain is attributable to
orchestration rather than to a richer evidence pool.

The agent maintains explicit state that accumulates evidence across steps "and
can be inspected post hoc to diagnose failure modes". Every node writes a record
into `PipelineResult.steps`, so a failed case can be replayed from the run log
without re-running the model.

DEVIATION FROM P1, needs a sentence in Chapter 6: P1 names LangGraph as the
orchestration framework. The loop below is implemented directly instead. The
architecture P1 specifies is unchanged — the same ReAct alternation, the same
decompose/multi-retrieve/synthesise blocks, the same fixed step bound, the same
inspectable accumulating state — but the graph is explicit Python rather than a
framework dependency. The reasons are that the providers are custom async httpx
clients rather than LangChain models, so a framework adapter would add surface
area without adding behaviour, and that a replication package with fewer
dependencies is easier to re-run as the field moves.
"""

from __future__ import annotations

from ..corpus import SandboxedCorpus
from ..providers.base import GenerationRequest, GenerationResponse, LLMProvider
from .base import PipelineResult, RetrievalPolicy, check_context_budget
from .base import hit_output_cap
from .prompts import (
    build_decompose_prompt,
    build_react_step_prompt,
    build_synthesise_prompt,
    extract_final_answer,
    parse_next_query,
    parse_subqueries,
)

PIPELINE_CLASS = "agentic"


class AgenticPipeline:
    def __init__(
        self,
        provider: LLMProvider,
        policy: RetrievalPolicy,
        *,
        thinking: bool = False,
        cot_prompt: bool = False,
        max_steps: int = 5,
        max_subqueries: int = 3,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        seed: int | None = None,
    ):
        # Agentic generators are a subset of the reasoning generators
        # (P1 Section 5.6.3), so the same one-reasoning-form rule applies.
        if thinking and cot_prompt:
            raise ValueError(
                "an agentic configuration inherits one reasoning form, never both "
                "(P1 5.6.2, 5.6.3)"
            )
        self.provider = provider
        self.policy = policy
        self.thinking = thinking
        self.cot_prompt = cot_prompt
        # Fixed across all agentic configurations "to keep cost and latency
        # comparable across experiments" (P1 Section 5.6.3).
        self.max_steps = max_steps
        self.max_subqueries = max_subqueries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.seed = seed

    async def _call(self, prompt: str) -> GenerationResponse:
        return await self.provider.generate(
            GenerationRequest(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=self.seed,
                thinking=self.thinking,
            )
        )

    async def run(self, corpus: SandboxedCorpus) -> PipelineResult:
        case = corpus.case
        result = PipelineResult(
            case_id=case.case_id,
            pipeline_class=PIPELINE_CLASS,
            final_answer=None,
            raw_text="",
        )
        seen_ids: list[str] = []
        evidence: list[tuple[str, list[dict[str, str]]]] = []
        traces: list[str] = []

        def account(resp: GenerationResponse) -> None:
            result.generator_calls += 1
            result.prompt_tokens += resp.prompt_tokens
            result.completion_tokens += resp.completion_tokens
            # Any call in the loop that hit the cap marks the instance. The
            # agentic class is where this bites hardest: it makes several calls,
            # each with a full thinking budget, so it has several chances to run
            # out before committing an answer.
            if hit_output_cap(resp):
                result.truncated = True
            if resp.reasoning_trace:
                traces.append(resp.reasoning_trace)

        def observe(subquery: str) -> None:
            passages = self.policy.re_retrieve(corpus, subquery)
            for p in passages:
                if p.passage_id not in seen_ids:
                    seen_ids.append(p.passage_id)
            evidence.append(
                (subquery, [{"passage_id": p.passage_id, "text": p.text} for p in passages])
            )
            result.steps.append(
                {
                    "node": "retrieve",
                    "query": subquery,
                    "passage_ids": [p.passage_id for p in passages],
                }
            )

        def finish(answer_text: str, raw: str) -> PipelineResult:
            result.final_answer = answer_text
            result.raw_text = raw
            result.reasoning_trace = "\n---\n".join(traces) if traces else None
            result.retrieved_passage_ids = seen_ids
            flat = [
                {"passage_id": pid, "text": text}
                for pid, text in _unique_texts(evidence)
            ]
            result.warnings.extend(
                check_context_budget(flat, case, self.policy.max_context_tokens)
            )
            return result

        def fail(error: str) -> PipelineResult:
            result.error = error
            result.reasoning_trace = "\n---\n".join(traces) if traces else None
            result.retrieved_passage_ids = seen_ids
            return result

        # --- Decompose -----------------------------------------------------
        decompose = await self._call(
            build_decompose_prompt(case.query, max_subqueries=self.max_subqueries)
        )
        account(decompose)
        if not decompose.ok:
            return fail(f"decompose: {decompose.error}")

        subqueries = parse_subqueries(decompose.text, max_subqueries=self.max_subqueries)
        # A decomposition that yields nothing usable would silently turn this
        # into a single-retrieval pipeline and erase the difference the study
        # measures, so fall back to the original query and say so.
        if not subqueries:
            subqueries = [case.query]
            result.warnings.append(
                f"{case.case_id}: decomposition produced no usable sub-queries; "
                "fell back to the original query"
            )
        result.steps.append(
            {"node": "decompose", "output": decompose.text, "subqueries": list(subqueries)}
        )

        # --- Multi-retrieve and observe, bounded by max_steps --------------
        pending = list(subqueries)
        for step in range(self.max_steps):
            subquery = pending.pop(0) if pending else case.query
            observe(subquery)

            remaining = self.max_steps - step - 1
            resp = await self._call(
                build_react_step_prompt(case.query, evidence, remaining=remaining)
            )
            account(resp)
            if not resp.ok:
                return fail(f"react step {step}: {resp.error}")

            answer = extract_final_answer(resp.text)
            result.steps.append(
                {
                    "node": "react",
                    "step": step,
                    "remaining": remaining,
                    "output": resp.text,
                    "committed": answer is not None,
                }
            )
            # A committed answer ends the loop, exactly as P1 5.6.3 specifies.
            if answer is not None:
                return finish(answer, resp.text)

            requested = parse_next_query(resp.text)
            if requested:
                pending.insert(0, requested)
            elif not pending:
                # No answer and no further request: nothing more to explore.
                break

        # --- Synthesise ----------------------------------------------------
        # Reached when the step bound is exhausted without a commitment. The
        # agent is asked once more over everything it gathered, so a step-limit
        # case still produces a scoreable response rather than an empty one.
        final = await self._call(build_synthesise_prompt(case.query, evidence))
        account(final)
        if not final.ok:
            return fail(f"synthesise: {final.error}")

        result.steps.append({"node": "synthesise", "output": final.text})
        result.warnings.append(
            f"{case.case_id}: step bound of {self.max_steps} reached without a "
            "committed answer; forced synthesis"
        )
        return finish(extract_final_answer(final.text), final.text)


def _unique_texts(
    evidence: list[tuple[str, list[dict[str, str]]]]
) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for _, passages in evidence:
        for p in passages:
            seen.setdefault(p["passage_id"], p["text"])
    return list(seen.items())
