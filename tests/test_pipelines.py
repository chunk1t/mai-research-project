"""Pipeline tests, asserted against hand-worked expectations.

The generator is a scripted fake throughout, so every assertion is about what
the pipeline does with a known response rather than about what a model happens
to say. Call counts, retrieval membership, and step records are all checked,
because those are the properties that make the three classes comparable.
"""

from __future__ import annotations

import pytest

from ragrobust.corpus import SandboxedCorpus
from ragrobust.dataset.noise import build_noise_testbed
from ragrobust.dataset.seeds import Seed
from ragrobust.pipelines import (
    AgenticPipeline,
    NaivePipeline,
    ReasoningPipeline,
    RetrievalPolicy,
)
from ragrobust.pipelines.base import check_context_budget
from ragrobust.pipelines.prompts import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    REFUSAL_SENTINEL,
    TASK_PREFIX,
    build_prompt,
    extract_final_answer,
    format_evidence,
    parse_next_query,
    parse_subqueries,
)
from ragrobust.providers.base import GenerationRequest, GenerationResponse, LLMProvider, RateLimiter
from ragrobust.schema import (
    NO_ANSWER,
    Dimension,
    Passage,
    PerturbationType,
    SeedSource,
    TestCase,
)


class ScriptedProvider(LLMProvider):
    """Returns queued responses in order and records every request it received."""

    def __init__(self, script: list[str], *, family: str = "alibaba", fail_at: int | None = None):
        super().__init__(
            name="fake", model="fake-model", family=family, limiter=RateLimiter(4)
        )
        self.script = list(script)
        self.requests: list[GenerationRequest] = []
        self.fail_at = fail_at

    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        n = len(self.requests)
        self.requests.append(req)
        if self.fail_at is not None and n == self.fail_at:
            raise RuntimeError("upstream exploded")
        text = self.script[n] if n < len(self.script) else self.script[-1]
        return GenerationResponse(
            text=text,
            model=self.model,
            family=self.family,
            reasoning_trace="trace" if req.thinking else None,
            prompt_tokens=10,
            completion_tokens=5,
        )


def answered(value: str) -> str:
    return f"Some prose here.\n{ANSWER_OPEN}{value}{ANSWER_CLOSE}"


def make_case(n_passages: int = 4, answerable: bool = True) -> TestCase:
    passages = [
        Passage(
            passage_id=f"p{i}",
            text=f"Passage {i} about the tower and its construction history number {i}.",
            is_answer_bearing=(i == 0 and answerable),
        )
        for i in range(n_passages)
    ]
    if not answerable:
        return TestCase(
            case_id="refusal-unans-x",
            query="In what year was the tower completed?",
            retrieved_passages=passages,
            answer=NO_ANSWER,
            dimension=Dimension.REFUSAL,
            perturbation_type=PerturbationType.ANSWER_PASSAGE_REMOVED,
            seed_source=SeedSource.TRIVIA_QA,
            seed_id="x",
        )
    return TestCase(
        case_id="refusal-ctrl-x",
        query="In what year was the tower completed?",
        retrieved_passages=passages,
        answer="1889",
        dimension=Dimension.REFUSAL,
        perturbation_type=PerturbationType.ANSWER_PASSAGE_RETAINED,
        seed_source=SeedSource.TRIVIA_QA,
        seed_id="x",
    )


def corpus_for(case: TestCase, pool: list[Passage] | None = None) -> SandboxedCorpus:
    return SandboxedCorpus(case, distractor_pool=pool or [])


def policy(**kw) -> RetrievalPolicy:
    return RetrievalPolicy(retriever="bm25", **kw)


# --------------------------------------------------------------------------
# Prompt contract (P1 Sections 5.6.4, 5.6.5)
# --------------------------------------------------------------------------


def test_task_framing_is_identical_across_all_three_classes():
    # P1 5.6.5 holds "the prompt template prefix that frames the task" constant.
    # Only the reasoning-elicitation block may differ, or a prompt difference
    # would ride along with the pipeline class and confound the comparison.
    prefix = TASK_PREFIX.format(
        open=ANSWER_OPEN, close=ANSWER_CLOSE, sentinel=REFUSAL_SENTINEL
    )
    plain = build_prompt("q", [{"passage_id": "a", "text": "t"}], reasoning=False)
    cot = build_prompt("q", [{"passage_id": "a", "text": "t"}], reasoning=True)
    assert plain.startswith(prefix)
    assert cot.startswith(prefix)
    # The CoT arm differs only by the added reasoning block.
    assert len(cot) > len(plain)


def test_prompt_permits_abstention():
    # Without this, refusal recall measures whether the model guessed the rules
    # rather than whether it abstains when evidence is insufficient (P1 5.4.1).
    prompt = build_prompt("q", [{"passage_id": "a", "text": "t"}])
    assert REFUSAL_SENTINEL in prompt
    assert ANSWER_OPEN in prompt and ANSWER_CLOSE in prompt


def test_final_answer_extraction_is_strict_about_closing():
    assert extract_final_answer(f"x {ANSWER_OPEN}1889{ANSWER_CLOSE} y") == "1889"
    # An unclosed field is unextractable, not salvaged: scoring a partial answer
    # would make a truncation failure look like a reasoning failure.
    assert extract_final_answer(f"x {ANSWER_OPEN}1889") is None
    assert extract_final_answer("no field at all") is None


def test_last_final_answer_wins():
    text = f"{ANSWER_OPEN}1887{ANSWER_CLOSE} on reflection {ANSWER_OPEN}1889{ANSWER_CLOSE}"
    assert extract_final_answer(text) == "1889"


def test_subquery_parsing_tolerates_model_formatting():
    assert parse_subqueries("1. when built\n2) who built it\n- where") == [
        "when built",
        "who built it",
        "where",
    ]
    assert parse_subqueries('"quoted query"') == ["quoted query"]
    assert parse_subqueries("a\nb\nc\nd", max_subqueries=2) == ["a", "b"]


def test_evidence_is_deduplicated_across_search_steps():
    # A passage re-retrieved by two sub-queries must appear once, or the agentic
    # class's effective noise ratio would depend on its search path.
    ev = [
        ("q1", [{"passage_id": "a", "text": "A text"}]),
        ("q2", [{"passage_id": "a", "text": "A text"}, {"passage_id": "b", "text": "B text"}]),
    ]
    rendered = format_evidence(ev)
    assert rendered.count("A text") == 1
    assert rendered.count("B text") == 1


def test_next_query_marker_parsing():
    assert parse_next_query("thinking\nNEXT_QUERY: who designed it") == "who designed it"
    assert parse_next_query('NEXT_QUERY: "quoted"') == "quoted"
    assert parse_next_query("no marker here") is None


# --------------------------------------------------------------------------
# Retrieval policy: the decision recorded in HANDOFF Section 2 item 5
# --------------------------------------------------------------------------


def test_initial_retrieval_preserves_membership_and_only_reorders():
    # The whole point: the retriever fixes ORDER, never membership, so the
    # constructed noise ratio is the ratio the generator sees (P1 5.4.1).
    case = make_case(n_passages=9)
    corpus = corpus_for(case)
    got = corpus.rank_case_passages(case.query, "bm25")
    assert len(got.passages) == 9
    assert {p.passage_id for p in got.passages} == {p.passage_id for p in case.retrieved_passages}


def test_initial_retrieval_never_returns_distractor_pool_passages():
    # The pool is reachable by agentic re-retrieval, but it is not part of the
    # case's constructed context and must not silently enlarge it.
    case = make_case(n_passages=3)
    pool = [Passage(passage_id=f"pool{i}", text="tower construction pool text") for i in range(5)]
    corpus = corpus_for(case, pool)
    ids = {p.passage_id for p in corpus.rank_case_passages(case.query, "bm25").passages}
    assert ids == {"p0", "p1", "p2"}


def test_re_retrieval_is_top_k_over_the_whole_sandbox():
    case = make_case(n_passages=3)
    pool = [Passage(passage_id=f"pool{i}", text="tower construction pool text") for i in range(5)]
    corpus = corpus_for(case, pool)
    got = policy(top_k=5).re_retrieve(corpus, "tower")
    assert len(got) == 5  # top_k, drawn from case set + pool


def test_noise_ratio_survives_the_initial_retrieval_at_every_level():
    # The regression this policy exists to prevent: under a top-5 cut, r=0.75
    # and r=0.90 both collapsed to 0.625 and the top of the NDC flattened.
    sig = "The tower was completed in 1889 after a two year build programme. " * 8
    dis = "The surrounding district hosts seasonal civic ceremonies each year. " * 5
    seed = Seed(
        seed_id="s0",
        source=SeedSource.TRIVIA_QA,
        query="In what year was the tower completed?",
        answer="1889",
        passages=[Passage(passage_id="s0-p0", text=sig, is_answer_bearing=True)],
    )
    pool = [Passage(passage_id=f"d{j}", text=dis) for j in range(80)]
    cases, _ = build_noise_testbed([seed], pool, None, n_cases=1)

    for case in cases:
        corpus = corpus_for(case)
        delivered = policy().initial(corpus)
        noise_tokens = sum(len(p.text.split()) for p in delivered if p.is_injected)
        total_tokens = sum(len(p.text.split()) for p in delivered)
        ratio = noise_tokens / total_tokens if total_tokens else 0.0
        assert ratio == pytest.approx(case.noise_ratio, abs=0.08)


def test_over_budget_context_warns_rather_than_truncating():
    # Dropping passages to fit would change the delivered noise ratio and put
    # the instance at the wrong point on the curve.
    case = make_case()
    big = [{"passage_id": "x", "text": "word " * 500}]
    warnings = check_context_budget(big, case, limit=100)
    assert len(warnings) == 1 and "CONTEXT OVERFLOW" in warnings[0]
    assert check_context_budget(big, case, limit=10_000) == []


# --------------------------------------------------------------------------
# Naive pipeline (P1 Section 5.6.1)
# --------------------------------------------------------------------------


async def test_naive_issues_exactly_one_generator_call():
    # The cost baseline the other two classes are measured against.
    provider = ScriptedProvider([answered("1889")])
    result = await NaivePipeline(provider, policy()).run(corpus_for(make_case()))
    assert result.ok
    assert result.generator_calls == 1
    assert len(provider.requests) == 1
    assert result.final_answer == "1889"


async def test_naive_never_requests_thinking_mode():
    provider = ScriptedProvider([answered("1889")])
    await NaivePipeline(provider, policy()).run(corpus_for(make_case()))
    assert provider.requests[0].thinking is False


async def test_naive_sees_every_passage_in_the_case():
    case = make_case(n_passages=7)
    provider = ScriptedProvider([answered("1889")])
    result = await NaivePipeline(provider, policy()).run(corpus_for(case))
    assert len(result.retrieved_passage_ids) == 7


async def test_provider_failure_is_recorded_not_raised():
    provider = ScriptedProvider([answered("1889")], fail_at=0)
    result = await NaivePipeline(provider, policy()).run(corpus_for(make_case()))
    assert not result.ok
    assert "upstream exploded" in (result.error or "")
    assert result.final_answer is None


# --------------------------------------------------------------------------
# Reasoning pipeline (P1 Section 5.6.2)
# --------------------------------------------------------------------------


async def test_reasoning_trained_form_sets_thinking_and_keeps_trace_out_of_the_answer():
    provider = ScriptedProvider([answered("1889")])
    result = await ReasoningPipeline(provider, policy(), thinking=True).run(
        corpus_for(make_case())
    )
    assert provider.requests[0].thinking is True
    assert result.final_answer == "1889"
    # The trace is carried separately, never inside the scored field (P1 5.6.4).
    assert result.reasoning_trace == "trace"
    assert "trace" not in (result.final_answer or "")


async def test_cot_prompted_form_adds_the_block_without_thinking_mode():
    provider = ScriptedProvider([answered("1889")])
    await ReasoningPipeline(provider, policy(), cot_prompt=True).run(corpus_for(make_case()))
    req = provider.requests[0]
    assert req.thinking is False
    assert "step by step" in req.prompt


def test_reasoning_forms_are_mutually_exclusive():
    # Stacking a CoT prompt onto a thinking-mode model would confound the two
    # reasoning forms P1 5.6.2 sets out to compare.
    with pytest.raises(ValueError, match="never both"):
        ReasoningPipeline(ScriptedProvider([]), policy(), thinking=True, cot_prompt=True)
    with pytest.raises(ValueError, match="must select one"):
        ReasoningPipeline(ScriptedProvider([]), policy())


async def test_reasoning_costs_one_call_like_naive():
    # P1 5.6.2 extracts the answer from one response "by a post-processing step",
    # so the reasoning class must not silently cost two calls per instance.
    provider = ScriptedProvider([answered("1889")])
    result = await ReasoningPipeline(provider, policy(), thinking=True).run(
        corpus_for(make_case())
    )
    assert result.generator_calls == 1


async def test_retrieval_is_identical_between_naive_and_reasoning():
    # "the reasoning-augmented configuration uses the same retrieval step as the
    # naive configuration" (P1 5.6.2) — what makes the comparison clean.
    case = make_case(n_passages=6)
    a = await NaivePipeline(ScriptedProvider([answered("x")]), policy()).run(corpus_for(case))
    b = await ReasoningPipeline(
        ScriptedProvider([answered("x")]), policy(), thinking=True
    ).run(corpus_for(case))
    assert a.retrieved_passage_ids == b.retrieved_passage_ids


# --------------------------------------------------------------------------
# Agentic pipeline (P1 Section 5.6.3)
# --------------------------------------------------------------------------


async def test_agentic_decomposes_then_commits():
    provider = ScriptedProvider(["when built\nwho built it", answered("1889")])
    result = await AgenticPipeline(provider, policy()).run(corpus_for(make_case()))
    assert result.ok
    assert result.final_answer == "1889"
    # One decompose call plus one react call: it committed at the first step.
    assert result.generator_calls == 2
    nodes = [s["node"] for s in result.steps]
    assert nodes == ["decompose", "retrieve", "react"]


async def test_agentic_loops_until_it_commits():
    provider = ScriptedProvider(
        [
            "first search\nsecond search",
            "NEXT_QUERY: more detail",
            "NEXT_QUERY: still more",
            answered("1889"),
        ]
    )
    result = await AgenticPipeline(provider, policy(), max_steps=5).run(corpus_for(make_case()))
    assert result.final_answer == "1889"
    assert result.generator_calls == 4  # decompose + three react steps
    assert [s["node"] for s in result.steps].count("retrieve") == 3


async def test_agentic_step_bound_is_enforced_and_forces_a_synthesis():
    # P1 5.6.3 fixes the bound "to keep cost and latency comparable"; a runaway
    # loop would make the agentic class's cost incomparable across cases.
    provider = ScriptedProvider(["a\nb", "NEXT_QUERY: again"])  # never commits
    result = await AgenticPipeline(provider, policy(), max_steps=3).run(corpus_for(make_case()))
    # 1 decompose + 3 react steps + 1 forced synthesise
    assert result.generator_calls == 5
    assert [s["node"] for s in result.steps].count("retrieve") == 3
    assert result.steps[-1]["node"] == "synthesise"
    assert any("step bound" in w for w in result.warnings)


async def test_agentic_re_retrieval_cannot_escape_the_sandbox():
    # The study's central control (P1 5.6.5). Even an agent searching directly
    # for the withheld answer must not surface an answer-bearing passage.
    case = make_case(n_passages=4, answerable=False)
    corpus = corpus_for(case)
    corpus.assert_perturbation_holds()
    provider = ScriptedProvider(
        ["what year was it completed\nwhen was construction finished", "NEXT_QUERY: the year"]
    )
    result = await AgenticPipeline(provider, policy(), max_steps=4).run(corpus)
    reachable = {p.passage_id for p in corpus.passages if p.is_answer_bearing}
    assert reachable == set()
    assert set(result.retrieved_passage_ids) <= {p.passage_id for p in corpus.passages}


async def test_agentic_state_is_inspectable_post_hoc():
    # P1 5.6.3 requires state that "can be inspected post hoc to diagnose
    # failure modes", so every node writes a replayable record.
    provider = ScriptedProvider(["sub one\nsub two", "NEXT_QUERY: again", answered("1889")])
    result = await AgenticPipeline(provider, policy()).run(corpus_for(make_case()))
    record = result.as_record()
    assert record["n_steps"] == len(result.steps)
    decompose = result.steps[0]
    assert decompose["subqueries"] == ["sub one", "sub two"]
    retrievals = [s for s in result.steps if s["node"] == "retrieve"]
    assert [r["query"] for r in retrievals] == ["sub one", "again"]
    assert all("passage_ids" in r for r in retrievals)


async def test_agentic_falls_back_loudly_when_decomposition_is_unusable():
    provider = ScriptedProvider(["", answered("1889")])
    result = await AgenticPipeline(provider, policy()).run(corpus_for(make_case()))
    assert result.final_answer == "1889"
    assert any("no usable sub-queries" in w for w in result.warnings)
    assert result.steps[1]["query"] == "In what year was the tower completed?"


async def test_agentic_failure_mid_loop_is_recorded_with_partial_state():
    provider = ScriptedProvider(["a\nb", "NEXT_QUERY: again"], fail_at=1)
    result = await AgenticPipeline(provider, policy()).run(corpus_for(make_case()))
    assert not result.ok
    assert "react step 0" in (result.error or "")
    # The retrieval that already happened is still on the record.
    assert result.retrieved_passage_ids


# --------------------------------------------------------------------------
# Output-cap truncation must be visible, not silently a null answer
# --------------------------------------------------------------------------


def test_hit_output_cap_detects_a_length_finish():
    from ragrobust.pipelines.base import hit_output_cap
    from ragrobust.providers.base import GenerationResponse

    def resp(reason):
        return GenerationResponse(text="x", model="m", family="f",
                                  meta={"finish_reason": reason})

    assert hit_output_cap(resp("length"))
    assert not hit_output_cap(resp("stop"))
    assert hit_output_cap(resp("stop"), resp("length"))  # any call in the loop
    assert not hit_output_cap(None)


def test_truncated_is_recorded_separately_from_a_wrong_answer():
    """A reasoning model can exhaust its budget inside the thinking block.

    The answer it had already reached is then never written out, and the
    instance looks identical to a wrong answer. Because this concentrates in the
    reasoning and agentic arms -- they think longest -- leaving it invisible
    would read as "reasoning does not help" for a mechanical reason. Observed on
    a real case: 3,386 completion tokens per call, unclosed <think>, no
    <final_answer>, null answer, on a case the model had already solved.
    """
    from ragrobust.pipelines.base import PipelineResult

    r = PipelineResult(case_id="c", pipeline_class="agentic",
                       final_answer=None, raw_text="<think>...", truncated=True)
    assert r.as_record()["truncated"] is True
    # Default stays False so an ordinary null answer is not mislabelled.
    assert PipelineResult(case_id="c", pipeline_class="naive",
                          final_answer=None, raw_text="").as_record()["truncated"] is False
