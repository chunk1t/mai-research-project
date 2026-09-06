"""Response parser and refusal classifier tests (P1 Section 5.6.4).

Expected values were worked out by hand. The classifier's fallback model is a
scripted fake, so every assertion is about what the protocol does with a known
response rather than about what a model happens to say.
"""

from __future__ import annotations

import pytest

from ragrobust.metrics.refusal import compute_refusal_f1
from ragrobust.metrics.scored import ResponseCategory
from ragrobust.parsing import (
    ResponseParser,
    contains_answer,
    exact_match,
    looks_like_refusal,
    normalise_answer,
    rule_classify,
    score_answer,
    token_f1,
    validation_sample,
)
from ragrobust.pipelines.base import PipelineResult
from ragrobust.pipelines.prompts import ANSWER_CLOSE, ANSWER_OPEN, REFUSAL_SENTINEL
from ragrobust.providers.base import GenerationRequest, GenerationResponse, LLMProvider, RateLimiter
from ragrobust.providers.factory import ConfigError, assert_disjoint_families, load_config
from ragrobust.schema import (
    NO_ANSWER,
    Dimension,
    Passage,
    PerturbationType,
    SeedSource,
    TestCase,
)


class ScriptedClassifier(LLMProvider):
    def __init__(self, replies: list[str], *, family: str = "anthropic", broken: bool = False):
        super().__init__(name="fake", model="fake", family=family, limiter=RateLimiter(2))
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.broken = broken

    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        self.prompts.append(req.prompt)
        if self.broken:
            raise RuntimeError("classifier down")
        text = self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]
        return GenerationResponse(text=text, model=self.model, family=self.family)


def case(*, answerable: bool = True, dimension: Dimension = Dimension.REFUSAL) -> TestCase:
    if answerable:
        return TestCase(
            case_id="ctrl-1",
            query="In what year was the tower completed?",
            retrieved_passages=[Passage(passage_id="p0", text="Built 1889.", is_answer_bearing=True)],
            answer="1889",
            dimension=dimension,
            perturbation_type=(
                PerturbationType.ANSWER_PASSAGE_RETAINED
                if dimension is Dimension.REFUSAL
                else PerturbationType.DISTRACTORS_ADDED
            ),
            noise_ratio=0.25 if dimension is Dimension.NOISE else None,
            seed_source=SeedSource.TRIVIA_QA,
            seed_id="s",
        )
    return TestCase(
        case_id="unans-1",
        query="In what year was the tower completed?",
        retrieved_passages=[Passage(passage_id="p0", text="A pleasant city.")],
        answer=NO_ANSWER,
        dimension=Dimension.REFUSAL,
        perturbation_type=PerturbationType.ANSWER_PASSAGE_REMOVED,
        seed_source=SeedSource.TRIVIA_QA,
        seed_id="s",
    )


def result(final: str | None, raw: str = "", **kw) -> PipelineResult:
    return PipelineResult(
        case_id=kw.pop("case_id", "ctrl-1"),
        pipeline_class=kw.pop("pipeline_class", "naive"),
        final_answer=final,
        raw_text=raw or (f"{ANSWER_OPEN}{final}{ANSWER_CLOSE}" if final else ""),
        **kw,
    )


# --------------------------------------------------------------------------
# Normalisation and correctness
# --------------------------------------------------------------------------


def test_normalisation_follows_the_squad_convention():
    assert normalise_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalise_answer("  A  Dog's  life. ") == "dogs life"
    assert normalise_answer("1889") == "1889"


def test_exact_match_is_strict_and_containment_is_forgiving():
    assert exact_match("1889", "1889")
    assert exact_match("The 1889", "1889")  # article stripped
    assert not exact_match("It was 1889", "1889")
    # Containment is the short-answer QA convention, including RGB (P1 5.4.3).
    assert contains_answer("It was completed in 1889", "1889")
    assert not contains_answer("It was completed in 1887", "1889")


def test_token_f1_worked_example():
    # pred normalises to 5 tokens [built, in, 1889, by, eiffel], gold to [1889].
    # common = 1, precision = 1/5 = 0.2, recall = 1/1 = 1.0,
    # F1 = 2 * 0.2 * 1.0 / 1.2 = 0.3333
    assert token_f1("built in 1889 by Eiffel", "1889") == pytest.approx(0.3333, abs=1e-4)
    assert token_f1("1889", "1889") == 1.0
    assert token_f1("1887", "1889") == 0.0


def test_score_answer_reports_components_separately():
    s = score_answer("It was completed in 1889", "1889")
    assert s.correct is True
    assert s.exact_match is False  # the stricter figure stays available for Ch 7
    assert s.contains is True


# --------------------------------------------------------------------------
# Rule-based classification
# --------------------------------------------------------------------------


def test_substantive_answer_is_classified_as_an_answer():
    v = rule_classify("1889")
    assert v.category is ResponseCategory.ANSWER and v.confident


def test_the_prompted_sentinel_is_a_refusal():
    v = rule_classify(REFUSAL_SENTINEL)
    assert v.category is ResponseCategory.REFUSAL and v.confident


@pytest.mark.parametrize(
    "field",
    [
        "The evidence is insufficient.",
        "There is not enough information to answer.",
        "The passages do not mention the completion year.",
        "This cannot be determined from the context.",
        "I don't know.",
        "unknown",
    ],
)
def test_curated_abstention_patterns_are_recognised(field):
    assert looks_like_refusal(field)
    assert rule_classify(field).category is ResponseCategory.REFUSAL


@pytest.mark.parametrize(
    "field",
    ["1889", "Possibly 1889", "1889, though one source says 1887", "The Eiffel Tower"],
)
def test_hedged_but_committed_answers_are_not_refusals(field):
    # A bare hedge accompanies a committed answer more often than it replaces
    # one; treating it as a refusal would inflate false positives on answerable
    # controls, the exact over-caution Refusal F1 exists to catch (P1 5.4.3).
    assert not looks_like_refusal(field)
    assert rule_classify(field).category is ResponseCategory.ANSWER


def test_missing_and_empty_fields_are_unparseable():
    assert rule_classify(None).category is ResponseCategory.UNPARSEABLE
    assert rule_classify("   ").category is ResponseCategory.UNPARSEABLE


def test_long_field_with_an_abstention_phrase_is_escalated_not_guessed():
    field = (
        "The first passage does not mention the completion year at all, however "
        "the second passage states clearly that the tower was completed in 1889 "
        "following a two year construction programme run by Gustave Eiffel."
    )
    v = rule_classify(field)
    assert v.confident is False  # goes to the language-model fallback


# --------------------------------------------------------------------------
# The protocol: only the final-answer field is ever read (P1 5.6.4)
# --------------------------------------------------------------------------


async def test_uncertain_trace_with_a_committed_answer_scores_as_an_answer():
    # P1 5.6.4 is explicit: "a trace that expresses uncertainty followed by a
    # committed answer is classified as a substantive answer, not a refusal".
    parser = ResponseParser()
    r = result(
        "1889",
        raw="I am not sure, the evidence may be insufficient, I cannot tell... "
        f"{ANSWER_OPEN}1889{ANSWER_CLOSE}",
    )
    r.reasoning_trace = "There is not enough information. I don't know. Unknown."
    scored = await parser.score(case(), r)
    assert scored.category is ResponseCategory.ANSWER
    assert scored.answer_correct is True


async def test_committed_trace_with_a_refusing_answer_scores_as_a_refusal():
    parser = ResponseParser()
    r = result(REFUSAL_SENTINEL)
    r.reasoning_trace = "The answer is clearly 1889 based on passage one."
    scored = await parser.score(case(), r)
    assert scored.category is ResponseCategory.REFUSAL
    # Whether abstaining was right is Refusal F1's business, not accuracy's.
    assert scored.answer_correct is None


async def test_unparseable_is_a_non_refusal_and_incorrect():
    # The conservative treatment of P1 5.6.4: it penalises ambiguous output
    # rather than rewarding it as a lucky abstention.
    parser = ResponseParser()
    scored = await parser.score(case(answerable=False), result(None, raw="rambling"))
    assert scored.category is ResponseCategory.UNPARSEABLE
    assert scored.answer_correct is None
    metrics = compute_refusal_f1([scored])
    assert metrics.fn == 1 and metrics.tp == 0  # counted as a failure to refuse


async def test_provider_error_is_unparseable_not_a_model_behaviour():
    parser = ResponseParser()
    r = result(None)
    r.error = "HTTP 500"
    scored = await parser.score(case(), r)
    assert scored.category is ResponseCategory.UNPARSEABLE
    assert scored.classification_method == "provider_error"


async def test_correctness_is_only_scored_on_answerable_cases():
    parser = ResponseParser()
    scored = await parser.score(case(answerable=False), result("1889"))
    # An unanswerable case has no answer to match, so there is nothing to score.
    assert scored.category is ResponseCategory.ANSWER
    assert scored.answer_correct is None


async def test_noise_dimension_carries_its_ratio_through_to_the_metric():
    parser = ResponseParser()
    scored = await parser.score(case(dimension=Dimension.NOISE), result("1889"))
    assert scored.noise_ratio == 0.25
    assert scored.answer_correct is True


# --------------------------------------------------------------------------
# Language-model fallback
# --------------------------------------------------------------------------


async def test_fallback_is_consulted_only_for_ambiguous_fields():
    fallback = ScriptedClassifier(["ANSWER"])
    parser = ResponseParser(fallback)
    await parser.score(case(), result("1889"))
    assert fallback.prompts == []  # confident rule verdict, no call issued
    assert parser.counts == {"rule": 1}


async def test_fallback_resolves_an_ambiguous_field():
    long_mixed = (
        "The first passage does not mention the year, however the second states "
        "the tower was completed in 1889 after a two year construction programme "
        "overseen by Gustave Eiffel and his engineering team."
    )
    fallback = ScriptedClassifier(["ANSWER"])
    parser = ResponseParser(fallback)
    scored = await parser.score(case(), result(long_mixed))
    assert len(fallback.prompts) == 1
    assert scored.category is ResponseCategory.ANSWER
    assert scored.classification_method == "llm_fallback"


async def test_fallback_never_sees_the_reasoning_trace():
    # The trace must not enter the classification, or a verbose hedge would be
    # scored instead of the commitment (P1 5.6.4).
    long_mixed = "The passages do not say, " + "and further detail follows here " * 6
    fallback = ScriptedClassifier(["REFUSAL"])
    parser = ResponseParser(fallback)
    r = result(long_mixed)
    r.reasoning_trace = "SECRET_TRACE_MARKER"
    await parser.score(case(), r)
    assert "SECRET_TRACE_MARKER" not in fallback.prompts[0]


async def test_fallback_failure_degrades_loudly_to_the_rule_verdict():
    long_mixed = "The passages do not say, " + "and further detail follows here " * 6
    parser = ResponseParser(ScriptedClassifier([], broken=True))
    verdict = await parser.classify("q", long_mixed)
    assert verdict.method == "rule_unconfident_no_fallback"
    assert verdict.warnings and "failed" in verdict.warnings[0]


async def test_no_configured_fallback_is_a_visible_spec_deviation():
    long_mixed = "The passages do not say, " + "and further detail follows here " * 6
    parser = ResponseParser()
    verdict = await parser.classify("q", long_mixed)
    assert any("SPEC DEVIATION" in w for w in verdict.warnings)


async def test_unusable_fallback_reply_falls_back_to_the_rule_verdict():
    long_mixed = "The passages do not say, " + "and further detail follows here " * 6
    parser = ResponseParser(ScriptedClassifier(["I am not sure what you mean"]))
    verdict = await parser.classify("q", long_mixed)
    assert verdict.method == "rule_unconfident_no_fallback"


# --------------------------------------------------------------------------
# Kappa validation support (P1 5.6.4) and anti-circularity
# --------------------------------------------------------------------------


async def test_validation_sample_is_stratified_across_categories():
    parser = ResponseParser()
    scored = []
    for i in range(30):
        scored.append(await parser.score(case(), result("1889", case_id=f"a{i}")))
    for i in range(5):
        scored.append(await parser.score(case(), result(REFUSAL_SENTINEL, case_id=f"r{i}")))
    for i in range(3):
        scored.append(await parser.score(case(), result(None, case_id=f"u{i}")))

    sample = validation_sample(scored, n=9)
    labels = {row["machine_label"] for row in sample}
    # Refusals and unparseable responses are rarer but are where parsing errors
    # matter most, so they must not be crowded out by answers.
    assert labels == {"answer", "refusal", "unparseable"}
    assert all(row["human_label"] == "" for row in sample)
    assert all("final_answer" in row for row in sample)


def test_shipped_config_keeps_the_classifier_off_the_evaluated_families():
    cfg = load_config("configs/models.yaml")
    assert cfg["refusal_classifier"]["family"] == "anthropic"
    evaluated = {g["family"] for g in cfg["generators"].values()}
    assert cfg["refusal_classifier"]["family"] not in evaluated
    assert assert_disjoint_families(cfg) is None


def test_classifier_sharing_a_family_with_an_evaluated_generator_is_rejected():
    # It decides whether an evaluated generator refused, so a sibling could
    # read its own phrasing as commitment and bias Refusal F1 one way.
    cfg = {
        "generators": {"g1": {"family": "alibaba"}},
        "case_generator": {"family": "google"},
        "crs_judge": {"family": "anthropic"},
        "refusal_classifier": {"family": "alibaba"},
    }
    with pytest.raises(ConfigError, match="refusal classifier shares"):
        assert_disjoint_families(cfg)


# --------------------------------------------------------------------------
# Recovering answers that omitted the delimiter (P1 5.6.4 deviation)
# --------------------------------------------------------------------------


def test_bare_sentinel_is_recovered_as_a_refusal():
    """GLM-4-9B largely ignores the delimiter instruction Qwen follows.

    On the live run it returned a bare "INSUFFICIENT EVIDENCE" for 49.3% of
    standard-arm responses against 0.4% for Qwen. Scoring those as unparseable
    would count a correct abstention as a failure, and because the behaviour is
    model-specific it would report that GLM never abstains.
    """
    from ragrobust.parsing.rules import recover_unparsed_answer as recover

    assert recover("INSUFFICIENT EVIDENCE") == ("INSUFFICIENT EVIDENCE", "bare_sentinel")
    assert recover("  insufficient evidence.  ")[0] == "INSUFFICIENT EVIDENCE"
    # Some models bracket it as though it were a tag.
    assert recover("<INSUFFICIENT EVIDENCE>")[0] == "INSUFFICIENT EVIDENCE"


def test_trailing_sentinel_after_a_justification_is_recovered():
    from ragrobust.parsing.rules import recover_unparsed_answer as recover

    text = ('The passages provided do not contain any information regarding who '
            'plays Jack. Therefore, the answer is:\n\n<INSUFFICIENT EVIDENCE>')
    assert recover(text) == ("INSUFFICIENT EVIDENCE", "trailing_sentinel")


def test_malformed_opening_tag_is_recovered():
    """Observed shape: the model put its answer where the tag name belongs."""
    from ragrobust.parsing.rules import recover_unparsed_answer as recover

    assert recover("<Tina Turner></final_answer>") == ("Tina Turner", "malformed_open_tag")
    assert recover("<Leslie></final_answer>")[0] == "Leslie"
    # The tag name itself is not an answer.
    assert recover("<final_answer></final_answer>") == (None, None)


def test_free_prose_and_truncated_traces_are_NOT_recovered():
    """The narrow scope is the point.

    Picking a clause out of undelimited prose would be the scorer inventing a
    commitment the model never marked -- and a truncated reasoning trace, which
    genuinely never committed, looks exactly like it.
    """
    from ragrobust.parsing.rules import recover_unparsed_answer as recover

    assert recover("The answer is clearly 1889 based on the passage.") == (None, None)
    assert recover("But the user's instruction says to put the answer in the tags. "
                   "So, the answer is that") == (None, None)
    assert recover("") == (None, None)
    assert recover("   \n  ") == (None, None)


def test_recovered_refusal_classifies_as_a_refusal():
    """End to end: recovery must feed the classifier, not just return a string."""
    from ragrobust.parsing.rules import recover_unparsed_answer, rule_classify

    answer, method = recover_unparsed_answer("INSUFFICIENT EVIDENCE")
    assert method == "bare_sentinel"
    verdict = rule_classify(answer)
    assert verdict.category is ResponseCategory.REFUSAL
