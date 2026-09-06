"""CRS judge prompt and score parsing (P1 5.4.3, 5.6.4, 5.9)."""

from __future__ import annotations

import pytest

from ragrobust.judge import (
    RUBRIC_LEVELS,
    build_crs_prompt,
    format_rubric,
    parse_crs_score,
)
from ragrobust.schema import Dimension, Passage, PerturbationType, SeedSource, TestCase


def mk_conflict_case() -> TestCase:
    return TestCase(
        case_id="conflict-x1",
        query="In what year was the Eiffel Tower completed?",
        retrieved_passages=[
            Passage(passage_id="p0-contra", text="The tower was finished in 1887.",
                    is_answer_bearing=False, is_injected=True),
            Passage(passage_id="p0", text="The tower was completed in 1889.",
                    is_answer_bearing=True),
        ],
        answer="1889",
        dimension=Dimension.CONFLICT,
        perturbation_type=PerturbationType.CONTRADICTION_INJECTED,
        seed_source=SeedSource.NATURAL_QUESTIONS,
        seed_id="s1",
    )


def test_rubric_has_exactly_the_five_p1_levels():
    assert sorted(RUBRIC_LEVELS) == [0, 1, 2, 3, 4]
    # P1's distinguishing words for each level, so a reworded rubric fails here
    # rather than silently changing what the metric measures.
    assert "ignores the conflict" in RUBRIC_LEVELS[0]
    assert "without identification of positions" in RUBRIC_LEVELS[1]
    assert "at least two positions" in RUBRIC_LEVELS[2]
    assert "principled resolution" in RUBRIC_LEVELS[3]
    assert "deferral to the user" in RUBRIC_LEVELS[4]


def test_prompt_contains_the_case_and_the_whole_rubric():
    case = mk_conflict_case()
    prompt = build_crs_prompt(case, "1889")
    assert case.query in prompt
    for level in RUBRIC_LEVELS.values():
        assert level in prompt
    for p in case.retrieved_passages:
        assert p.text in prompt


def test_prompt_preserves_the_randomised_passage_order():
    """P1 5.9's position-bias mitigation must reach the judge, not stop at the
    generator. The contradiction is stored first in this case, and the prompt
    must present it first rather than reordering to put truth first."""
    case = mk_conflict_case()
    prompt = build_crs_prompt(case, "1889")
    assert prompt.index("finished in 1887") < prompt.index("completed in 1889")


def test_prompt_never_carries_a_reasoning_trace():
    """P1 5.6.4 scores only the final answer."""
    case = mk_conflict_case()
    prompt = build_crs_prompt(case, "1889")
    assert "<think>" not in prompt
    # The signature accepts only the answer, so a trace cannot be passed by
    # accident; this asserts the answer is what appears.
    assert "1889" in prompt


def test_missing_answer_is_stated_not_blank():
    prompt = build_crs_prompt(mk_conflict_case(), "   ")
    assert "no answer" in prompt.lower()


def test_score_parsing_requires_the_delimited_field():
    assert parse_crs_score("<score>3</score>") == 3
    assert parse_crs_score("Reasoning here.\n<score>0</score>") == 0
    assert parse_crs_score("<SCORE>4</SCORE>") == 4
    assert parse_crs_score("<score> 2 </score>") == 2
    # A bare digit in prose must NOT be mined: the judge often restates the
    # rubric, and "Score 3: explicit recognition" is not a verdict.
    assert parse_crs_score("Score 3: explicit recognition and a resolution.") is None
    assert parse_crs_score("I would say this is a 4.") is None
    assert parse_crs_score("") is None
    # Out of range is not silently clamped.
    assert parse_crs_score("<score>7</score>") is None


def test_format_rubric_lists_levels_in_order():
    lines = format_rubric().splitlines()
    assert [l.split(":")[0] for l in lines] == [f"Score {i}" for i in range(5)]


# --------------------------------------------------------------------------
# Validating the judge against a human rater (P1 Objective 2)
# --------------------------------------------------------------------------


def _judged(n=120):
    return [{"case_id": f"conflict-c{i}", "config_id":
             ["naive|bm25|standard_1", "reasoning|bm25|reasoning_1",
              "agentic|dpr|reasoning_2"][i % 3],
             "final_answer": f"answer {i}", "crs_score": i % 3}
            for i in range(n)]


def test_validation_sample_meets_the_p1_minimum_and_spans_classes():
    """P1 asks for at least one hundred cases.

    Balanced across pipeline classes so the judge is not validated against only
    one answering style -- a naive one-line answer and an agentic synthesis are
    different objects to score.
    """
    from ragrobust.judge import select_judge_validation_sample

    sample = select_judge_validation_sample(_judged(), n=100)
    assert len(sample) == 100
    classes = {r["config_id"].split("|")[0] for r in sample}
    assert classes == {"naive", "reasoning", "agentic"}
    counts = [sum(1 for r in sample if r["config_id"].startswith(c)) for c in classes]
    assert max(counts) - min(counts) <= 2, f"classes unbalanced: {counts}"


def test_validation_sample_is_deterministic():
    from ragrobust.judge import select_judge_validation_sample

    a = select_judge_validation_sample(_judged(), n=60, rng_seed=5)
    b = select_judge_validation_sample(_judged(), n=60, rng_seed=5)
    c = select_judge_validation_sample(_judged(), n=60, rng_seed=6)
    key = lambda s: [(r["config_id"], r["case_id"]) for r in s]  # noqa: E731
    assert key(a) == key(b)
    assert key(a) != key(c)


def test_judge_agreement_on_hand_computed_values():
    """Ten items, computed by hand.

      judge: 0,0,0,0,0,2,2,2,2,2   human: 0,0,0,0,2,2,2,2,2,0
      exact agreements: items 1-4 (0/0) and 6-9 (2/2) = 8, so Po = 0.80
      marginals: judge 5 zeros / 5 twos; human 5 zeros / 5 twos
      Pe = (5/10 * 5/10) + (5/10 * 5/10) = 0.50
      kappa = (0.80 - 0.50) / (1 - 0.50) = 0.600  -- exactly the gate
    """
    from ragrobust.judge import compute_judge_agreement

    pairs = list(zip([0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
                     [0, 0, 0, 0, 2, 2, 2, 2, 2, 0]))
    r = compute_judge_agreement(pairs)
    assert r.kappa == pytest.approx(0.600, abs=1e-6)
    assert r.observed_agreement == pytest.approx(0.80)
    assert r.exact_matches == 8
    assert r.meets_gate is True  # 0.600 >= 0.60


def test_quadratic_kappa_credits_near_misses_that_plain_kappa_does_not():
    """On an ordinal rubric, 0-vs-1 is not the same error as 0-vs-4.

    Plain kappa treats them identically, so a judge that is consistently close
    but rarely exact reads as random. Reporting both makes a near-miss on the
    gate diagnosable rather than simply fatal.
    """
    from ragrobust.judge import compute_judge_agreement

    near = compute_judge_agreement([(2, 1), (2, 3), (1, 2), (3, 2)] * 5)
    far = compute_judge_agreement([(0, 4), (4, 0)] * 10)
    assert near.quadratic_kappa > far.quadratic_kappa
    assert near.within_one == near.n
    assert far.within_one == 0


def test_agreement_rejects_out_of_range_scores():
    from ragrobust.judge import compute_judge_agreement

    with pytest.raises(ValueError, match="outside"):
        compute_judge_agreement([(2, 7)])
    with pytest.raises(ValueError, match="no rated pairs"):
        compute_judge_agreement([])
