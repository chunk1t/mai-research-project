"""Tests for schema invariants and the three perturbation generators."""

from __future__ import annotations

import random

import numpy as np
import pytest
import yaml

from ragrobust.corpus import SandboxedCorpus, build_distractor_pool
from ragrobust.dataset.conflict import build_conflict_case, verify_contradiction
from ragrobust.dataset.noise import (
    achieved_ratio,
    build_noise_testbed,
    distractor_states_answer,
    select_distractor_pool,
    select_distractors_for_ratio,
)
from ragrobust.dataset.refusal import build_refusal_testbed, select_replacement_passage
from ragrobust.dataset.seeds import (
    DEFAULT_EVIDENCE_TOKENS,
    Seed,
    extract_answer_window,
    extract_context_windows,
    filter_seeds,
    strip_markup_for_embedding,
    verify_seed_integrity,
)
from ragrobust.schema import (
    NO_ANSWER,
    NOISE_RATIOS,
    Benchmark,
    Dimension,
    Passage,
    PerturbationType,
    SeedSource,
    TestCase,
)

RNG = random.Random(0)


def mk_seed(i: int, answer: str = "1889", n_extra: int = 3) -> Seed:
    passages = [
        Passage(
            passage_id=f"s{i}-p0",
            text=f"The tower in city {i} was completed in {answer} after two years.",
            is_answer_bearing=True,
        )
    ]
    for j in range(n_extra):
        passages.append(
            Passage(
                passage_id=f"s{i}-p{j + 1}",
                text=f"City {i} has many landmarks and attracts visitors each season number {j}.",
            )
        )
    return Seed(
        seed_id=f"s{i}",
        source=SeedSource.NATURAL_QUESTIONS if i % 2 else SeedSource.TRIVIA_QA,
        query=f"In what year was the tower in city {i} completed?",
        answer=answer,
        passages=passages,
    )


def mk_embeddings(seeds: list[Seed], dim: int = 16) -> dict[str, np.ndarray]:
    """Deterministic pseudo-embeddings with controlled similarity structure."""
    out: dict[str, np.ndarray] = {}
    for s in seeds:
        base = np.array([RNG.gauss(0, 1) for _ in range(dim)])
        for k, p in enumerate(s.passages):
            jitter = np.array([RNG.gauss(0, 0.35) for _ in range(dim)])
            out[p.passage_id] = base + jitter * (1 + 0.1 * k)
    return out


# --------------------------------------------------------------------------
# Schema invariants
# --------------------------------------------------------------------------


def test_unanswerable_case_rejects_answer_bearing_passage():
    with pytest.raises(ValueError, match="retains an answer-bearing passage"):
        TestCase(
            case_id="bad",
            query="q",
            retrieved_passages=[Passage(passage_id="p", text="t", is_answer_bearing=True)],
            answer=NO_ANSWER,
            dimension=Dimension.REFUSAL,
            perturbation_type=PerturbationType.ANSWER_PASSAGE_REMOVED,
            seed_source=SeedSource.NATURAL_QUESTIONS,
            seed_id="s",
        )


def test_answerable_control_requires_real_answer():
    with pytest.raises(ValueError, match="must have a real answer"):
        TestCase(
            case_id="bad",
            query="q",
            retrieved_passages=[Passage(passage_id="p", text="t", is_answer_bearing=True)],
            answer=NO_ANSWER,
            dimension=Dimension.REFUSAL,
            perturbation_type=PerturbationType.ANSWER_PASSAGE_RETAINED,
            seed_source=SeedSource.NATURAL_QUESTIONS,
            seed_id="s",
        )


def test_noise_ratio_required_only_for_noise_cases():
    with pytest.raises(ValueError, match="noise case requires noise_ratio"):
        TestCase(
            case_id="bad",
            query="q",
            retrieved_passages=[Passage(passage_id="p", text="t", is_answer_bearing=True)],
            answer="a",
            dimension=Dimension.NOISE,
            perturbation_type=PerturbationType.DISTRACTORS_ADDED,
            seed_source=SeedSource.NATURAL_QUESTIONS,
            seed_id="s",
        )


def test_perturbation_must_match_dimension():
    with pytest.raises(ValueError, match="invalid for dimension"):
        TestCase(
            case_id="bad",
            query="q",
            retrieved_passages=[Passage(passage_id="p", text="t", is_answer_bearing=True)],
            answer="a",
            dimension=Dimension.CONFLICT,
            perturbation_type=PerturbationType.DISTRACTORS_ADDED,
            seed_source=SeedSource.NATURAL_QUESTIONS,
            seed_id="s",
        )


# --------------------------------------------------------------------------
# Seed filtering
# --------------------------------------------------------------------------


def test_filter_drops_multi_answer_and_short_evidence():
    good = mk_seed(1)
    multi = mk_seed(2)
    multi.answer = "1889; 1887"
    short = mk_seed(3, n_extra=0)
    short.passages = [Passage(passage_id="x", text="tiny", is_answer_bearing=True)]

    kept, stats = filter_seeds([good, multi, short], min_evidence_tokens=10)
    assert [s.seed_id for s in kept] == ["s1"]
    assert stats["multi_answer"] == 1
    assert stats["evidence_too_short"] == 1


# --------------------------------------------------------------------------
# Refusal testbed
# --------------------------------------------------------------------------


def test_refusal_testbed_is_balanced_and_valid():
    seeds = [mk_seed(i) for i in range(60)]
    emb = mk_embeddings(seeds)
    cases, stats = build_refusal_testbed(
        seeds, emb, n_unanswerable=20, n_answerable=20,
        similarity_floor=-1.0, similarity_ceiling=1.0,
    )
    unans = [c for c in cases if c.perturbation_type is PerturbationType.ANSWER_PASSAGE_REMOVED]
    ctrl = [c for c in cases if c.perturbation_type is PerturbationType.ANSWER_PASSAGE_RETAINED]

    assert len(unans) == 20
    assert len(ctrl) == 20
    # Both classes present is what makes Refusal F1 precision well defined.
    assert all(c.answer == NO_ANSWER for c in unans)
    assert all(c.is_answerable for c in ctrl)
    # Seed partitions must be disjoint.
    assert not ({c.seed_id for c in unans} & {c.seed_id for c in ctrl})


def test_replacement_passage_never_leaks_the_answer():
    seeds = [mk_seed(i) for i in range(10)]
    emb = mk_embeddings(seeds)
    target = seeds[0].answer_passages[0]
    # A candidate that contains the answer must be rejected even if similar.
    leaky = Passage(passage_id="leak", text="It was completed in 1889 indeed.")
    emb["leak"] = emb[target.passage_id]
    repl = select_replacement_passage(
        target, "1889", [leaky], emb, similarity_floor=-1.0, similarity_ceiling=1.0
    )
    assert repl is None


# --------------------------------------------------------------------------
# Noise testbed
# --------------------------------------------------------------------------


def test_noise_testbed_emits_five_instances_per_seed():
    seeds = [mk_seed(i) for i in range(10)]
    pool = [
        Passage(passage_id=f"d{j}", text=" ".join(["filler"] * 60)) for j in range(40)
    ]
    cases, stats = build_noise_testbed(seeds, pool, n_cases=10)

    assert stats["n_seeds"] == 10
    assert stats["n_instances"] == 50  # 10 seeds x 5 ratios
    by_ratio: dict[float, int] = {}
    for c in cases:
        by_ratio[c.noise_ratio] = by_ratio.get(c.noise_ratio, 0) + 1
    assert set(by_ratio) == set(NOISE_RATIOS)
    assert all(v == 10 for v in by_ratio.values())


def test_zero_ratio_has_no_distractors_and_high_ratio_is_mostly_noise():
    seeds = [mk_seed(0)]
    pool = [Passage(passage_id=f"d{j}", text=" ".join(["filler"] * 50)) for j in range(80)]
    cases, _ = build_noise_testbed(seeds, pool, n_cases=1)
    by_ratio = {c.noise_ratio: c for c in cases}

    assert achieved_ratio(by_ratio[0.0].retrieved_passages) == 0.0
    # The 90 percent case must actually be dominated by noise, not merely labelled so.
    assert achieved_ratio(by_ratio[0.90].retrieved_passages) > 0.80


def test_noise_case_always_retains_the_answer_passage():
    seeds = [mk_seed(0)]
    pool = [Passage(passage_id=f"d{j}", text=" ".join(["filler"] * 50)) for j in range(80)]
    cases, _ = build_noise_testbed(seeds, pool, n_cases=1)
    for c in cases:
        assert any(p.is_answer_bearing for p in c.retrieved_passages)


# --------------------------------------------------------------------------
# Similarity thresholds, exercised at realistic values
#
# The other tests disable the thresholds (floor -1.0, ceiling 1.0) so they can
# focus on counts and balance. These tests pin the threshold logic itself, using
# unit vectors at known angles so the cosine is exact rather than approximate.
# --------------------------------------------------------------------------


def unit(theta: float) -> np.ndarray:
    return np.array([np.cos(theta), np.sin(theta)])


def test_replacement_respects_floor_and_ceiling():
    target = Passage(passage_id="t", text="The tower was completed in 1889.",
                     is_answer_bearing=True)
    # cos(0)=1.00 near-duplicate, cos(60 deg)=0.50 on topic, cos(85 deg)=0.087 off topic
    near_dup = Passage(passage_id="dup", text="A tower finished long ago.")
    on_topic = Passage(passage_id="ok", text="The tower has a famous iron frame.")
    off_topic = Passage(passage_id="far", text="Marine biology of the deep ocean.")
    emb = {
        "t": unit(0.0),
        "dup": unit(0.0),
        "ok": unit(np.pi / 3),
        "far": unit(np.pi / 2 * 0.944),
    }

    got = select_replacement_passage(
        target, "1889", [near_dup, on_topic, off_topic], emb,
        similarity_floor=0.45, similarity_ceiling=0.80,
    )
    # Near-duplicate is above the ceiling, off-topic is below the floor.
    assert got is not None
    assert got.passage_id == "ok"


def test_replacement_returns_none_when_nothing_is_in_band():
    target = Passage(passage_id="t", text="x", is_answer_bearing=True)
    off = Passage(passage_id="far", text="y")
    emb = {"t": unit(0.0), "far": unit(np.pi / 2)}  # cosine 0.0
    assert select_replacement_passage(
        target, "z", [off], emb, similarity_floor=0.45, similarity_ceiling=0.80
    ) is None


def test_distractor_pool_excludes_off_topic_passages():
    seed = mk_seed(0)
    anchor_id = seed.answer_passages[0].passage_id
    on_topic = Passage(passage_id="on", text="Nearby structures in the same district.")
    off_topic = Passage(passage_id="off", text="Quantum chromodynamics lattice methods.")
    emb = {
        anchor_id: unit(0.0),
        "on": unit(np.pi / 3),        # cosine 0.50, above floor
        "off": unit(np.pi / 2 * 0.96),  # cosine ~0.06, below floor
    }
    pool, warn = select_distractor_pool(
        seed, [on_topic, off_topic], emb, similarity_floor=0.30, similarity_ceiling=0.95
    )
    assert warn is None
    assert [p.passage_id for p in pool] == ["dist-on"]


def test_distractor_pool_is_ordered_nearest_first_and_capped():
    """P1 5.5.2 selects by "retrieving topically nearest neighbours".

    A floor alone does not do that: select_distractors_for_ratio() shuffles the
    pool, so an unbounded pool draws mostly from its weakest tail. This is the
    defect that made every sampled noise case fail the 5.5.3 review.
    """
    seed = mk_seed(0)
    anchor_id = seed.answer_passages[0].passage_id
    # Hand-computed cosines against unit(0.0): cos(pi/6)=0.866, cos(pi/4)=0.707,
    # cos(pi/3)=0.500. All three clear a 0.40 floor.
    emb = {
        anchor_id: unit(0.0),
        "near": unit(np.pi / 6),
        "mid": unit(np.pi / 4),
        "far": unit(np.pi / 3),
    }
    cands = [
        Passage(passage_id="far", text="weakly related"),
        Passage(passage_id="near", text="strongly related"),
        Passage(passage_id="mid", text="moderately related"),
    ]

    pool, warn = select_distractor_pool(
        seed, cands, emb, similarity_floor=0.40, similarity_ceiling=0.95
    )
    assert warn is None
    assert [p.passage_id for p in pool] == ["dist-near", "dist-mid", "dist-far"]

    # The cap keeps the two nearest and drops the tail, regardless of the order
    # the candidates arrived in.
    capped, _ = select_distractor_pool(
        seed, cands, emb, similarity_floor=0.40, similarity_ceiling=0.95, max_pool=2
    )
    assert [p.passage_id for p in capped] == ["dist-near", "dist-mid"]


def test_missing_embeddings_degrades_loudly_not_silently():
    seed = mk_seed(0)
    cand = [Passage(passage_id="c", text="anything at all")]
    pool, warn = select_distractor_pool(seed, cand, None)
    assert pool  # still usable
    assert warn is not None and "SKIPPED" in warn

    _, stats = build_noise_testbed([seed], cand, None, n_cases=1)
    assert any("SPEC DEVIATION" in w for w in stats["warnings"])


# --------------------------------------------------------------------------
# Conflict testbed
# --------------------------------------------------------------------------


class FakeNLI:
    def __init__(self, score: float):
        self.score = score

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        return {"contradiction": self.score, "entailment": 1 - self.score}


def test_conflict_rejects_unchanged_and_leaky_generations():
    orig = "The tower was completed in 1889 after two years."
    ok, score, reason = verify_contradiction(orig, orig, "1889", FakeNLI(0.99))
    assert not ok and reason == "unchanged"

    still_true = "The tower was completed in 1889 and also 1887."
    ok, score, reason = verify_contradiction(orig, still_true, "1889", FakeNLI(0.99))
    assert not ok and reason == "true_answer_still_present"


def test_conflict_rejects_weak_contradiction():
    orig = "The tower was completed in 1889 after two years."
    gen = "The tower was completed in 1887 after two years."
    ok, score, reason = verify_contradiction(orig, gen, "1889", FakeNLI(0.30))
    assert not ok and reason == "below_contradiction_threshold"


def test_conflict_case_contains_both_positions():
    seed = mk_seed(0)
    gen = "The tower in city 0 was completed in 1887 after two years."
    case = build_conflict_case(seed, gen, 0.95, gen_params={})
    assert len(case.retrieved_passages) == 2
    assert sum(p.is_answer_bearing for p in case.retrieved_passages) == 1
    assert case.is_answerable  # ground truth is still 1889


def test_conflict_presentation_order_is_randomised():
    """P1 5.9 requires randomized presentation order for the CRS judge.

    Over 60 cases a fair two-element shuffle puts the true passage first roughly
    half the time. The bound checked here is deliberately loose (5..55 of 60)
    because the point is to catch a constant order, not to test the quality of
    the PRNG: the old implementation scored exactly 60, and any implementation
    that shuffles at all lands far inside the bound. The probability of a
    correct implementation failing it is below 1e-9.
    """
    cases = [
        build_conflict_case(mk_seed(i), f"Contradictory text {i}.", 0.9, gen_params={})
        for i in range(60)
    ]
    first = sum(c.retrieved_passages[0].is_answer_bearing for c in cases)
    assert 5 < first < 55, f"true passage first in {first}/60 cases -- not randomised"


def test_conflict_presentation_order_is_reproducible():
    """Same seed id must give the same order across rebuilds (P1 5.6.5).

    `hash()` is salted per process, so a hash-derived order would satisfy the
    randomisation test above and still differ between runs, which would make the
    released benchmark unreproducible from its recorded seed.
    """
    a = build_conflict_case(mk_seed(7), "Contradictory text.", 0.9, gen_params={})
    b = build_conflict_case(mk_seed(7), "Contradictory text.", 0.9, gen_params={})
    assert [p.passage_id for p in a.retrieved_passages] == [
        p.passage_id for p in b.retrieved_passages
    ]

    # A different presentation seed must be able to produce a different order,
    # or the seed is not actually driving the shuffle.
    orders = {
        tuple(
            p.passage_id
            for p in build_conflict_case(
                mk_seed(7), "Contradictory text.", 0.9, gen_params={}, rng_seed=k
            ).retrieved_passages
        )
        for k in range(20)
    }
    assert len(orders) == 2


def test_conflict_records_true_passage_position():
    """The recorded position must match where the answer passage actually is.

    Checked as an invariant across cases rather than against a fixed expected
    index: a recorded position that drifts from the real one would silently
    corrupt any Chapter 7 test for a residual position effect.
    """
    for i in range(40):
        case = build_conflict_case(
            mk_seed(i), f"Contradictory text {i}.", 0.9, gen_params={}
        )
        recorded = case.gen_params["true_passage_position"]
        actual = [p.is_answer_bearing for p in case.retrieved_passages].index(True)
        assert recorded == actual


# --------------------------------------------------------------------------
# Sandboxed corpus: the central experimental control
# --------------------------------------------------------------------------


def test_sandbox_blocks_recovery_of_withheld_evidence():
    seeds = [mk_seed(i) for i in range(10)]
    emb = mk_embeddings(seeds)
    cases, _ = build_refusal_testbed(
        seeds, emb, n_unanswerable=5, n_answerable=0,
        similarity_floor=-1.0, similarity_ceiling=1.0,
    )
    unans = cases[0]
    corpus = SandboxedCorpus(unans, distractor_pool=[])
    corpus.assert_perturbation_holds()

    # Even an exhaustive retrieval cannot surface the answer.
    res = corpus.retrieve(unans.query, k=999, retriever="bm25")
    assert not any(p.is_answer_bearing for p in res.passages)


def test_sandbox_rejects_answer_bearing_distractor_pool():
    seeds = [mk_seed(i) for i in range(10)]
    emb = mk_embeddings(seeds)
    cases, _ = build_refusal_testbed(
        seeds, emb, n_unanswerable=5, n_answerable=0,
        similarity_floor=-1.0, similarity_ceiling=1.0,
    )
    leak = [Passage(passage_id="leak", text="completed in 1889", is_answer_bearing=True)]
    with pytest.raises(ValueError, match="defeat the perturbation"):
        SandboxedCorpus(cases[0], distractor_pool=leak)


def test_build_distractor_pool_filters_answer_leaks():
    cands = [
        Passage(passage_id="a", text="Completed in 1889 exactly."),
        Passage(passage_id="b", text="A pleasant city with parks."),
    ]
    pool = build_distractor_pool(cands, "1889")
    assert [p.passage_id for p in pool] == ["b"]


# --------------------------------------------------------------------------
# Config: anti-circularity audit (P1 Section 5.9)
# --------------------------------------------------------------------------


def test_model_families_are_disjoint_across_roles():
    with open("configs/models.yaml") as fh:
        cfg = yaml.safe_load(fh)

    evaluated = {g["family"] for g in cfg["generators"].values()}
    ragas = {cfg["ragas_judge"]["family"]}
    case_gen = {cfg["case_generator"]["family"]}
    judge = {cfg["crs_judge"]["family"]}

    assert not (evaluated & case_gen), "case generator shares a family with an evaluated generator"
    assert not (evaluated & judge), "CRS judge shares a family with an evaluated generator"
    assert not (case_gen & judge), "CRS judge shares a family with the case generator"
    # RAGAS is the comparison baseline, so it must not share a family with the
    # generators it is being used to assess (P1 Section 5.7).
    assert not (evaluated & ragas), "RAGAS judge shares a family with an evaluated generator"


def test_agentic_generators_are_a_subset_of_reasoning():
    with open("configs/models.yaml") as fh:
        cfg = yaml.safe_load(fh)
    reasoning = {k for k in cfg["generators"] if k.startswith("reasoning")}
    assert set(cfg["agentic_generators"]) <= reasoning


def test_benchmark_manifest_counts():
    seeds = [mk_seed(i) for i in range(30)]
    emb = mk_embeddings(seeds)
    refusal, _ = build_refusal_testbed(
        seeds, emb, n_unanswerable=5, n_answerable=5,
        similarity_floor=-1.0, similarity_ceiling=1.0,
    )
    pool = [Passage(passage_id=f"d{j}", text=" ".join(["filler"] * 50)) for j in range(60)]
    noise, _ = build_noise_testbed(seeds[20:], pool, n_cases=4)

    bench = Benchmark(version="test-0.1", cases=refusal + noise)
    m = bench.manifest()
    assert m["n_cases"] == 10 + 20
    assert m["counts_by_dimension"]["refusal"] == 10
    assert m["counts_by_dimension"]["noise"] == 20
    assert len(m["content_hash"]) == 16


# --------------------------------------------------------------------------
# Evidence preparation for the 800-token budget (configs/dataset.yaml)
#
# These cover the two loader bugs found on 24 August: both decided
# `is_answer_bearing` from different text than they stored, producing
# "answerable" cases whose answer passage did not contain the answer. The
# schema trusts the flag rather than reading the text, so nothing downstream
# would have caught it.
# --------------------------------------------------------------------------


def test_answer_window_is_centred_on_the_answer_not_the_document_start():
    before = " ".join(f"w{i}" for i in range(200))
    after = " ".join(f"x{i}" for i in range(200))
    document = f"{before} PARIS {after}"

    window = extract_answer_window(document, "paris", 50)
    assert window is not None
    assert len(window.split()) == 50
    assert "paris" in window.lower()
    # Taking the opening 50 words would have missed it entirely.
    assert " ".join(document.split()[:50]).lower().find("paris") == -1

    # Genuinely centred, not merely anchored at the answer. Context on both
    # sides matters: an answer pinned to the first token of every passage is a
    # positional artefact, and P1 Section 5.5.2 holds positional sensitivity
    # constant precisely so it cannot covary with a perturbation.
    tokens = window.split()
    assert tokens.index("PARIS") > 5, "answer sits at the window edge, not centred"
    assert any(t.startswith("w") for t in tokens), "no preceding context retained"
    assert any(t.startswith("x") for t in tokens), "no following context retained"


def test_answer_window_returns_none_when_the_answer_is_absent():
    # The signal to drop the seed rather than store a passage that does not
    # support the answer it claims to.
    assert extract_answer_window("nothing relevant here at all", "1889", 50) is None
    assert extract_answer_window("some text", "", 50) is None


def test_short_document_is_returned_whole():
    assert extract_answer_window("built in 1889 exactly", "1889", 500) == "built in 1889 exactly"


def test_answer_window_respects_the_token_budget():
    document = " ".join(["filler"] * 400 + ["1889"] + ["filler"] * 400)
    for budget in (20, 100, 350):
        window = extract_answer_window(document, "1889", budget)
        assert window is not None
        assert len(window.split()) <= budget
        assert "1889" in window


def test_answer_at_the_document_edges_is_still_captured():
    at_start = "1889 " + " ".join(["filler"] * 500)
    at_end = " ".join(["filler"] * 500) + " 1889"
    for document in (at_start, at_end):
        window = extract_answer_window(document, "1889", 50)
        assert window is not None and "1889" in window


def test_strip_markup_keeps_every_word_of_prose():
    # Hand-computed: the eight prose words survive, the six tags do not.
    passage = (
        "<P> Virgin Australia commenced services in 2000 . </P> "
        "<Table> <Tr> <Td> Fleet </Td> </Tr> </Table>"
    )
    assert strip_markup_for_embedding(passage) == (
        "Virgin Australia commenced services in 2000 . Fleet"
    )


def test_strip_markup_is_embedding_only_and_leaves_the_passage_alone():
    # The transform must never be written back: pipelines see the raw markup,
    # and only the vector is computed from the stripped text.
    original = "<P> text </P>"
    p = Passage(passage_id="p0", text=original)
    strip_markup_for_embedding(p.text)
    assert p.text == original


def test_strip_markup_does_not_eat_prose_containing_angle_brackets():
    # A comparison in prose is not markup. The 80-character bound stops a stray
    # "<" from swallowing the rest of the passage.
    assert strip_markup_for_embedding("a < b and c > d") == "a < b and c > d"


def test_context_windows_tile_the_document_without_overlap():
    # 300 words at 100 per window tiles into exactly three: w0-w99, w100-w199,
    # w200-w299. Hand-computed; the answer is absent so nothing is filtered.
    document = " ".join(f"w{i}" for i in range(300))
    windows = extract_context_windows(document, "zzz", 100)

    assert len(windows) == 3
    assert [len(w.split()) for w in windows] == [100, 100, 100]
    assert windows[0].split()[0] == "w0"
    assert windows[1].split()[0] == "w100"
    assert windows[2].split()[-1] == "w299"


def test_context_window_carrying_the_answer_is_dropped():
    # PARIS sits at index 150, which lands inside the SECOND window (w100-w199).
    # That window must be dropped, leaving the first and third.
    words = [f"w{i}" for i in range(300)]
    words[150] = "PARIS"
    windows = extract_context_windows(" ".join(words), "paris", 100)

    assert len(windows) == 2
    assert all("paris" not in w.lower() for w in windows)
    assert windows[0].split()[0] == "w0"
    assert windows[1].split()[0] == "w200"


def test_answer_match_is_conservative_not_word_boundary():
    # "King George I" is a bare substring of "King George III". For evidence the
    # word-boundary test is right, but for a DISTRACTOR the conservative test is:
    # losing a candidate costs nothing, admitting one that states the answer
    # breaks the noise testbed's second review criterion (P1 5.5.3).
    document = " ".join(["filler"] * 60 + ["King", "George", "III"] + ["filler"] * 60)
    assert extract_context_windows(document, "King George I", 100) == []


def test_short_tail_window_is_discarded():
    # 230 words at 100 per window: starts 0, 100, 200. The tail holds only 30
    # words, below the 50-token floor, so it is dropped as a fragment.
    document = " ".join(f"w{i}" for i in range(230))
    windows = extract_context_windows(document, "zzz", 100)
    assert len(windows) == 2

    # 260 words leaves a 60-word tail, which clears the floor and is kept.
    document = " ".join(f"w{i}" for i in range(260))
    windows = extract_context_windows(document, "zzz", 100)
    assert len(windows) == 3
    assert len(windows[2].split()) == 60


def test_document_shorter_than_the_floor_yields_nothing():
    assert extract_context_windows(" ".join(["w"] * 20), "zzz", 100) == []


def test_context_windows_respect_the_requested_cap():
    # 1,000 words at 100 per window could tile into ten; the cap holds it to six
    # so one long document cannot dominate the shared distractor corpus.
    document = " ".join(f"w{i}" for i in range(1000))
    assert len(extract_context_windows(document, "zzz", 100, n_windows=6)) == 6


def test_seed_integrity_catches_a_mislabelled_answer_passage():
    # Exactly the corruption the old NQ loader produced: flagged answer-bearing,
    # but the stored text does not contain the answer.
    bad = Seed(
        seed_id="bad",
        source=SeedSource.NATURAL_QUESTIONS,
        query="q",
        answer="1889",
        passages=[Passage(passage_id="p0", text="no year here", is_answer_bearing=True)],
    )
    problems = verify_seed_integrity(bad)
    assert len(problems) == 1
    assert "does not occur in the stored text" in problems[0]


def test_seed_integrity_passes_a_well_formed_seed():
    good = Seed(
        seed_id="good",
        source=SeedSource.NATURAL_QUESTIONS,
        query="q",
        answer="1889",
        passages=[
            Passage(passage_id="p0", text="It was completed in 1889.", is_answer_bearing=True),
            Passage(passage_id="p1", text="A pleasant city.", is_answer_bearing=False),
        ],
    )
    assert verify_seed_integrity(good) == []


def test_seed_integrity_requires_an_answer_bearing_passage():
    orphan = Seed(
        seed_id="orphan",
        source=SeedSource.TRIVIA_QA,
        query="q",
        answer="1889",
        passages=[Passage(passage_id="p0", text="It was 1889.", is_answer_bearing=False)],
    )
    assert any("no answer-bearing passage" in p for p in verify_seed_integrity(orphan))


def test_prepared_evidence_fits_the_noise_budget():
    # The whole point of the 700-token default: a noise case at r=0.90 carries
    # signal/(1-0.90) = 10x the signal, and must fit max_context_tokens=8192.
    document = " ".join(["word"] * 5000 + ["1889"] + ["word"] * 5000)
    window = extract_answer_window(document, "1889", DEFAULT_EVIDENCE_TOKENS)
    assert window is not None
    signal_tokens = len(window.split())
    assert signal_tokens <= DEFAULT_EVIDENCE_TOKENS
    assert signal_tokens / (1 - 0.90) <= 8192


# --------------------------------------------------------------------------
# Answer matching: word boundaries, not bare substrings
# --------------------------------------------------------------------------


def test_word_boundary_matching_rejects_substrings_of_longer_words():
    """The defect that failed a real smoke run.

    A TriviaQA seed with gold "King George I" was flagged answer-bearing because
    its passage said "King George III". The passage did not answer the question;
    the model refused correctly and would have been scored wrong.
    """
    from ragrobust.matching import contains_on_word_boundary

    assert not contains_on_word_boundary("King George I", "King George III agreed")
    assert contains_on_word_boundary("King George I", "succeeded by King George I in 1714")
    assert not contains_on_word_boundary("UK", "Slovakia and Ukraine")
    assert not contains_on_word_boundary("glycine", "N-methylglycine")
    # The short-answer convention P1 5.4.3 relies on must still hold.
    assert contains_on_word_boundary("1889", "It was completed in 1889")
    # Accents are folded, because the loaders store some answers unaccented.
    assert contains_on_word_boundary("Bogota", "riots in Bogotá spread")


def test_seed_integrity_catches_a_substring_only_flag():
    """verify_seed_integrity must reject the flag the smoke run exposed."""
    seed = Seed(
        seed_id="tqa-804",
        source=SeedSource.TRIVIA_QA,
        query="Who succeeded Queen Anne to the throne?",
        answer="King George I",
        passages=[Passage(passage_id="p0",
                          text="King George III agreed to surrender the hereditary revenues.",
                          is_answer_bearing=True)],
    )
    problems = verify_seed_integrity(seed)
    assert problems, "a substring-only match must not pass the integrity gate"
    assert "does not occur" in problems[0]


def test_scoring_does_not_credit_a_longer_wrong_answer():
    """contains_answer graded 'King George III' correct against 'King George I'."""
    from ragrobust.parsing.rules import contains_answer, score_answer

    assert not contains_answer("King George III", "King George I")
    assert contains_answer("It was King George I", "King George I")
    assert not score_answer("King George III", "King George I").correct


def test_distractors_are_taken_nearest_first_not_shuffled():
    """The pool arrives ranked; consuming it in order is what keeps it topical.

    A shuffle here silently undoes select_distractor_pool()'s ranking. At a low
    ratio only one distractor fits, so a shuffle picks a random member of the
    pool rather than the nearest -- which is what left 314 of 500 r=0.25 cases
    with no same-article distractor and failing the P1 5.5.3 review.
    """
    import random as _random

    # 100 signal tokens at r=0.25 gives a budget of 100*0.25/0.75 = 33.3 tokens,
    # so exactly one 30-token distractor is taken. It must be the nearest.
    pool = [
        Passage(passage_id=f"dist-rank{i}", text=" ".join(["w"] * 30), is_injected=True)
        for i in range(12)
    ]
    for seed in (1, 2, 3, 99):
        chosen = select_distractors_for_ratio(100, pool, 0.25, _random.Random(seed))
        assert [p.passage_id for p in chosen] == ["dist-rank0"], \
            "distractor choice must not depend on the RNG, only on the ranking"


def test_distractor_answer_check_sees_through_markup_and_inversion():
    """Hand-built from the two leak classes found in the built testbed.

    A raw `answer.lower() in text.lower()` passes all three of these, which is
    how eighteen instances kept a distractor that stated the gold answer.
    """
    # Markup splitting the answer, and NQ's spaced-out punctuation.
    assert distractor_states_answer("the <I>Double</I> was released", "The Double")
    assert distractor_states_answer("Blue laws in the United States .", "Blue laws in the United States")
    # Wikipedia bibliography inversion: "Sparks , Nicholas" for "Nicholas Sparks".
    assert distractor_states_answer("Sparks , Nicholas . A Walk to Remember", "Nicholas Sparks")
    # Genuinely absent stays absent.
    assert not distractor_states_answer("a passage about something else", "The Double")
    # Word boundaries still hold: "King George I" must not match "King George III".
    assert not distractor_states_answer("King George III reigned", "King George I")
    # Inversion is restricted to alphabetic pairs, or the prose "1978 in ..."
    # would be read as the answer "in 1978" reversed.
    assert not distractor_states_answer("published 1978 in London", "in 1978")


def test_distractor_leak_detection_catches_partial_answer_forms():
    """Matching only the full gold string let partial forms through.

    Found by hand in three of twenty-nine sampled noise cases, each sitting in a
    passage about the query's own subject, so the answer was plainly
    recoverable: "Anakin" for "Anakin Skywalker", "Leonardo" for "Leonardo da
    Vinci", "Wembley" for "Wembley Stadium".
    """
    from ragrobust.dataset.seeds import distractor_states_answer as leaks

    assert leaks("a Force Ghost of Anakin appears", "Anakin Skywalker")
    assert leaks("Leonardo's work was not a fresco", "Leonardo da Vinci")
    assert leaks("games would take place at Wembley", "Wembley Stadium")
    assert leaks("discussion of the Boleyn family", "Anne Boleyn")
    # TriviaQA renders golds in upper case; components must still be found.
    assert leaks("a biography of Stevenson", "ROBERT LOUIS STEVENSON")


def test_distractor_leak_detection_ignores_common_components():
    """A common token cannot identify an answer on its own.

    Rejecting on these would empty distractor pools for no gain: an r=0.90 case
    needs nine distractor tokens per signal token, so every needless rejection
    pushes the pool deeper into its tail.
    """
    from ragrobust.dataset.seeds import distractor_states_answer as leaks

    assert not leaks("New South Wales fleeces", "South Africa")
    assert not leaks("the November election", "November 1999")
    assert not leaks("an island in the north", "Holy Island")
    assert not leaks("John arrived early", "John Smith")


def test_distractor_leak_detection_errs_toward_over_rejection():
    """Over-rejection is the correct direction and is deliberate.

    A distractor discarded because it mentions "phosphorus" when the answer is
    "Phosphorus pentoxide" costs one pool candidate. A distractor KEPT because
    only part of the answer appears makes the case trivially answerable and
    flattens the curve the noise dimension exists to bend.
    """
    from ragrobust.dataset.seeds import distractor_states_answer as leaks

    assert leaks("phosphorus is a reactive element", "Phosphorus pentoxide")


def test_single_word_answers_are_unaffected_by_the_component_rule():
    """A one-word answer has no components; the full-string test covers it."""
    from ragrobust.dataset.seeds import distractor_states_answer as leaks

    assert leaks("the answer is Titanium here", "Titanium")
    assert not leaks("a passage about zirconium", "Titanium")
