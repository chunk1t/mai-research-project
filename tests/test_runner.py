"""Experiment matrix and resumable bookkeeping (P1 5.6.5, 5.8).

Expected counts are taken from P1 Table 5.5 by hand -- three pipeline classes,
two retrievers, two generators each, twelve configurations -- and asserted
against the expansion rather than read back from it.
"""

from __future__ import annotations

import json

import pytest
import yaml
from pathlib import Path

from ragrobust.runner import (
    Configuration,
    ResultStore,
    WorkItem,
    expand_matrix,
    pending_work,
    run_plan,
)
from ragrobust.schema import Dimension, Passage, PerturbationType, SeedSource, TestCase

CFG = {
    "generators": {
        "standard_1": {"family": "alibaba"},
        "standard_2": {"family": "zhipu"},
        "reasoning_1": {"family": "alibaba"},
        "reasoning_2": {"family": "zhipu"},
    },
    "agentic_generators": ["reasoning_1", "reasoning_2"],
    "retrievers": ["bm25", "dpr"],
}


def mk_case(i: int) -> TestCase:
    return TestCase(
        case_id=f"case-{i}",
        query=f"q{i}",
        retrieved_passages=[Passage(passage_id=f"p{i}", text="text", is_answer_bearing=True)],
        answer="1889",
        dimension=Dimension.CONFLICT,
        perturbation_type=PerturbationType.CONTRADICTION_INJECTED,
        seed_source=SeedSource.NATURAL_QUESTIONS,
        seed_id=f"s{i}",
    )


def test_matrix_is_twelve_configurations():
    """P1 Table 5.5: 3 classes x 2 retrievers x 2 generators = 12."""
    configs = expand_matrix(CFG)
    assert len(configs) == 12
    assert len({c.config_id for c in configs}) == 12
    for pclass in ("naive", "reasoning", "agentic"):
        assert sum(1 for c in configs if c.pipeline_class == pclass) == 4
    for retriever in ("bm25", "dpr"):
        assert sum(1 for c in configs if c.retriever == retriever) == 6


def test_each_class_draws_the_right_generators():
    """Naive gets the standard arms, reasoning and agentic the reasoning arms."""
    configs = expand_matrix(CFG)
    got = {
        pclass: {c.generator_key for c in configs if c.pipeline_class == pclass}
        for pclass in ("naive", "reasoning", "agentic")
    }
    assert got["naive"] == {"standard_1", "standard_2"}
    assert got["reasoning"] == {"reasoning_1", "reasoning_2"}
    assert got["agentic"] == {"reasoning_1", "reasoning_2"}


def test_agentic_generators_must_subset_the_reasoning_set():
    """P1 5.6.3. A standard generator here would confound the central contrast."""
    bad = {**CFG, "agentic_generators": ["standard_1"]}
    with pytest.raises(ValueError, match="subset of the reasoning generators"):
        expand_matrix(bad)

    unknown = {**CFG, "agentic_generators": ["reasoning_9"]}
    with pytest.raises(ValueError, match="unknown generators"):
        expand_matrix(unknown)


def test_shipped_config_yields_twelve_configurations():
    """The real configs/models.yaml, not a fixture -- P1 commits to twelve."""
    path = Path(__file__).resolve().parent.parent / "configs" / "models.yaml"
    cfg = yaml.safe_load(path.read_text())
    assert len(expand_matrix(cfg)) == 12


def test_resume_skips_recorded_pairs_only(tmp_path):
    """Hand-computed: 3 cases x 12 configs = 36; record 2, leaving 34."""
    cases = [mk_case(i) for i in range(3)]
    configs = expand_matrix(CFG)
    path = tmp_path / "results.jsonl"

    with ResultStore(path) as store:
        assert run_plan(cases, configs, store)["pending"] == 36
        items = list(pending_work(cases, configs, store))[:2]
        for it in items:
            store.record(it, {"final_answer": "1889", "error": None})

    # Reopened from disk: the file is the only source of truth about progress.
    with ResultStore(path) as store:
        plan = run_plan(cases, configs, store)
        assert plan["already_done"] == 2
        assert plan["pending"] == 34
        remaining = {i.key for i in pending_work(cases, configs, store)}
        assert all(i.key not in remaining for i in items)


def test_a_truncated_row_is_rerun_not_treated_as_done(tmp_path):
    """A crash mid-write must cost one pair, never silently skip it."""
    path = tmp_path / "results.jsonl"
    cases = [mk_case(0)]
    configs = expand_matrix(CFG)
    with ResultStore(path) as store:
        item = next(pending_work(cases, configs, store))
        store.record(item, {"final_answer": "1889"})
    # Simulate a partial line appended by an interrupted write.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"key": "naive|dpr|standard_1::case-0", "final_ans')

    with ResultStore(path) as store:
        assert store.malformed_rows == 1
        pending_keys = {i.key for i in pending_work(cases, configs, store)}
        assert "naive|dpr|standard_1::case-0" in pending_keys
        assert item.key not in pending_keys


def test_records_carry_the_case_and_config_identity(tmp_path):
    """A results row must be traceable to the exact arm that produced it (P1 5.6.5)."""
    path = tmp_path / "results.jsonl"
    case = mk_case(0)
    config = Configuration("agentic", "dpr", "reasoning_2")
    with ResultStore(path) as store:
        store.record(WorkItem(case, config), {"final_answer": "1889"})
    row = json.loads(path.read_text().strip())
    assert row["case_id"] == "case-0"
    assert row["config_id"] == "agentic|dpr|reasoning_2"
    assert row["pipeline_class"] == "agentic"
    assert row["retriever"] == "dpr"
    assert row["generator_key"] == "reasoning_2"
    assert row["dimension"] == "conflict"


def test_pending_work_alternates_between_generator_models(tmp_path):
    """Both model servers must be kept busy.

    Strict configuration-major order hits one model at a time, so the four Qwen
    arms run while the GLM containers idle. With five containers per model
    against a ten-GPU limit that halves the fleet for the entire run.
    """
    cases = [mk_case(i) for i in range(3)]
    configs = expand_matrix(CFG)
    with ResultStore(tmp_path / "r.jsonl") as store:
        keys = [i.config.generator_key for i in list(pending_work(cases, configs, store))[:8]]
    families = ["qwen" if k.endswith("_1") else "glm" for k in keys]
    assert len(set(families)) == 2, f"only one model exercised in the first 8 items: {keys}"
    # Consecutive items should not all target the same model.
    runs = max(len(list(g)) for _, g in __import__("itertools").groupby(families))
    assert runs < len(families), "work is not alternating between models"


def test_configuration_major_order_is_still_available(tmp_path):
    """The tidy ordering remains, for a run that wants contiguous configs."""
    cases = [mk_case(i) for i in range(3)]
    configs = expand_matrix(CFG)
    with ResultStore(tmp_path / "r.jsonl") as store:
        first = [i.config.config_id for i in
                 list(pending_work(cases, configs, store, interleave_by_generator=False))[:3]]
    assert len(set(first)) == 1


def test_interleaving_still_covers_every_pair_exactly_once(tmp_path):
    """Round-robin must not drop or duplicate work."""
    cases = [mk_case(i) for i in range(4)]
    configs = expand_matrix(CFG)
    with ResultStore(tmp_path / "r.jsonl") as store:
        keys = [i.key for i in pending_work(cases, configs, store)]
    assert len(keys) == len(cases) * len(configs) == 48
    assert len(set(keys)) == 48


def test_every_pipeline_class_gets_the_same_decoding_parameters():
    """P1 5.6.5 holds decoding parameters constant across configurations.

    The pipeline classes ship different max_tokens defaults -- 1024 for naive,
    2048 for reasoning and agentic -- so a runner that passes none gives the
    naive arm half the output budget and confounds the class comparison. The
    value must come from `controls`, identically, for all three.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_experiment", Path(__file__).resolve().parent.parent / "scripts" / "run_experiment.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    models_cfg = {
        "generators": {
            "standard_1": {"family": "alibaba"},
            "reasoning_1": {"family": "alibaba", "thinking": True},
        },
        "controls": {"max_output_tokens": 4096, "temperature": 0.0, "seed": 7,
                     "max_agentic_steps": 5},
    }
    got = {
        pclass: mod.make_pipeline(
            Configuration(pclass, "bm25",
                          "standard_1" if pclass == "naive" else "reasoning_1"),
            provider=None, policy=None, models_cfg=models_cfg,
        ).max_tokens
        for pclass in ("naive", "reasoning", "agentic")
    }
    assert set(got.values()) == {4096}, f"decoding budget differs by class: {got}"


def test_output_budget_fits_the_served_context_window():
    """max_context_tokens + max_output_tokens must fit vLLM's max_model_len.

    The window must hold the largest real prompt plus the full output budget.
    Measured with the Qwen tokenizer, the largest case is 10,574 tokens; with
    ~350 of template and an 8,192 output budget that is 19,116 against the
    20,480 the deploy serves. Asserted against the deploy script itself so
    raising one without the other fails here rather than truncating silently.
    """
    import yaml
    from pathlib import Path

    controls = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "models.yaml").read_text()
    )["controls"]
    import re

    src = (Path(__file__).resolve().parent.parent / "deploy" / "modal_vllm.py").read_text()
    served = int(re.search(r"^MAX_MODEL_LEN = (\d+)", src, re.M).group(1))
    LARGEST_PROMPT = 17568  # measured on 150 r=0.90 cases, AutoTokenizer("Qwen/Qwen3-8B")
    TEMPLATE = 350
    assert LARGEST_PROMPT + TEMPLATE + controls["max_output_tokens"] <= served, (
        f"largest prompt + output budget exceeds the {served}-token served window"
    )


def test_distractor_pool_does_not_depend_on_the_subset_being_run():
    """P1 5.6.5's 'controlled distractor pool' must not vary with invocation.

    Building it from the selected cases made a --smoke run construct a different
    sandbox from the full run: the same case, configuration and seed answered
    "1934" in one and "INSUFFICIENT EVIDENCE" in the other. The pool must come
    from the whole benchmark so a smoke result predicts the full-run result.
    """
    from ragrobust.corpus import build_distractor_pool

    # Non-answer-bearing passages are what a distractor pool is made of; the
    # builder skips answer-bearing ones by design.
    everything = [
        Passage(passage_id=f"d{i}", text=f"unrelated filler passage {i}", is_answer_bearing=False)
        for i in range(30)
    ]
    from_all = build_distractor_pool(everything, "1889", max_size=20)
    from_subset = build_distractor_pool(everything[:3], "1889", max_size=20)
    assert len(from_all) == 20 and len(from_subset) == 3, (
        f"pool size tracks the source set: {len(from_all)} vs {len(from_subset)}"
    )

    src = (Path(__file__).resolve().parent.parent / "scripts" / "run_experiment.py").read_text()
    assert "for c in all_cases for p in c.retrieved_passages" in src, \
        "runner must build the distractor pool from the full benchmark, not the run subset"


def test_errored_instances_are_retried_on_resume(tmp_path):
    """A transient failure must not become permanent.

    Cold containers, stale replicas after a redeploy, and read timeouts all
    produce errored rows. Marking them complete would make one blip permanent
    across every future resume of a 34,000-instance run, and the instances would
    be silently missing from the results rather than retried.
    """
    path = tmp_path / "results.jsonl"
    cases = [mk_case(0)]
    configs = expand_matrix(CFG)

    with ResultStore(path) as store:
        items = list(pending_work(cases, configs, store))
        store.record(items[0], {"final_answer": "1889", "error": None})
        store.record(items[1], {"final_answer": None, "error": "ReadTimeout"})

    with ResultStore(path) as store:
        pending = {i.key for i in pending_work(cases, configs, store)}
        assert items[0].key not in pending, "a clean result must not be re-run"
        assert items[1].key in pending, "an errored instance must be retried"
        assert store.errored == {items[1].key}
        assert run_plan(cases, configs, store)["retrying_after_error"] == 1


def test_a_later_success_supersedes_an_earlier_failure(tmp_path):
    """Retries append; the store must treat the newest row as authoritative."""
    path = tmp_path / "results.jsonl"
    cases = [mk_case(0)]
    configs = expand_matrix(CFG)

    with ResultStore(path) as store:
        item = next(pending_work(cases, configs, store))
        store.record(item, {"final_answer": None, "error": "ReadTimeout"})
    with ResultStore(path) as store:
        item = next(i for i in pending_work(cases, configs, store) if i.key == item.key)
        store.record(item, {"final_answer": "1889", "error": None})
    with ResultStore(path) as store:
        assert item.key not in {i.key for i in pending_work(cases, configs, store)}
        assert store.errored == set()


def _bench_shape():
    """A miniature of the real benchmark: balanced refusal, 5 ratios per seed."""
    from ragrobust.schema import NO_ANSWER
    out = []
    for i in range(20):
        out.append(mk_case(i))                      # conflict
    for i in range(20):
        c = mk_case(1000 + i).model_copy(update={
            "dimension": Dimension.REFUSAL,
            "perturbation_type": PerturbationType.ANSWER_PASSAGE_REMOVED,
            "answer": NO_ANSWER,
            "retrieved_passages": [Passage(passage_id=f"u{i}", text="t", is_answer_bearing=False)],
        })
        out.append(c)
        out.append(mk_case(2000 + i).model_copy(update={
            "dimension": Dimension.REFUSAL,
            "perturbation_type": PerturbationType.ANSWER_PASSAGE_RETAINED,
        }))
    for seed in range(10):
        for r in (0.0, 0.25, 0.5, 0.75, 0.9):
            out.append(mk_case(3000 + seed * 10 + int(r * 100)).model_copy(update={
                "dimension": Dimension.NOISE,
                "perturbation_type": PerturbationType.DISTRACTORS_ADDED,
                "noise_ratio": r,
                "seed_id": f"nseed-{seed}",
            }))
    return out


def test_smoke_subset_keeps_the_refusal_testbed_balanced():
    """All-controls made Refusal F1 report 0.0 for every configuration."""
    from ragrobust.runner import smoke_subset
    from ragrobust.schema import NO_ANSWER

    sub = smoke_subset(_bench_shape(), 4)
    ref = [c for c in sub if c.dimension is Dimension.REFUSAL]
    unans = sum(1 for c in ref if c.answer == NO_ANSWER)
    assert unans >= 1 and unans < len(ref), f"unbalanced smoke refusal set: {unans}/{len(ref)}"


def test_smoke_subset_takes_whole_noise_seeds_with_every_ratio():
    """A curve needs all five points; the old slice fitted one seed's ratios."""
    from ragrobust.runner import smoke_subset

    sub = smoke_subset(_bench_shape(), 3)
    noise = [c for c in sub if c.dimension is Dimension.NOISE]
    seeds = {c.seed_id for c in noise}
    assert len(seeds) == 3, f"expected 3 whole seeds, got {len(seeds)}"
    for s in seeds:
        ratios = {c.noise_ratio for c in noise if c.seed_id == s}
        assert ratios == {0.0, 0.25, 0.5, 0.75, 0.9}, f"seed {s} missing ratios: {ratios}"


def test_smoke_subset_is_deterministic():
    a = smoke_ids(4); b = smoke_ids(4)
    assert a == b


def smoke_ids(n):
    from ragrobust.runner import smoke_subset
    return [c.case_id for c in smoke_subset(_bench_shape(), n)]
