"""Tests for the offline demonstration (`scripts/demo.py`).

Two kinds of check live here, and they guard different things.

The pure ones fix the logic `verify` uses to decide whether a departure from
the stored record is accounted for. That logic is what stands between an honest
report of the run's reproducibility and a check that quietly excuses whatever
it happens to find, so its expectations are worked out by hand below.

The data-dependent ones guard the curated cases against the benchmark. A demo
that names a case which has been renamed fails in front of an audience; here it
fails in the suite. They skip when the run artefacts are absent, since `runs/`
and `data/cache/` are not in version control.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import demo  # noqa: E402

from ragrobust.cache import ResponseCache  # noqa: E402
from ragrobust.corpus import SandboxedCorpus  # noqa: E402
from ragrobust.pipelines.base import RetrievalPolicy  # noqa: E402
from ragrobust.pipelines.naive import NaivePipeline  # noqa: E402
from ragrobust.pipelines.prompts import ANSWER_CLOSE, ANSWER_OPEN  # noqa: E402
from ragrobust.providers.base import (  # noqa: E402
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    RateLimiter,
)
from ragrobust.providers.cached import CachedProvider  # noqa: E402
from ragrobust.providers.replay import ReplayProvider  # noqa: E402
from ragrobust.schema import (  # noqa: E402
    Dimension,
    Passage,
    PerturbationType,
    SeedSource,
    TestCase,
)

HAVE_ARTEFACTS = (ROOT / "data" / "cases" / "benchmark.jsonl").exists() and (
    ROOT / "runs" / "results.jsonl"
).exists()
needs_run = pytest.mark.skipif(not HAVE_ARTEFACTS, reason="run artefacts not present")


# --------------------------------------------------------------------------
# Which configurations a demonstration runs
# --------------------------------------------------------------------------


def test_the_three_classes_share_a_base_model_within_an_arm():
    """Expected by hand from P1 Sections 5.6.2 and 5.6.3.

    The naive class runs the standard generator, the reasoning class its
    reasoning form, and the agentic class inherits the reasoning generator
    because `agentic_generators` must be a subset of the reasoning set. Arm 1 is
    therefore standard_1 / reasoning_1 / reasoning_1 -- one base model, with
    orchestration as the only thing that changes down the column.
    """
    assert [c.config_id for c in demo.configs_for(1, "bm25")] == [
        "naive|bm25|standard_1",
        "reasoning|bm25|reasoning_1",
        "agentic|bm25|reasoning_1",
    ]
    assert [c.config_id for c in demo.configs_for(2, "dpr")] == [
        "naive|dpr|standard_2",
        "reasoning|dpr|reasoning_2",
        "agentic|dpr|reasoning_2",
    ]


def test_the_sibling_of_an_arm_is_the_other_retriever_and_nothing_else():
    key = "naive|bm25|standard_1::conflict-nq-1"
    assert demo._sibling(key) == "naive|dpr|standard_1::conflict-nq-1"
    assert demo._sibling(demo._sibling(key)) == key


# --------------------------------------------------------------------------
# Whether a cache entry is shared
#
# The cache is keyed on the prompt, and the prompt is built from the retrieved
# passages in order. Two retriever arms that produced the same ordered list
# therefore issued one request and share one entry; anything else is two
# entries. Order matters: the same passages ranked differently are a different
# prompt.
# --------------------------------------------------------------------------


def _stored(bm25_ids, dpr_ids=None):
    out = {"naive|bm25|standard_1::c1": {"retrieved_passage_ids": bm25_ids}}
    if dpr_ids is not None:
        out["naive|dpr|standard_1::c1"] = {"retrieved_passage_ids": dpr_ids}
    return out


def test_identical_retrieval_order_means_one_shared_entry():
    stored = _stored(["p1", "p2", "p3"], ["p1", "p2", "p3"])
    assert demo._entry_is_shared("naive|bm25|standard_1::c1", stored) is True


def test_the_same_passages_in_a_different_order_are_two_entries():
    """A reordering changes the prompt, so it cannot be a shared entry."""
    stored = _stored(["p1", "p2", "p3"], ["p2", "p1", "p3"])
    assert demo._entry_is_shared("naive|bm25|standard_1::c1", stored) is False


def test_a_missing_sibling_is_unknown_rather_than_unshared():
    """Undecidable must not read as 'unshared'.

    An unshared instance is held to byte-exact reproduction; treating an
    unrecorded sibling as unshared would apply that standard to an instance
    whose sharing was never established, and fail the run for it.
    """
    assert demo._entry_is_shared("naive|bm25|standard_1::c1", _stored(["p1"])) is None


# --------------------------------------------------------------------------
# Provenance: every response replay serves is one the run recorded
# --------------------------------------------------------------------------


def test_provenance_index_covers_final_texts_and_intermediate_steps():
    """Hand-computed: three distinct texts under standard_1, one under reasoning_1.

    Step outputs have to be indexed as well as final texts. The agentic
    decompose response is never a `raw_text` anywhere -- it only ever appears
    inside `steps` -- so an index built from final texts alone would report a
    perfectly ordinary replayed decomposition as a response the run never made.
    """
    stored = {
        "naive|bm25|standard_1::c1": {"raw_text": "A"},
        "naive|dpr|standard_1::c2": {"raw_text": "B"},
        "agentic|bm25|standard_1::c3": {
            "raw_text": "A",  # a duplicate collapses; the index is a set
            "steps": [{"node": "decompose", "output": "C"}, {"node": "retrieve"}],
        },
        "naive|bm25|reasoning_1::c4": {"raw_text": "D"},
    }
    index = demo._provenance_index(stored)

    assert len(index["standard_1"]) == 3
    assert len(index["reasoning_1"]) == 1
    assert demo._sha("C") in index["standard_1"]
    assert demo._sha("D") not in index["standard_1"]


def test_own_decompose_reads_the_decompose_step_only():
    row = {"steps": [{"node": "retrieve", "output": "wrong"},
                     {"node": "decompose", "output": "right"}]}
    assert demo._own_decompose(row) == "right"
    assert demo._own_decompose({"steps": [{"node": "retrieve"}]}) is None
    assert demo._own_decompose({}) is None


def test_sha_is_over_the_utf8_bytes():
    assert demo._sha("héllo") == hashlib.sha1("héllo".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def test_snippet_never_exceeds_its_limit():
    """A 10-character body at limit 5 is nine characters plus the ellipsis."""
    assert demo.snippet("abcdefghij", 5) == "abcd…"
    assert demo.snippet("abcde", 5) == "abcde"
    assert demo.snippet("  a\n\n  b  ", 20) == "a b"


def test_sentence_split_keeps_terminators_and_drops_blanks():
    assert demo._sentences("One. Two! Three?  Four") == ["One.", "Two!", "Three?", "Four"]
    assert demo._sentences("   ") == []


def test_ink_emits_nothing_when_disabled():
    assert demo.Ink(False).bad("x") == "x"
    assert demo.Ink(True).bad("x") != "x"


# --------------------------------------------------------------------------
# The replay path, end to end, with no network anywhere
# --------------------------------------------------------------------------


class ScriptedProvider(LLMProvider):
    """Answers once with a fixed text and counts how often it was called."""

    def __init__(self, text: str):
        super().__init__(name="fake", model="fake-model", family="alibaba",
                         limiter=RateLimiter(2))
        self.text = text
        self.calls = 0

    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        self.calls += 1
        return GenerationResponse(text=self.text, model=self.model, family=self.family,
                                  prompt_tokens=11, completion_tokens=7)


def _case() -> TestCase:
    return TestCase(
        case_id="conflict-demo-1",
        query="In what year was the tower completed?",
        retrieved_passages=[
            Passage(passage_id="p0", text="The tower was completed in 1889 after two years.",
                    is_answer_bearing=True),
            Passage(passage_id="p0-contra", text="The tower was completed in 1934 after two years.",
                    is_injected=True),
        ],
        answer="1889",
        dimension=Dimension.CONFLICT,
        perturbation_type=PerturbationType.CONTRADICTION_INJECTED,
        seed_source=SeedSource.NATURAL_QUESTIONS,
        seed_id="nq-1",
    )


async def test_a_recorded_run_replays_through_the_real_pipeline(tmp_path):
    """Run once against a model, then again with the model taken away.

    This is the demonstration in miniature: the same `NaivePipeline`, the same
    sandboxed corpus, the same parsing, and a provider that can only read the
    cache. The scripted provider's call count is the proof -- it does not move
    on the second run, so nothing was generated.
    """
    case = _case()
    corpus = SandboxedCorpus(case, distractor_pool=[])
    policy = RetrievalPolicy(retriever="bm25")
    cache = ResponseCache(tmp_path / "c.sqlite")
    scripted = ScriptedProvider(f"Reasoning here.\n{ANSWER_OPEN}1889{ANSWER_CLOSE}")

    live = await NaivePipeline(CachedProvider(scripted, cache), policy, seed=20260721).run(corpus)
    assert scripted.calls == 1
    assert live.final_answer == "1889"

    replayed = await NaivePipeline(
        ReplayProvider(cache, name="replay", model="fake-model", family="alibaba"),
        policy,
        seed=20260721,
    ).run(SandboxedCorpus(_case(), distractor_pool=[]))

    assert scripted.calls == 1, "the replay reached the model"
    assert replayed.final_answer == live.final_answer
    assert replayed.raw_text == live.raw_text
    assert replayed.retrieved_passage_ids == live.retrieved_passage_ids
    cache.close()


async def test_a_changed_seed_is_a_different_run_and_is_not_served(tmp_path):
    """The cache key covers the seed, so replay must not substitute a near miss."""
    cache = ResponseCache(tmp_path / "c.sqlite")
    scripted = ScriptedProvider(f"{ANSWER_OPEN}1889{ANSWER_CLOSE}")
    policy = RetrievalPolicy(retriever="bm25")
    await NaivePipeline(CachedProvider(scripted, cache), policy, seed=20260721).run(
        SandboxedCorpus(_case(), distractor_pool=[])
    )

    provider = ReplayProvider(cache, name="replay", model="fake-model", family="alibaba")
    # The miss propagates out of the pipeline rather than being absorbed into a
    # result. `NaivePipeline` inspects `resp.error`, so a provider that
    # returned a failed response here would produce an instance with a null
    # answer -- indistinguishable, downstream, from a model that emitted no
    # answer field. `scripts/demo.py` catches this and says which key was absent.
    from ragrobust.providers.replay import CacheMiss

    with pytest.raises(CacheMiss):
        await NaivePipeline(provider, policy, seed=1).run(
            SandboxedCorpus(_case(), distractor_pool=[])
        )
    assert provider.n_misses == 1
    cache.close()


# --------------------------------------------------------------------------
# The curated cases, against the benchmark
# --------------------------------------------------------------------------


@needs_run
def test_every_curated_case_exists_in_the_benchmark():
    ids = {c.case_id for c in demo.load_corpus(with_results=False).cases}
    missing = []
    for show in demo.SHOWCASE:
        parts = show.command.split()
        if parts[0] == "case":
            missing += [parts[1]] if parts[1] not in ids else []
        elif parts[0] == "noise":
            missing += [
                f"noise-{parts[1]}-r{r}"
                for r in demo.NOISE_RATIOS
                if f"noise-{parts[1]}-r{r}" not in ids
            ]
    assert missing == [], f"curated cases not in the benchmark: {missing}"


def test_every_curated_command_is_runnable_and_declares_its_arm():
    """The printed command and the `arm` field must not drift apart.

    `list` prints the command a presenter types; `arm` is what the tests and
    the HTML build replay. If those disagree, the demonstration on stage and
    the demonstration in the browser show different configurations.
    """
    for show in demo.SHOWCASE:
        parts = show.command.split()
        assert parts[0] in {"case", "noise"}, show.command
        declared = int(parts[parts.index("--generator") + 1]) if "--generator" in parts else 1
        assert declared == show.arm, f"{show.key}: command says arm {declared}, field says {show.arm}"
        assert show.arm in (1, 2)
    assert len({s.key for s in demo.SHOWCASE}) == len(demo.SHOWCASE), "duplicate showcase key"


def test_the_rubric_gloss_covers_every_score_the_run_can_produce():
    """CRS is 0 to 4 inclusive (P1 5.4.3); a missing gloss would raise on print."""
    assert sorted(demo.CRS_RUBRIC) == [0, 1, 2, 3, 4]
