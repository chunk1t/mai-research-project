#!/usr/bin/env python3
"""Replay the finished experiment offline, with no network and no GPU.

    python scripts/demo.py list                       # the curated cases
    python scripts/demo.py case conflict-nq-4643      # three classes, one case
    python scripts/demo.py noise tqa-2481             # one seed across five noise ratios
    python scripts/demo.py verify --n 200             # provenance and coverage check

Every generator call in the run was content-addressed into
`data/cache/generators.sqlite` (`cache.py`), so a finished configuration can be
re-executed through the real `NaivePipeline`, `ReasoningPipeline` and
`AgenticPipeline` objects with `ReplayProvider` in place of the model. The
pipelines, the sandboxed corpus, the retrieval and the parsing all run for
real; only the four evaluated generators are served from the cache.

This is a demonstration of the executed study, not a re-execution of it. It
cannot produce a new number, and it is not offered as evidence that the models
are deterministic -- Appendix A.1 records that they are not, bit for bit, across
batch compositions. What `verify` establishes is narrower and checkable: every
byte this replay shows was recorded by the run itself.

Two properties are load-bearing and both are enforced rather than asserted.

No network. `ReplayProvider` imports no HTTP client and raises `CacheMiss`
rather than falling through to a model, so an apparently offline demonstration
cannot quietly be fetching answers from a Modal container.

The full benchmark is loaded even to run one case. `build_distractor_pool` draws
from every passage in the benchmark, and `run_experiment.py` records what
happens when it does not: a pool assembled from a subset builds a different
sandbox, and the same case under the same configuration and seed answered
"1934" in one and "INSUFFICIENT EVIDENCE" in the other. Loading 2,858 cases
costs about a tenth of a second, so there is no reason to cut the corner that
would make the demonstration disagree with the report.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ragrobust.cache import ResponseCache  # noqa: E402
from ragrobust.corpus import SandboxedCorpus, build_distractor_pool  # noqa: E402
from ragrobust.pipelines.base import RetrievalPolicy  # noqa: E402
from ragrobust.pipelines.prompts import build_decompose_prompt  # noqa: E402
from ragrobust.providers.base import GenerationRequest  # noqa: E402
from ragrobust.providers.factory import load_config  # noqa: E402
from ragrobust.providers.replay import CacheMiss, ReplayProvider  # noqa: E402
from ragrobust.runner import Configuration  # noqa: E402
from ragrobust.schema import NO_ANSWER, TestCase  # noqa: E402
from run_experiment import make_pipeline  # noqa: E402

BENCHMARK = ROOT / "data" / "cases" / "benchmark.jsonl"
CACHE_PATH = ROOT / "data" / "cache" / "generators.sqlite"
RESULTS = ROOT / "runs" / "results.jsonl"
SCORED = ROOT / "runs" / "scored.jsonl"
MODELS_CONFIG = ROOT / "configs" / "models.yaml"

# The generator pair. Naming them by digit rather than by model keeps the
# comparison the study makes visible on the command line: arm 1 and arm 2 are
# different base models, and P1 Section 5.6.2 pairs each with its own reasoning
# form, so `standard_1` and `reasoning_1` are the same weights differing only in
# the reasoning variable.
ARM_CLASSES = {
    "naive": "standard_{n}",
    "reasoning": "reasoning_{n}",
    "agentic": "reasoning_{n}",
}

# The Conflict Resolution Score rubric of P1 Section 5.4.3, abbreviated for a
# terminal. Printed beside a score so a reviewer can read the number without
# being handed the report first.
CRS_RUBRIC = {
    0: "no acknowledgement of the conflict",
    1: "hedges, but does not identify the conflict",
    2: "identifies that the passages disagree",
    3: "identifies the disagreement and locates it in the passages",
    4: "locates it and resolves or explicitly declines to resolve it",
}


# --------------------------------------------------------------------------
# The curated cases
#
# Chosen to argue the thesis rather than to flatter it: two of the eight are
# failures of the measurement instrument rather than of a pipeline, and the
# headline conflict case is one where the naive arm returns the right answer
# and still scores zero. Every id is checked against the benchmark by
# `tests/test_artefacts.py`, so a case that is renamed or dropped fails the
# suite rather than the demonstration.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Showcase:
    key: str
    command: str
    dimension: str
    arm: int
    headline: str
    watch: str


SHOWCASE: tuple[Showcase, ...] = (
    Showcase(
        "refusal-hallucination", "case refusal-unans-nq-2842", "RQ1 refusal", 1,
        "The answer passage was removed. Reasoning answers anyway.",
        "Naive and agentic return the sentinel. The reasoning arm commits to "
        "'22' with nothing in the context supporting it -- one false positive, "
        "and the whole cost of Refusal F1's precision term.",
    ),
    Showcase(
        "refusal-over-refusal", "case refusal-ctrl-nq-1131 --generator 2", "RQ1 refusal", 2,
        "An answerable control that two of the three arms refuse.",
        "The other half of Refusal F1. Abstention is only a virtue against the "
        "answerable controls, which is why P1 5.4.3 requires both halves of the "
        "testbed to exist before the metric is defined at all.",
    ),
    Showcase(
        "conflict-reasoning-wins", "case conflict-nq-4643", "RQ2 conflict", 1,
        "Naive commits to one side, reasoning names the disagreement.",
        "CRS 0 against CRS 4 on identical evidence, with the agentic arm "
        "abstaining for a third outcome. The clearest single case for why "
        "accuracy alone cannot answer RQ2.",
    ),
    Showcase(
        "conflict-agentic-wins", "case conflict-nq-780", "RQ2 conflict", 1,
        "The agentic arm reaches the top of the rubric here, and reasoning does not.",
        "Read beside the previous case: the ordering of the three classes "
        "changes from case to case, which is what the aggregate CRS difference "
        "of 0.0342 looks like underneath.",
    ),
    Showcase(
        "conflict-silent-failure", "case conflict-tqa-5864 --generator 2", "RQ2 conflict", 2,
        "The right answer, scored zero. This is the finding I did not design for.",
        "Two arms return the gold answer and neither mentions that the passages "
        "contradict each other: correct under exact match, zero under the "
        "rubric. The third arm sees the disagreement and gets the answer wrong. "
        "One case, and the two metrics rank the arms in opposite orders.",
    ),
    Showcase(
        "noise-ladder", "noise tqa-2481", "RQ3 noise", 1,
        "One question at five noise ratios: the Noise Degradation Curve, one seed wide.",
        "Naive holds to r=0.50 and then abstains; reasoning holds all five; "
        "agentic holds four and breaks at r=0.90. Three shapes of degradation "
        "from one seed.",
    ),
    Showcase(
        "scoring-artefact-swine", "noise nq-1022", "Limitation 15", 1,
        "A curve that goes down and then up again -- because of the scorer, not the model.",
        "'pigs' is scored correct, 'Swine' is not, and the same arm returns "
        "each at different ratios. Limitation 15 in one screen: string matching "
        "understates every arm, and not uniformly.",
    ),
    Showcase(
        "scoring-artefact-midler", "case noise-nq-1419-r90", "Limitation 15", 1,
        "'Midler' scored wrong where 'Bette Midler' is scored right.",
        "The same failure as the previous case in its simplest form. Worth "
        "showing when someone asks how much of the noise degradation is real.",
    ),
)


# --------------------------------------------------------------------------
# Terminal rendering
# --------------------------------------------------------------------------

class Ink:
    """ANSI helpers. Disabled wholesale when the output is not a terminal."""

    def __init__(self, enabled: bool):
        self.on = enabled

    def _wrap(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def bold(self, s: str) -> str:
        return self._wrap("1", s)

    def dim(self, s: str) -> str:
        return self._wrap("2", s)

    def head(self, s: str) -> str:
        return self._wrap("1;36", s)

    def good(self, s: str) -> str:
        return self._wrap("32", s)

    def bad(self, s: str) -> str:
        return self._wrap("31", s)

    def warn(self, s: str) -> str:
        return self._wrap("33", s)

    def key(self, s: str) -> str:
        return self._wrap("35", s)


WIDTH = 96


def rule(ink: Ink, ch: str = "-") -> str:
    return ink.dim(ch * WIDTH)


def wrap(text: str, indent: str = "      ", width: int = WIDTH) -> str:
    body = " ".join((text or "").split())
    return textwrap.fill(body, width=width, initial_indent=indent, subsequent_indent=indent)


def snippet(text: str, limit: int = 240) -> str:
    body = " ".join((text or "").split())
    return body if len(body) <= limit else body[: limit - 1] + "…"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

# Only the fields the demonstration reads. `results.jsonl` is 98 MB and holds
# every reasoning trace; keeping the whole thing parsed costs several hundred
# megabytes for no gain, and a demonstration that swaps on a laptop is not one
# you want to give in front of a panel.
_KEEP = (
    "case_id", "config_id", "final_answer", "raw_text", "retrieved_passage_ids",
    "generator_calls", "steps", "truncated", "error",
)


@dataclass
class Corpus:
    """Everything the demonstration reads, loaded once."""

    cases: list[TestCase]
    by_id: dict[str, TestCase]
    all_passages: list[Any]
    models_cfg: dict[str, Any]
    cache: ResponseCache
    stored: dict[str, dict[str, Any]] = field(default_factory=dict)
    scored: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def controls(self) -> dict[str, Any]:
        return self.models_cfg.get("controls") or {}


def load_corpus(*, with_results: bool = True) -> Corpus:
    cases = [
        TestCase.model_validate_json(line)
        for line in BENCHMARK.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stored: dict[str, dict[str, Any]] = {}
    scored: dict[str, dict[str, Any]] = {}
    if with_results:
        with open(RESULTS, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Append-only: a later row for the same key supersedes an
                # earlier one, which is exactly the resume semantics
                # `ResultStore` applies when it rebuilds its completed index.
                stored[row["key"]] = {k: row.get(k) for k in _KEEP}
        with open(SCORED, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                scored[f"{row['config_id']}::{row['case_id']}"] = row
    return Corpus(
        cases=cases,
        by_id={c.case_id: c for c in cases},
        all_passages=[p for c in cases for p in c.retrieved_passages],
        models_cfg=load_config(MODELS_CONFIG),
        cache=ResponseCache(CACHE_PATH),
        stored=stored,
        scored=scored,
    )


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------


@dataclass
class Replayed:
    config_id: str
    case_id: str
    result: Any | None
    miss: CacheMiss | None
    calls_served: int
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.result is not None


async def replay(corpus: Corpus, case_id: str, config: Configuration) -> Replayed:
    """Run one (case, configuration) pair through the real pipeline, offline."""
    case = corpus.by_id[case_id]
    provider = ReplayProvider.for_generator(corpus.cache, config.generator_key, corpus.models_cfg)
    policy = RetrievalPolicy(
        retriever=config.retriever,
        top_k=int(corpus.controls.get("top_k", 5)),
        max_context_tokens=int(corpus.controls.get("max_context_tokens", 8192)),
        encoder=_encoder_for(config.retriever),
    )
    pool = build_distractor_pool(corpus.all_passages, case.answer, max_size=20)
    sandbox = SandboxedCorpus(case, distractor_pool=pool)
    if config.retriever == "dpr":
        sandbox.index_dense(policy.encoder)
    pipeline = make_pipeline(config, provider, policy, corpus.models_cfg)

    t0 = time.perf_counter()
    try:
        result = await pipeline.run(sandbox)
        miss = None
    except CacheMiss as exc:
        result, miss = None, exc
    return Replayed(
        config_id=config.config_id,
        case_id=case_id,
        result=result,
        miss=miss,
        calls_served=provider.n_hits,
        elapsed_s=time.perf_counter() - t0,
    )


_ENCODER: Any = None


def _encoder_for(retriever: str) -> Any:
    """Load the dense encoder only if a DPR arm is actually requested.

    Deliberately lazy and deliberately not the default. `bm25` needs nothing
    beyond the repository; `dpr` needs `sentence-transformers` to download a
    model, which is a poor thing to discover in a lecture theatre. The DPR arms
    replay perfectly well once the encoder is present -- this only decides when
    the cost is paid.
    """
    global _ENCODER
    if retriever != "dpr":
        return None
    if _ENCODER is None:
        import yaml
        from sentence_transformers import SentenceTransformer

        name = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())["embedding_model"]
        print(f"loading dense encoder {name} (first run downloads it)...", file=sys.stderr)
        _ENCODER = SentenceTransformer(name)
    return _ENCODER


def configs_for(arm: int, retriever: str) -> list[Configuration]:
    """The three pipeline classes on one generator arm, in narrative order."""
    return [
        Configuration(pclass, retriever, template.format(n=arm))
        for pclass, template in ARM_CLASSES.items()
    ]


# --------------------------------------------------------------------------
# `list`
# --------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace, ink: Ink) -> int:
    print()
    print(ink.head("  Curated cases"))
    print(ink.dim("  Each replays offline from data/cache/generators.sqlite. "
                  "No network, no GPU, no API key."))
    print()
    for s in SHOWCASE:
        print(f"  {ink.bold(s.dimension):<28} {ink.key('python scripts/demo.py ' + s.command)}")
        print(wrap(s.headline, indent="    "))
        print(ink.dim(wrap(s.watch, indent="      ")))
        print()
    print(rule(ink))
    print(f"  {ink.key('python scripts/demo.py verify --n 200')}   "
          f"{ink.dim('provenance and cache-coverage check over a random sample')}")
    print()
    return 0


# --------------------------------------------------------------------------
# `case`
# --------------------------------------------------------------------------


def _passage_label(case: TestCase, passage_id: str) -> str:
    for p in case.retrieved_passages:
        if p.passage_id == passage_id:
            if p.is_answer_bearing:
                return "answer-bearing"
            if p.is_injected:
                return "injected"
            return "context"
    # Reached only for an agentic re-retrieval that pulled from the distractor
    # pool, which is the one place a pipeline sees a passage the case did not
    # carry -- and still inside the sandbox (P1 5.6.5).
    return "distractor pool"


def _print_case_header(case: TestCase, ink: Ink) -> None:
    gold = "(none -- unanswerable)" if case.answer == NO_ANSWER else case.answer
    print()
    print(rule(ink, "="))
    print(f"  {ink.head(case.case_id)}   {ink.dim(case.dimension.value)} / "
          f"{ink.dim(case.perturbation_type.value)}"
          + (f"   {ink.dim('noise ratio ' + format(case.noise_ratio, '.2f'))}"
             if case.noise_ratio is not None else ""))
    print(rule(ink, "="))
    print(f"  {ink.bold('Question')}  {case.query}")
    print(f"  {ink.bold('Gold')}      {gold}")
    print(f"  {ink.bold('Evidence')}  {len(case.retrieved_passages)} passage(s) in the case's "
          f"own sandbox, plus a distractor pool drawn from the benchmark")


def _print_evidence(case: TestCase, ink: Ink, full: bool) -> None:
    print()
    for i, p in enumerate(case.retrieved_passages, 1):
        tag = "answer-bearing" if p.is_answer_bearing else ("injected" if p.is_injected else "context")
        colour = ink.good if p.is_answer_bearing else (ink.warn if p.is_injected else ink.dim)
        print(f"    [{i}] {ink.dim(p.passage_id)}  {colour(tag)}  "
              f"{ink.dim(str(p.token_estimate()) + ' tokens')}")
        print(ink.dim(wrap(p.text if full else snippet(p.text, 300))))


def _print_conflict_diff(case: TestCase, ink: Ink) -> None:
    """Show the one claim that differs between the original and the contradiction.

    Worth printing because the two passages are otherwise byte-identical and
    several hundred tokens long, so a reader looking at the raw text cannot see
    what was injected. HANDOFF Section 5 records that conflict generation stays
    passage-level rather than following P1 5.5.2's literal sentence-level
    procedure, on the grounds that the outcome still satisfies P1's "single,
    identifiable claim" requirement. This is that claim, extracted by diffing
    the pair -- so the departure is demonstrable rather than merely asserted.
    """
    import difflib

    original = next((p for p in case.retrieved_passages if p.is_answer_bearing), None)
    injected = next((p for p in case.retrieved_passages if p.is_injected), None)
    if original is None or injected is None:
        return
    a = _sentences(original.text)
    b = _sentences(injected.text)
    diff = [x for x in difflib.unified_diff(a, b, n=0, lineterm="")
            if not x.startswith(("---", "+++", "@@"))]
    removed = [x[1:].strip() for x in diff if x.startswith("-")]
    added = [x[1:].strip() for x in diff if x.startswith("+")]
    if not removed and not added:
        return
    print()
    print(f"  {ink.bold('The injected contradiction')}   "
          f"{ink.dim('diffed against the original passage')}")
    for line in removed[:3]:
        print(ink.good(wrap("original:  " + line, indent="      ")))
    for line in added[:3]:
        print(ink.warn(wrap("injected:  " + line, indent="      ")))


def _sentences(text: str) -> list[str]:
    """Crude sentence split, adequate for showing a diff and nothing else."""
    import re

    parts = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return [x for x in parts if x.strip()]


def _print_arm(rep: Replayed, corpus: Corpus, ink: Ink, show_trace: bool) -> None:
    pclass = rep.config_id.split("|")[0]
    label = {"naive": "NAIVE", "reasoning": "REASONING", "agentic": "AGENTIC"}[pclass]
    print()
    if not rep.ok:
        # Only reachable in the agentic class, and only through a shared entry:
        # the decompose prompt is built from the query alone, so every instance
        # with that query addresses one cache entry and the run's last writer
        # holds it. A chain that starts from another instance's decomposition
        # retrieves different evidence and asks a question this configuration
        # never asked, so the next key is genuinely absent rather than missing.
        print("  " + ink.bold(label.ljust(24)) + " " + ink.dim(rep.config_id))
        print(f"      {ink.warn('chain forked at the shared decompose entry')} "
              f"{ink.dim('(Appendix A.1)')}")
        recorded = (corpus.stored.get(f"{rep.config_id}::{rep.case_id}") or {}).get("final_answer")
        if recorded:
            print(ink.dim(wrap(f"this configuration recorded: {snippet(recorded, 200)}")))
        return

    res = rep.result
    key = f"{rep.config_id}::{rep.case_id}"
    scored = corpus.scored.get(key) or {}
    stored = corpus.stored.get(key) or {}
    answer = res.final_answer
    if answer is None:
        plain, paint = "(no <final_answer> field -- unparseable)", ink.warn
    elif answer.strip() == "INSUFFICIENT EVIDENCE":
        plain, paint = "INSUFFICIENT EVIDENCE   (abstained)", ink.warn
    else:
        plain, paint = snippet(answer, 400), ink.bold

    print("  " + ink.bold(label.ljust(24)) + " " + ink.dim(rep.config_id))
    print(f"      {ink.dim('calls')} {res.generator_calls}  "
          f"{ink.dim('served from cache')} {rep.calls_served}  "
          f"{ink.dim('passages seen')} {len(res.retrieved_passage_ids)}  "
          f"{ink.dim('replay')} {rep.elapsed_s * 1000:.0f} ms")
    # Colour is applied per line, after wrapping: wrapping a string that
    # already carries escape sequences counts them toward the column width.
    for line in wrap(plain, indent="      ").splitlines():
        print(paint(line))

    verdict = []
    if scored.get("answer_correct") is True:
        verdict.append(ink.good("answer correct"))
    elif scored.get("answer_correct") is False:
        verdict.append(ink.bad("answer incorrect"))
    if scored.get("category"):
        verdict.append(ink.dim(f"classified {scored['category']}"))
    crs = scored.get("crs_score")
    if crs is not None:
        paint = ink.good if crs >= 3 else (ink.warn if crs >= 1 else ink.bad)
        verdict.append(paint(f"CRS {crs}") + ink.dim(f" -- {CRS_RUBRIC[crs]}"))
    if verdict:
        print("      " + ink.dim(" | ").join(verdict))

    # The scorer recovers an answer from the surrounding prose when the
    # delimited field is missing (Section 6.5.4); 1,983 responses needed it.
    # Showing the recovery keeps the demonstration's answer and the report's
    # scored answer from silently disagreeing on those instances.
    if res.final_answer is None and scored.get("final_answer"):
        print(ink.dim(f"      recovered by the scorer as {scored['final_answer']!r}"
                      f" ({scored.get('recovery_method') or 'rule'}), Section 6.5.4"))

    if stored.get("raw_text") and stored["raw_text"] != res.raw_text:
        print(ink.dim(wrap("served the entry this run shares with the other retriever arm, "
                           "which ranked this case identically (Appendix A.1); the record for "
                           "this configuration reads:")))
        print(ink.dim(wrap(snippet(stored.get("final_answer") or "(no answer field)", 220),
                           indent="        ")))

    if res.steps:
        for step in res.steps:
            if step.get("node") == "decompose":
                subs = step.get("subqueries") or []
                print(ink.dim(f"      decomposed into {len(subs)} sub-quer"
                              f"{'y' if len(subs) == 1 else 'ies'}:"))
                for s in subs:
                    print(ink.dim(f"        - {s}"))
            elif step.get("node") == "retrieve":
                ids = step.get("passage_ids") or []
                print(ink.dim(f"      re-retrieved for {step.get('query')!r} -> {len(ids)} passage(s)"))
    if show_trace and res.reasoning_trace:
        print(ink.dim("      reasoning trace (retained for qualitative analysis, "
                      "never scored -- P1 5.6.4):"))
        print(ink.dim(wrap(snippet(res.reasoning_trace, 600), indent="        ")))


def _print_provenance(reps: list[Replayed], corpus: Corpus, ink: Ink) -> None:
    served = sum(r.calls_served for r in reps)
    matched = sum(
        1 for r in reps
        if r.ok and (corpus.stored.get(f"{r.config_id}::{r.case_id}") or {}).get("raw_text")
        == r.result.raw_text
    )
    print()
    print(rule(ink))
    line = (f"  {served} generator call(s), all served from the local cache. "
            f"0 network calls -- ReplayProvider holds no client.")
    print(ink.dim(line))
    live = len([r for r in reps if r.ok])
    if matched != live:
        print(ink.dim(f"  {matched} of {live} arms reproduced this configuration's own stored "
                      "record byte for byte. Run `verify` for the accounting."))
    print()


def cmd_case(args: argparse.Namespace, ink: Ink) -> int:
    corpus = load_corpus()
    if args.case_id not in corpus.by_id:
        print(f"unknown case '{args.case_id}'. Try: python scripts/demo.py list", file=sys.stderr)
        return 2
    case = corpus.by_id[args.case_id]
    _print_case_header(case, ink)
    _print_evidence(case, ink, full=args.full_text)
    if case.dimension.value == "conflict":
        _print_conflict_diff(case, ink)
    print()
    print(rule(ink))

    reps: list[Replayed] = []
    for config in configs_for(args.generator, args.retriever):
        rep = asyncio.run(replay(corpus, args.case_id, config))
        reps.append(rep)
        _print_arm(rep, corpus, ink, args.trace)
    _print_provenance(reps, corpus, ink)
    corpus.cache.close()
    return 0


# --------------------------------------------------------------------------
# `noise`
# --------------------------------------------------------------------------

NOISE_RATIOS = ("00", "25", "50", "75", "90")


def cmd_noise(args: argparse.Namespace, ink: Ink) -> int:
    """One seed across all five noise ratios: the Noise Degradation Curve, one seed wide."""
    corpus = load_corpus()
    ids = [f"noise-{args.seed_id}-r{r}" for r in NOISE_RATIOS]
    missing = [c for c in ids if c not in corpus.by_id]
    if missing:
        print(f"seed '{args.seed_id}' has no noise ladder ({missing[0]} not in the benchmark). "
              "Try: python scripts/demo.py list", file=sys.stderr)
        return 2

    case0 = corpus.by_id[ids[0]]
    print()
    print(rule(ink, "="))
    print(f"  {ink.head('noise ladder ' + args.seed_id)}   "
          f"{ink.dim('one question, five noise ratios (P1 5.4.2)')}")
    print(rule(ink, "="))
    print(f"  {ink.bold('Question')}  {case0.query}")
    print(f"  {ink.bold('Gold')}      {case0.answer}")
    print()
    print(ink.dim("  " + "".ljust(21) + "".join(f"  r=0.{r}  ".center(10) for r in NOISE_RATIOS)))

    grid: dict[str, list[Replayed]] = {}
    for config in configs_for(args.generator, args.retriever):
        row: list[Replayed] = []
        cells: list[str] = []
        for case_id in ids:
            rep = asyncio.run(replay(corpus, case_id, config))
            row.append(rep)
            cells.append(_cell(rep, corpus, ink))
        grid[config.pipeline_class] = row
        # Padded before colouring: an escape sequence has width in the string
        # and none on the screen, so `f"{ink.bold(x):<21}"` misaligns the grid.
        print("  " + ink.bold(config.pipeline_class.ljust(21)) + "".join(cells))

    print()
    for pclass, row in grid.items():
        answers = []
        for r, rep in zip(NOISE_RATIOS, row):
            a = rep.result.final_answer if rep.ok else None
            answers.append(f"r=0.{r}: {snippet(a, 46) if a else '(miss)'}")
        print(f"  {ink.dim(pclass)}")
        for a in answers:
            print(ink.dim(f"      {a}"))
    print()
    print(rule(ink))
    print(ink.dim(f"  {sum(r.calls_served for row in grid.values() for r in row)} generator "
                  "call(s), all served from the local cache. 0 network calls."))
    print(ink.dim("  A cell is scored against the gold answer by exact match and token F1 "
                  "(P1 5.6.4). A curve that recovers as noise rises is a scoring artefact, "
                  "not a model that improves -- limitation 15."))
    print()
    corpus.cache.close()
    return 0


def _cell(rep: Replayed, corpus: Corpus, ink: Ink) -> str:
    if not rep.ok:
        return ink.bad("  MISS  ") + "  "
    scored = corpus.scored.get(f"{rep.config_id}::{rep.case_id}") or {}
    correct = scored.get("answer_correct")
    if correct is True:
        return ink.good("   OK   ") + "  "
    if correct is False:
        return ink.bad(" WRONG  ") + "  "
    if (scored.get("category") or "") == "refusal":
        return ink.warn(" ABSTAIN") + "  "
    return ink.dim(" UNPARS ") + "  "


# --------------------------------------------------------------------------
# `verify`
#
# What this can and cannot establish is worth being exact about, because the
# obvious claim -- "replay reproduces the run" -- is not true in general and
# the report says so (Appendix A.1, A.6, A.8).
#
# The cache is keyed on the prompt and the decoding parameters, not on the
# configuration that issued the call. Two configurations that build a
# byte-identical prompt therefore share one entry, and the run wrote both: the
# survivor is whichever finished last. Two ways that happens here. Two
# retriever arms can rank a case's passages into the same order, which makes
# their single generator call identical. And the agentic decompose prompt is
# built from the query alone, so every instance sharing a query shares that
# entry -- all five noise ratios of a seed, both retrievers.
#
# So three checks, in increasing order of what they would catch:
#
#   Provenance. Every response served is one this run recorded, for the same
#     generator. This is the claim that matters: replay invents nothing.
#   Coverage. Which calls were served, and where a miss occurred. A miss is
#     only possible where an earlier shared entry forked the chain, so it can
#     only occur in the agentic class.
#   Reproduction. Instances whose entry is provably unshared -- the two
#     retriever arms ranked the case differently -- must reproduce their stored
#     record byte for byte, with no exceptions permitted.
# --------------------------------------------------------------------------


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _provenance_index(stored: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    """Every response text this run recorded, hashed, per generator."""
    index: dict[str, set[str]] = defaultdict(set)
    for key, row in stored.items():
        generator = key.split("::")[0].split("|")[2]
        if row.get("raw_text"):
            index[generator].add(_sha(row["raw_text"]))
        for step in row.get("steps") or []:
            if step.get("output"):
                index[generator].add(_sha(step["output"]))
    return index


def _sibling(key: str) -> str:
    return key.replace("|bm25|", "|dpr|") if "|bm25|" in key else key.replace("|dpr|", "|bm25|")


def _entry_is_shared(key: str, stored: dict[str, dict[str, Any]]) -> bool | None:
    """Whether this single-call instance shares its cache entry with the other arm.

    Decided from the recorded retrieval order, which is what the prompt is
    built from: same ordered passage ids under the same generator means the two
    retriever arms issued a byte-identical request and therefore addressed one
    entry. Returns None when the sibling arm has no record to compare against.
    """
    sibling = stored.get(_sibling(key))
    if sibling is None:
        return None
    own = stored[key]
    return list(sibling.get("retrieved_passage_ids") or []) == list(
        own.get("retrieved_passage_ids") or []
    )


def _decompose_served(corpus: Corpus, case: TestCase, generator: str) -> str | None:
    """The decompose response the cache holds for this case's query and arm."""
    spec = corpus.models_cfg["generators"][generator]
    provider = ReplayProvider.for_generator(corpus.cache, generator, corpus.models_cfg)
    req = GenerationRequest(
        prompt=build_decompose_prompt(case.query, max_subqueries=3),
        max_tokens=int(corpus.controls.get("max_output_tokens", 4096)),
        temperature=float(corpus.controls.get("temperature", 0.0)),
        seed=corpus.controls.get("seed"),
        thinking=bool(spec.get("thinking", False)),
    )
    hit = corpus.cache.get(provider.key_for(req))
    return hit.text if hit else None


def _own_decompose(row: dict[str, Any]) -> str | None:
    for step in row.get("steps") or []:
        if step.get("node") == "decompose":
            return step.get("output")
    return None


def cmd_verify(args: argparse.Namespace, ink: Ink) -> int:
    import random

    t0 = time.perf_counter()
    corpus = load_corpus()
    provenance = _provenance_index(corpus.stored)

    pool = [
        k for k in corpus.stored
        if f"|{args.retriever}|" in k and corpus.stored[k].get("raw_text")
    ]
    if not pool:
        print(f"no stored instances for retriever '{args.retriever}'", file=sys.stderr)
        return 2
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n, len(pool)))

    tally: Counter[str] = Counter()
    violations: list[str] = []
    forked: list[str] = []
    calls = 0

    print()
    print(rule(ink, "="))
    print(f"  {ink.head('verify')}   {len(sample)} instances, retriever={args.retriever}, "
          f"sample seed {args.seed}")
    print(rule(ink, "="))

    for i, key in enumerate(sample, 1):
        config_id, case_id = key.split("::")
        pclass, retriever, generator = config_id.split("|")
        rep = asyncio.run(replay(corpus, case_id, Configuration(pclass, retriever, generator)))
        calls += rep.calls_served
        row = corpus.stored[key]
        tally[f"class:{pclass}"] += 1

        if not rep.ok:
            tally["miss"] += 1
            served = _decompose_served(corpus, corpus.by_id[case_id], generator)
            if pclass != "agentic":
                violations.append(f"{key}: cache miss in a single-call class")
            elif served is not None and served != _own_decompose(row):
                # The chain forked at the shared decompose entry, so the next
                # prompt was one this configuration never issued. Explained.
                forked.append(key)
            else:
                violations.append(f"{key}: cache miss with no shared-entry explanation")
            continue

        text = rep.result.raw_text
        if _sha(text) not in provenance[generator]:
            violations.append(f"{key}: served a response this run never recorded")
        else:
            tally["provenance_ok"] += 1

        exact = text == row.get("raw_text")
        tally["exact" if exact else "differs"] += 1

        if pclass != "agentic":
            shared = _entry_is_shared(key, corpus.stored)
            if shared is False:
                tally["unshared"] += 1
                if not exact:
                    violations.append(
                        f"{key}: unshared entry did not reproduce its stored record"
                    )
                else:
                    tally["unshared_exact"] += 1
            elif shared is True:
                tally["shared"] += 1
                sibling = corpus.stored.get(_sibling(key)) or {}
                if exact or text == sibling.get("raw_text"):
                    tally["shared_accounted"] += 1
                else:
                    violations.append(
                        f"{key}: shared entry matched neither arm's stored record"
                    )
        if args.progress and i % args.progress == 0:
            print(ink.dim(f"    {i}/{len(sample)}..."))

    ok = len(sample) - tally["miss"]
    elapsed = time.perf_counter() - t0
    print()
    print(f"  {ink.bold('Coverage')}")
    print(f"      {calls} generator call(s) served from data/cache/generators.sqlite")
    print(f"      {ink.good('0')} network calls -- ReplayProvider imports no HTTP client "
          f"and raises on a miss")
    print(f"      {ok}/{len(sample)} instances replayed to a final answer; "
          f"{tally['miss']} could not")
    if forked:
        print(ink.dim(f"      all {len(forked)} were agentic chains that forked at the "
                      "shared decompose entry, which is keyed on the query alone"))
    print()
    print(f"  {ink.bold('Provenance')}   {ink.dim('the claim that matters')}")
    print(f"      {tally['provenance_ok']}/{ok} replayed responses are byte-identical to a "
          f"response this run recorded")
    print()
    print(f"  {ink.bold('Reproduction')}")
    print(f"      {tally['exact']}/{ok} reproduced this configuration's own stored record exactly "
          f"({100 * tally['exact'] / max(ok, 1):.1f}%)")
    if tally["unshared"]:
        print(f"      {tally['unshared_exact']}/{tally['unshared']} single-call instances whose "
              f"cache entry is provably unshared reproduced it exactly")
    if tally["shared"]:
        print(f"      {tally['shared_accounted']}/{tally['shared']} single-call instances that "
              f"share an entry with the other retriever arm returned one of the two "
              f"recorded responses")
    print()
    print(rule(ink))
    if violations:
        print(f"  {ink.bad('FAILED')}  {len(violations)} unexplained outcome(s):")
        for v in violations[:12]:
            print(ink.bad(f"      {v}"))
        if len(violations) > 12:
            print(ink.bad(f"      ... and {len(violations) - 12} more"))
        print()
        corpus.cache.close()
        return 1
    print(f"  {ink.good('PASSED')}  every response served came out of this run, and every "
          f"departure from the stored record is accounted for.")
    print(ink.dim(f"  {elapsed:.1f}s wall clock, offline throughout."))
    print()
    corpus.cache.close()
    return 0


# --------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        prog="demo.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--no-color", action="store_true", help="plain output, for piping or slides")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    # Accepted after the subcommand as well as before it. A presenter types
    # `demo.py verify --no-color`, not `demo.py --no-color verify`, and an
    # argparse usage error is a poor thing to project. SUPPRESS so that
    # omitting it here does not overwrite the top-level flag with a default.
    common.add_argument("--no-color", action="store_true", default=argparse.SUPPRESS,
                        help="plain output, for piping or slides")
    common.add_argument("--generator", type=int, choices=(1, 2), default=1,
                        help="generator arm: 1 or 2 (see configs/models.yaml)")
    common.add_argument("--retriever", choices=("bm25", "dpr"), default="bm25",
                        help="dpr downloads a sentence-transformers model on first use")

    sub.add_parser("list", help="the curated cases").set_defaults(fn=cmd_list)

    c = sub.add_parser("case", parents=[common], help="three pipeline classes on one case")
    c.add_argument("case_id")
    c.add_argument("--trace", action="store_true",
                   help="print reasoning traces (retained, never scored -- P1 5.6.4)")
    c.add_argument("--full-text", action="store_true", help="print passages in full")
    c.set_defaults(fn=cmd_case)

    n = sub.add_parser("noise", parents=[common], help="one seed across all five noise ratios")
    n.add_argument("seed_id", help="e.g. tqa-2481 (not the full case id)")
    n.set_defaults(fn=cmd_noise)

    v = sub.add_parser("verify", parents=[common], help="provenance and cache-coverage check")
    v.add_argument("--n", type=int, default=200, help="instances to replay (default 200)")
    v.add_argument("--seed", type=int, default=20260721, help="sampling seed")
    v.add_argument("--progress", type=int, default=0, help="print progress every N instances")
    v.set_defaults(fn=cmd_verify)

    args = p.parse_args()
    ink = Ink(enabled=sys.stdout.isatty() and not getattr(args, "no_color", False))
    return args.fn(args, ink)


if __name__ == "__main__":
    raise SystemExit(main())
