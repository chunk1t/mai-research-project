"""Experiment matrix expansion and resumable execution bookkeeping (P1 5.6.5).

    "The full experimental matrix consists of three pipeline classes by two
    retrievers by two generators, yielding twelve primary configurations. Each
    configuration is run against the entire benchmark."

This module holds only the parts that can be decided without a model: which
configurations exist, which (case, configuration) pairs still need running, and
how a finished pair is recorded. Execution itself lives in the run script, so
the matrix and the resume logic stay unit-testable with no GPU and no network.

Resumability is a correctness property here, not a convenience (P1 5.8). The
full matrix is ~34,800 instances and ~70,000 generator calls; a run that cannot
resume is one interruption away from being unaffordable. The response cache
already prevents paying twice for an identical call, but it cannot tell you
which pairs are outstanding -- that is what the result store's completed-key
index is for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from .schema import TestCase

PIPELINE_CLASSES = ("naive", "reasoning", "agentic")


@dataclass(frozen=True)
class Configuration:
    """One cell of the P1 Table 5.5 matrix."""

    pipeline_class: str
    retriever: str
    generator_key: str

    @property
    def config_id(self) -> str:
        return f"{self.pipeline_class}|{self.retriever}|{self.generator_key}"

    def as_dict(self) -> dict[str, str]:
        return {
            "config_id": self.config_id,
            "pipeline_class": self.pipeline_class,
            "retriever": self.retriever,
            "generator_key": self.generator_key,
        }


def expand_matrix(models_cfg: Mapping[str, Any]) -> list[Configuration]:
    """Build the twelve configurations from `configs/models.yaml`.

    Generator selection per class follows P1 Section 5.6: the naive class uses
    the standard generators, the reasoning class the reasoning generators, and
    the agentic class `agentic_generators`, which P1 Section 5.6.3 requires to
    be a subset of the reasoning set. That subset relation is enforced here
    rather than assumed, because an agentic arm running a standard generator
    would silently compare orchestration against a different base model and
    confound the study's central contrast.
    """
    generators = models_cfg.get("generators") or {}
    standard = sorted(k for k, g in generators.items() if k.startswith("standard"))
    reasoning = sorted(k for k, g in generators.items() if k.startswith("reasoning"))
    agentic = list(models_cfg.get("agentic_generators") or [])
    retrievers = list(models_cfg.get("retrievers") or [])

    if not retrievers:
        raise ValueError("models.yaml defines no retrievers")
    missing = [k for k in agentic if k not in generators]
    if missing:
        raise ValueError(f"agentic_generators names unknown generators: {missing}")
    not_reasoning = [k for k in agentic if k not in reasoning]
    if not_reasoning:
        raise ValueError(
            f"agentic_generators must be a subset of the reasoning generators "
            f"(P1 5.6.3); {not_reasoning} are not"
        )

    by_class = {"naive": standard, "reasoning": reasoning, "agentic": agentic}
    configs: list[Configuration] = []
    for pclass in PIPELINE_CLASSES:
        for retriever in retrievers:
            for gen in by_class[pclass]:
                configs.append(Configuration(pclass, retriever, gen))
    return configs


@dataclass(frozen=True)
class WorkItem:
    case: TestCase
    config: Configuration

    @property
    def key(self) -> str:
        return f"{self.config.config_id}::{self.case.case_id}"


class ResultStore:
    """Append-only JSONL results with an in-memory completed-key index.

    Append-only rather than rewritten, so a crash mid-write costs one row rather
    than the whole file. The index is rebuilt by scanning on open, which is
    cheap next to a run measured in GPU-hours and means the file on disk is the
    single source of truth about what has been done -- there is no separate
    checkpoint to fall out of sync with it.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        self._errored: set[str] = set()
        self._malformed = 0
        if self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        # A row truncated by a crash. Counted and skipped so the
                        # pair is simply re-run, never silently treated as done.
                        self._malformed += 1
                        continue
                    key = row.get("key")
                    if not key:
                        continue
                    if row.get("error"):
                        # An errored instance is NOT complete. Transient
                        # failures happen -- a cold container, a stale replica
                        # after a redeploy, a read timeout -- and marking them
                        # done would make a single blip permanent across every
                        # future resume of a 34,000-instance run.
                        self._errored.add(key)
                        self._done.discard(key)
                    else:
                        self._done.add(key)
                        self._errored.discard(key)
        self._fh = open(self.path, "a", encoding="utf-8")

    @property
    def completed(self) -> set[str]:
        return set(self._done)

    @property
    def malformed_rows(self) -> int:
        return self._malformed

    @property
    def errored(self) -> set[str]:
        """Keys whose latest row carries an error; these are retried on resume."""
        return set(self._errored)

    def is_done(self, item: WorkItem) -> bool:
        return item.key in self._done

    def record(self, item: WorkItem, payload: dict[str, Any]) -> None:
        row = {
            "key": item.key,
            "case_id": item.case.case_id,
            "dimension": item.case.dimension.value,
            "noise_ratio": item.case.noise_ratio,
            **item.config.as_dict(),
            **payload,
        }
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()
        # Only a clean result completes the pair; see the loader above. The row
        # is still written either way, so a failure is visible and countable
        # rather than silently absent.
        if payload.get("error"):
            self._errored.add(item.key)
        else:
            self._done.add(item.key)
            self._errored.discard(item.key)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def pending_work(
    cases: list[TestCase],
    configs: list[Configuration],
    store: ResultStore,
    *,
    interleave_by_generator: bool = True,
) -> Iterator[WorkItem]:
    """Yield only the (case, configuration) pairs not already recorded.

    Within a configuration, work stays case-ordered so a partial configuration
    is still a contiguous prefix of the benchmark.

    ACROSS configurations, work round-robins between generator models by
    default. Strict configuration-major order keeps a partial result set tidy,
    but it hits ONE model at a time: the four Qwen arms run while the GLM
    containers idle, and vice versa. With five containers per model against a
    ten-GPU workspace limit, that halves the fleet for the whole run -- measured
    at ~0.2-0.36 instances/sec sustained, which is ~12 hours for the full
    matrix. Alternating keeps both models busy.

    The cost is that an interruption leaves several configurations part-done
    rather than a few finished. That is affordable precisely because the store
    resumes per (case, configuration) pair, so no partial work is lost.
    """
    def items_for(config: Configuration) -> Iterator[WorkItem]:
        for case in cases:
            item = WorkItem(case, config)
            if not store.is_done(item):
                yield item

    if not interleave_by_generator:
        for config in configs:
            yield from items_for(config)
        return

    # Group configurations by the model they call, then round-robin the groups.
    groups: dict[str, list[Configuration]] = {}
    for c in configs:
        groups.setdefault(c.generator_key, []).append(c)
    streams = [items_for(c) for keyed in groups.values() for c in keyed]
    # Order the streams so consecutive ones use different generators.
    ordered: list[Iterator[WorkItem]] = []
    per_key = {k: [items_for(c) for c in v] for k, v in groups.items()}
    while any(per_key.values()):
        for k in list(per_key):
            if per_key[k]:
                ordered.append(per_key[k].pop(0))
    exhausted = set()
    while len(exhausted) < len(ordered):
        for i, stream in enumerate(ordered):
            if i in exhausted:
                continue
            try:
                yield next(stream)
            except StopIteration:
                exhausted.add(i)


def run_plan(cases: list[TestCase], configs: list[Configuration], store: ResultStore) -> dict:
    """Summary of what a run would do, for printing before it starts."""
    total = len(cases) * len(configs)
    pending = sum(1 for _ in pending_work(cases, configs, store))
    return {
        "n_cases": len(cases),
        "n_configurations": len(configs),
        "total_instances": total,
        "already_done": total - pending,
        "pending": pending,
        "retrying_after_error": len(store.errored),
        "malformed_rows_skipped": store.malformed_rows,
    }


def smoke_subset(cases: list[TestCase], n: int) -> list[TestCase]:
    """A miniature of the benchmark, not just its first N cases per dimension.

    Taking the head of each dimension produced a slice that could not exercise
    the metrics it was meant to validate: all five refusal cases came back
    answerable controls, so Refusal F1 reported 0.0 for every configuration
    because recall was undefined, and all five noise cases were the five ratios
    of ONE seed, so the Noise Degradation Curve was fitted through a single
    case.

    So: refusal is split evenly between unanswerable cases and answerable
    controls (P1 5.4.3's balance), noise takes `n` whole seeds with every ratio
    each (the curve needs all five points per seed), and conflict takes `n`
    cases. Selection is by sorted case id, so a smoke slice is reproducible and
    a resumed smoke run reuses the same cache entries.
    """
    from .schema import NO_ANSWER, Dimension

    by_dim: dict[Dimension, list[TestCase]] = {}
    for c in sorted(cases, key=lambda c: c.case_id):
        by_dim.setdefault(c.dimension, []).append(c)

    out: list[TestCase] = []

    refusal = by_dim.get(Dimension.REFUSAL, [])
    unans = [c for c in refusal if c.answer == NO_ANSWER]
    ctrl = [c for c in refusal if c.answer != NO_ANSWER]
    half = max(1, n // 2)
    out += unans[:half] + ctrl[: n - min(half, len(unans))]

    out += by_dim.get(Dimension.CONFLICT, [])[:n]

    noise = by_dim.get(Dimension.NOISE, [])
    seeds: list[str] = []
    for c in noise:
        if c.seed_id not in seeds:
            seeds.append(c.seed_id)
        if len(seeds) >= n:
            break
    chosen = set(seeds)
    out += [c for c in noise if c.seed_id in chosen]
    return out
