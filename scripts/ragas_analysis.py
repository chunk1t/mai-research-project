#!/usr/bin/env python3
"""RAGAS secondary analysis over a held-out subset (P1 Section 5.7).

    python scripts/ragas_analysis.py select     # choose the held-out subset
    python scripts/ragas_analysis.py score      # run the RAGAS judge over it
    python scripts/ragas_analysis.py analyse    # correlations and coverage gap

P1 5.7 poses two questions and this answers both:

  1. Do RAGAS scores CORRELATE with the purpose-built metrics? If they do,
     reference-free monitoring already captures these failure modes.
  2. Are there SYSTEMATIC cases where RAGAS misses failures the purpose-built
     metrics catch? If there are, that is a coverage gap in the existing tools.

The stages are separate because the middle one costs money and the outer two do
not. `select` is deterministic from the config seed, `score` is cached and
resumable, and `analyse` can be re-run freely while Chapter 7 is drafted.

The judge is `ragas_judge` from configs/models.yaml, a google-family model.
That is a deliberate, documented departure from strict role disjointness: P1
5.9 names three roles -- case generator, evaluated generators, CRS judge -- and
RAGAS is not among them, but google also writes the conflict contradictions.
`providers/factory.py` enforces the constraint that DOES bind here, namely that
the RAGAS judge must not share a family with any evaluated generator, since
RAGAS is the baseline those generators are being measured against.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragrobust.cache import ResponseCache  # noqa: E402
from ragrobust.providers.base import GenerationRequest  # noqa: E402
from ragrobust.providers.cached import CachedProvider  # noqa: E402
from ragrobust.providers.factory import build_provider  # noqa: E402
from ragrobust.ragas import (  # noqa: E402
    answer_relevance_score,
    build_context_relevance_prompt,
    build_question_prompt,
    build_statement_prompt,
    build_verdict_prompt,
    context_relevance_score,
    coverage_gap,
    faithfulness_score,
    parse_generated_questions,
    parse_selected_sentences,
    parse_statements,
    parse_verdicts,
    spearman,
    split_sentences,
    RagasScores,
)

RUNS = ROOT / "runs"
SUBSET = RUNS / "ragas_subset.jsonl"
SCORES = RUNS / "ragas_scores.jsonl"
REPORT = RUNS / "ragas_analysis.json"
CACHE_PATH = ROOT / "data" / "cache" / "ragas.sqlite"

PIPELINE_CLASSES = ("naive", "reasoning", "agentic")
DIMENSIONS = ("refusal", "conflict", "noise")


# --------------------------------------------------------------------------
# Stage 1: choose the held-out subset
# --------------------------------------------------------------------------


def load_validation_case_ids() -> set[str]:
    """Case ids already consumed by a human-labelled sample.

    Excluding them is what makes the subset HELD OUT in the sense P1 5.7 needs.
    A case that calibrated the CRS judge has already shaped the purpose-built
    score it would be compared against here, so including it would let the two
    sides of the comparison share an error instead of being independent.
    """
    ids: set[str] = set()
    for name in ("primary_labels.csv", "second_annotator_labels.csv",
                 "crs_judge_scores.csv"):
        path = RUNS / "validation" / name
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("case_id"):
                    ids.add(row["case_id"])
    return ids


def do_select(args) -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())["ragas"]
    per_stratum = int(cfg["instances_per_stratum"])
    rng = random.Random(int(cfg["rng_seed"]))

    excluded = load_validation_case_ids() if cfg.get("exclude_validation_cases") else set()
    print(f"excluding {len(excluded)} case id(s) already used in a human-labelled sample")

    # Strata are (dimension, pipeline class). Configuration is NOT a stratum:
    # the comparison is between RAGAS and the purpose-built metrics, and the
    # class is what P1 5.7's comparison is reported over.
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_rows = 0
    with open(ROOT / args.scored, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            if row["case_id"] in excluded:
                continue
            cls = row["config_id"].split("|")[0]
            strata[(row["dimension"], cls)].append(row)

    picked: list[dict] = []
    shortfalls: list[str] = []
    for dim in DIMENSIONS:
        for cls in PIPELINE_CLASSES:
            pool = strata.get((dim, cls), [])
            # Sorted before shuffling so the draw does not depend on file order.
            pool.sort(key=lambda r: (r["config_id"], r["case_id"]))
            rng.shuffle(pool)
            take = pool[:per_stratum]
            if len(take) < per_stratum:
                shortfalls.append(
                    f"{dim}/{cls}: {len(take)} of {per_stratum} available"
                )
            picked.extend(take)

    picked.sort(key=lambda r: (r["dimension"], r["config_id"], r["case_id"]))
    with open(SUBSET, "w", encoding="utf-8") as fh:
        for row in picked:
            fh.write(json.dumps(row) + "\n")

    print(f"read {n_rows} scored instances")
    print(f"selected {len(picked)} across {len(DIMENSIONS) * len(PIPELINE_CLASSES)} strata")
    for dim in DIMENSIONS:
        counts = {c: sum(1 for r in picked
                         if r["dimension"] == dim and r["config_id"].startswith(c))
                  for c in PIPELINE_CLASSES}
        print(f"  {dim:9s} {counts}")
    if shortfalls:
        # Never silently return fewer: a thinned stratum changes what the miss
        # rate is computed over.
        print("\nSHORTFALL -- these strata could not be filled:")
        for s in shortfalls:
            print(f"  {s}")
    print(f"\n-> {SUBSET}")
    return 0


# --------------------------------------------------------------------------
# Stage 2: score the subset with the RAGAS judge
# --------------------------------------------------------------------------


def load_passages() -> dict[str, str]:
    """passage_id -> text over the WHOLE benchmark.

    Global rather than per-case because the agentic pipeline re-retrieves from
    the controlled distractor pool, which `run_experiment.py` assembles from
    every case's passages. A per-case map leaves ~20% of agentic passage
    references unresolved.
    """
    out: dict[str, str] = {}
    with open(ROOT / "data" / "cases" / "benchmark.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            case = json.loads(line)
            for p in case["retrieved_passages"]:
                out.setdefault(p["passage_id"], p["text"])
    return out


def load_contexts(keys: set[str]) -> dict[str, tuple[str, list[str]]]:
    """key -> (query, passage_ids) for the instances in the subset."""
    out: dict[str, tuple[str, list[str]]] = {}
    queries: dict[str, str] = {}
    with open(ROOT / "data" / "cases" / "benchmark.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                case = json.loads(line)
                queries[case["case_id"]] = case["query"]
    with open(RUNS / "results.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["key"] in keys:
                out[row["key"]] = (
                    queries.get(row["case_id"], ""),
                    list(row.get("retrieved_passage_ids") or []),
                )
    return out


async def score_one(judge, encoder, query, context, answer, n_questions) -> RagasScores:
    """The three RAGAS metrics for one instance, per Es et al. (2024) Section 3."""
    scores = RagasScores()

    # --- Faithfulness: decompose, then verify each statement. Two stages, as
    # in the paper; collapsing them into one call would let the model decide
    # what to assert and whether it is supported in the same breath.
    st = await judge.generate(GenerationRequest(
        prompt=build_statement_prompt(query, answer), max_tokens=512, temperature=0.0))
    statements = parse_statements(st.text) if st.ok else []
    scores.n_statements = len(statements)
    if not st.ok:
        scores.notes.append(f"statement extraction failed: {st.error}")
    elif not statements:
        scores.notes.append(
            "answer asserts no verifiable statement (refusal or empty); "
            "faithfulness is 0/0 and undefined, NOT zero"
        )
    else:
        vd = await judge.generate(GenerationRequest(
            prompt=build_verdict_prompt(context, statements),
            max_tokens=256, temperature=0.0))
        verdicts = parse_verdicts(vd.text, len(statements)) if vd.ok else None
        if verdicts is None:
            scores.notes.append("verdicts unparseable or incomplete")
        else:
            scores.n_supported = sum(1 for v in verdicts if v)
            scores.faithfulness = faithfulness_score(verdicts)

    # --- Answer relevance: generate questions from the answer, embed, average.
    qg = await judge.generate(GenerationRequest(
        prompt=build_question_prompt(answer, n_questions),
        max_tokens=256, temperature=0.0))
    questions = parse_generated_questions(qg.text) if qg.ok else []
    if not answer.strip():
        questions = []
    scores.n_generated_questions = len(questions)
    if questions:
        embs = encoder.encode([query] + questions)
        scores.answer_relevance = answer_relevance_score(
            [float(x) for x in embs[0]],
            [[float(x) for x in row] for row in embs[1:]],
        )
    else:
        scores.notes.append("no questions could be generated from the answer")

    # --- Context relevance: select the sentences needed to answer the question.
    sentences = split_sentences(context)
    scores.n_context_sentences = len(sentences)
    if sentences:
        cr = await judge.generate(GenerationRequest(
            prompt=build_context_relevance_prompt(query, sentences),
            max_tokens=256, temperature=0.0))
        selected = parse_selected_sentences(cr.text, len(sentences)) if cr.ok else []
        if not cr.ok:
            scores.notes.append(f"context relevance call failed: {cr.error}")
        else:
            scores.n_relevant_sentences = len(selected)
            scores.context_relevance = context_relevance_score(
                len(selected), len(sentences))
    return scores


async def do_score_async(args) -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())["ragas"]
    models_cfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())

    rows = [json.loads(l) for l in open(SUBSET, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    keys = {f"{r['config_id']}::{r['case_id']}" for r in rows}

    # Resume: anything already scored is skipped, so an interrupted run costs
    # nothing to restart (P1 5.8).
    already: set[str] = set()
    if SCORES.exists() and not args.overwrite:
        already = {json.loads(l)["key"] for l in open(SCORES, encoding="utf-8") if l.strip()}
    todo = [r for r in rows if f"{r['config_id']}::{r['case_id']}" not in already]
    print(f"{len(rows)} in subset, {len(already)} already scored, {len(todo)} to do")
    if not todo:
        print("nothing to do")
        return 0

    print("loading passages and contexts...")
    passages = load_passages()
    contexts = load_contexts(keys)

    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    ds_cfg = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())
    print(f"loading encoder {ds_cfg['embedding_model']}...")
    encoder = SentenceTransformer(ds_cfg["embedding_model"])

    spec = models_cfg["ragas_judge"]
    judge = CachedProvider(
        build_provider("ragas_judge", spec, models_cfg), ResponseCache(CACHE_PATH))
    await judge.open()
    print(f"RAGAS judge: {spec['model']} (family {spec['family']})")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = 0
    started = time.time()
    out_fh = open(SCORES, "a" if already else "w", encoding="utf-8")

    async def one(row) -> None:
        nonlocal done
        key = f"{row['config_id']}::{row['case_id']}"
        query, pids = contexts.get(key, ("", []))
        context = "\n\n".join(passages[p] for p in pids if p in passages)
        async with sem:
            scores = await score_one(
                judge, encoder, query, context, row.get("final_answer") or "",
                int(cfg["n_generated_questions"]))
        async with lock:
            done += 1
            out_fh.write(json.dumps({
                "key": key,
                "case_id": row["case_id"],
                "config_id": row["config_id"],
                "dimension": row["dimension"],
                "pipeline_class": row["config_id"].split("|")[0],
                **scores.as_dict(),
            }) + "\n")
            out_fh.flush()
            if done % 25 == 0 or done == len(todo):
                rate = done / max(1e-9, time.time() - started)
                print(f"  {done}/{len(todo)}  {rate:.2f}/s  "
                      f"eta {(len(todo) - done) / max(1e-9, rate) / 60:.1f} min")

    await asyncio.gather(*(one(r) for r in todo))
    out_fh.close()
    await judge.close()
    print(f"\n-> {SCORES}")
    return 0


def do_score(args) -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("GOOGLE_API_KEY unset", file=sys.stderr)
        return 2
    if not SUBSET.exists():
        print(f"{SUBSET} missing; run `select` first", file=sys.stderr)
        return 2
    return asyncio.run(do_score_async(args))


# --------------------------------------------------------------------------
# Stage 3: the comparison
# --------------------------------------------------------------------------

# The purpose-built verdict per dimension. Each is the SAME judgement the
# headline metric is built from, so the comparison is against what the study
# actually reports rather than a proxy invented for this table.
#
#   refusal   the pipeline behaved correctly on this instance: abstained on an
#             unanswerable case, or answered an answerable control correctly.
#   conflict  CRS > 0, i.e. the response acknowledged the disagreement at all.
#   noise     the answer was correct.


def purpose_built_failure(row: dict) -> bool | None:
    dim = row["dimension"]
    if dim == "refusal":
        if row.get("is_answerable"):
            return not bool(row.get("answer_correct"))
        # Unanswerable: correct behaviour is abstention.
        return row.get("category") != "refusal"
    if dim == "conflict":
        crs = row.get("crs_score")
        return None if crs is None else crs == 0
    if dim == "noise":
        ac = row.get("answer_correct")
        return None if ac is None else not ac
    return None


def do_analyse(args) -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "dataset.yaml").read_text())["ragas"]
    threshold = float(cfg["pass_threshold"])

    scored_by_key = {}
    with open(ROOT / args.scored, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                scored_by_key[f"{r['config_id']}::{r['case_id']}"] = r

    ragas_rows = [json.loads(l) for l in open(SCORES, encoding="utf-8") if l.strip()]
    print(f"{len(ragas_rows)} RAGAS-scored instances")

    metrics = ("faithfulness", "answer_relevance", "context_relevance")
    report: dict = {
        "n_instances": len(ragas_rows),
        "pass_threshold": threshold,
        "judge": yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
                     ["ragas_judge"],
        "coverage": {},
        "correlations": {},
        "coverage_gap": {},
        "descriptive": {},
    }

    # How often each RAGAS metric could be computed at all. A metric that is
    # undefined on most of a dimension has already answered P1 5.7's question
    # for that dimension.
    for dim in DIMENSIONS:
        rows = [r for r in ragas_rows if r["dimension"] == dim]
        if not rows:
            continue
        report["coverage"][dim] = {
            m: {
                "n": len(rows),
                "n_defined": sum(1 for r in rows if r.get(m) is not None),
                "fraction_defined": round(
                    sum(1 for r in rows if r.get(m) is not None) / len(rows), 4),
            }
            for m in metrics
        }
        report["descriptive"][dim] = {
            m: (round(sum(r[m] for r in rows if r.get(m) is not None)
                      / max(1, sum(1 for r in rows if r.get(m) is not None)), 4)
                if any(r.get(m) is not None for r in rows) else None)
            for m in metrics
        }

    # P1 5.7 question 1: do RAGAS scores correlate with the purpose-built ones?
    for dim in DIMENSIONS:
        rows = [r for r in ragas_rows if r["dimension"] == dim]
        per_metric = {}
        for m in metrics:
            xs, ys = [], []
            for r in rows:
                s = r.get(m)
                base = scored_by_key.get(r["key"])
                if s is None or base is None:
                    continue
                if dim == "conflict":
                    target = base.get("crs_score")
                else:
                    fail = purpose_built_failure(base)
                    target = None if fail is None else (0.0 if fail else 1.0)
                if target is None:
                    continue
                xs.append(float(s))
                ys.append(float(target))
            rho = spearman(xs, ys)
            per_metric[m] = {
                "spearman_rho": None if rho is None else round(rho, 4),
                "n_pairs": len(xs),
                "purpose_built_metric": (
                    "CRS (0-4)" if dim == "conflict"
                    else "correct behaviour (1) vs failure (0)"),
            }
        report["correlations"][dim] = per_metric

    # P1 5.7 question 2: does RAGAS miss the failures the metrics catch?
    for dim in DIMENSIONS:
        rows = [r for r in ragas_rows if r["dimension"] == dim]
        per_metric = {}
        for m in metrics:
            scores, failed = [], []
            for r in rows:
                base = scored_by_key.get(r["key"])
                if base is None:
                    continue
                fail = purpose_built_failure(base)
                if fail is None:
                    continue
                scores.append(r.get(m))
                failed.append(fail)
            if not scores:
                continue
            gap = coverage_gap(scores, failed, metric=m, dimension=dim,
                               threshold=threshold)
            per_metric[m] = gap.as_dict()
        report["coverage_gap"][dim] = per_metric

    # P1 4.6.2's first prediction, tested directly: "a model that fabricates an
    # answer from insufficient context can still receive a high faithfulness
    # score". The sharper version this data shows is that RAGAS cannot score a
    # correct ABSTENTION at all -- faithfulness is 0/0 on a refusal -- while
    # answer relevance still returns a middling number for it, because the
    # question generator invents questions from the sentinel string.
    abstained, committed = [], []
    for r in ragas_rows:
        base = scored_by_key.get(r["key"])
        if base is None:
            continue
        (abstained if base.get("category") == "refusal" else committed).append(r)

    def summarise(rows: list[dict]) -> dict:
        out: dict = {"n": len(rows)}
        for m in metrics:
            defined = [r[m] for r in rows if r.get(m) is not None]
            out[m] = {
                "n_defined": len(defined),
                "fraction_defined": round(len(defined) / len(rows), 4) if rows else None,
                "mean_where_defined": (
                    round(sum(defined) / len(defined), 4) if defined else None),
            }
        return out

    report["abstention"] = {
        "note": (
            "Instances where the pipeline abstained, against those where it "
            "committed to an answer. Faithfulness is 0/0 on an abstention and "
            "is reported as undefined rather than coerced; the fraction_defined "
            "row is therefore the measurement, not a data-quality figure."
        ),
        "abstained": summarise(abstained),
        "committed": summarise(committed),
    }

    # Does RAGAS track the noise ratio? This is the fairest test available to
    # the framework on RQ3's dimension: the noise ratio is the one thing the
    # perturbation varies, so a reference-free metric that detects noise at all
    # should move with it. Context relevance in particular is a ratio of
    # selected to TOTAL sentences, and adding distractors inflates the
    # denominator, so it has a mechanical reason to respond -- which makes a
    # flat curve here the stronger negative result.
    by_ratio: dict[str, dict] = {}
    for r in ragas_rows:
        if r["dimension"] != "noise":
            continue
        base = scored_by_key.get(r["key"])
        if base is None or base.get("noise_ratio") is None:
            continue
        bucket = by_ratio.setdefault(f"{float(base['noise_ratio']):.2f}",
                                     {"n": 0, **{m: [] for m in metrics}})
        bucket["n"] += 1
        for m in metrics:
            if r.get(m) is not None:
                bucket[m].append(float(r[m]))
    report["noise_ratio_response"] = {
        "note": (
            "Mean RAGAS score by noise ratio on the noise testbed. The "
            "purpose-built NDC is built from exactly this x-axis, so a RAGAS "
            "metric that does not move across it is not measuring noise."
        ),
        "by_ratio": {
            k: {"n": v["n"], **{m: (round(sum(v[m]) / len(v[m]), 4) if v[m] else None)
                                for m in metrics}}
            for k, v in sorted(by_ratio.items())
        },
    }

    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\n-> {REPORT}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="choose the held-out subset")
    s.add_argument("--scored", default="runs/scored.jsonl")
    s.set_defaults(fn=do_select)

    c = sub.add_parser("score", help="run the RAGAS judge over the subset")
    c.add_argument("--concurrency", type=int, default=8)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--overwrite", action="store_true")
    c.set_defaults(fn=do_score)

    a = sub.add_parser("analyse", help="correlations and coverage gap")
    a.add_argument("--scored", default="runs/scored.jsonl")
    a.set_defaults(fn=do_analyse)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
