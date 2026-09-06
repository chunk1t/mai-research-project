#!/usr/bin/env python3
"""Score a run into per-configuration metrics (P1 Sections 5.4.3, 5.6.4).

    python scripts/score_results.py --results runs/results.jsonl
    python scripts/score_results.py --results runs/validation_run.jsonl --no-judge

Three stages, kept separate because they have different costs and different
failure modes. Classification is local and free. CRS judging costs Anthropic
calls and is cached. Metric computation is pure arithmetic over the first two.

Only the delimited final answer is classified and judged; the reasoning trace is
never passed to either (P1 5.6.4).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragrobust.cache import ResponseCache  # noqa: E402
from ragrobust.judge import build_crs_prompt, parse_crs_score  # noqa: E402
from ragrobust.metrics.conflict import compute_crs  # noqa: E402
from ragrobust.metrics.noise import (  # noqa: E402
    accuracy_by_ratio_from_responses,
    compute_ndc,
)
from ragrobust.metrics.refusal import compute_refusal_f1  # noqa: E402
from ragrobust.metrics.scored import ResponseCategory, ScoredResponse  # noqa: E402
from ragrobust.parsing.classifier import ResponseParser  # noqa: E402
from ragrobust.parsing.rules import recover_unparsed_answer, score_answer  # noqa: E402
from ragrobust.providers.base import GenerationRequest  # noqa: E402
from ragrobust.providers.cached import CachedProvider  # noqa: E402
from ragrobust.providers.factory import build_provider, load_config  # noqa: E402
from ragrobust.schema import NO_ANSWER, Dimension, TestCase  # noqa: E402

DATA = ROOT / "data"
JUDGE_CACHE = DATA / "cache" / "judge.sqlite"


def load_benchmark(path: Path) -> dict[str, TestCase]:
    with open(path, encoding="utf-8") as fh:
        return {c.case_id: c for c in
                (TestCase.model_validate_json(line) for line in fh if line.strip())}


async def classify_all(rows, cases, parser) -> tuple[dict[str, list[ScoredResponse]], dict]:
    """Reduce raw pipeline rows to ScoredResponse, grouped by configuration.

    Also recovers responses that committed an answer without the delimiter (see
    `recover_unparsed_answer`), counting every recovery by method and by
    generator so Chapter 6 can report exactly how many were affected and why.
    """
    by_config: dict[str, list[ScoredResponse]] = defaultdict(list)
    recovered: dict[str, int] = defaultdict(int)
    recovered_by_gen: dict[str, int] = defaultdict(int)
    unrecoverable = 0
    for row in rows:
        case = cases.get(row["case_id"])
        if case is None:
            continue
        final = row.get("final_answer")
        method_used = None
        if final is None:
            final, method = recover_unparsed_answer(row.get("raw_text") or "")
            method_used = method
            if method:
                recovered[method] += 1
                recovered_by_gen[row.get("generator_key", "?")] += 1
            elif (row.get("raw_text") or "").strip():
                unrecoverable += 1
        cls = await parser.classify(case.query, final)

        correct = em = f1 = None
        if cls.category is ResponseCategory.ANSWER and case.answer != NO_ANSWER:
            s = score_answer(final or "", case.answer)
            correct, em, f1 = s.correct, s.exact_match, s.token_f1

        by_config[row["config_id"]].append(ScoredResponse(
            case_id=case.case_id,
            dimension=case.dimension,
            category=cls.category,
            answer_correct=correct,
            is_answerable=case.is_answerable,
            noise_ratio=case.noise_ratio,
            final_answer=final,
            classification_method=cls.method,
            exact_match=em,
            token_f1=f1,
            # Carried for the analysis layer. seed_id is what the noise
            # bootstrap resamples on -- an NDC is fitted across the five ratios
            # of ONE seed, so resampling instances would tear curves apart.
            # `truncated` separates "could not terminate" from "resolved the
            # conflict badly", which CRS otherwise conflates.
            seed_id=case.seed_id,
            truncated=bool(row.get("truncated")),
            recovery_method=method_used,
        ))
    stats = {
        "recovered_by_method": dict(recovered),
        "recovered_by_generator": dict(recovered_by_gen),
        "recovered_total": sum(recovered.values()),
        "unrecoverable_nonempty": unrecoverable,
    }
    return by_config, stats


async def judge_conflicts(by_config, cases, judge, limit: int | None):
    """Attach CRS rubric scores to conflict responses.

    Judged per (configuration, case): the same case answered differently by two
    configurations is two different things to score, so the judge cannot be
    memoised on the case alone. The response cache handles the identity, since
    the prompt embeds the answer.
    """
    n_judged = n_unparseable = 0
    for config_id, responses in by_config.items():
        conflicts = [r for r in responses if r.dimension is Dimension.CONFLICT]
        if limit:
            conflicts = conflicts[:limit]
        for r in conflicts:
            case = cases[r.case_id]
            prompt = build_crs_prompt(case, r.final_answer or "")
            resp = await judge.generate(
                # 200, not 16. The judge is asked to QUOTE the words in the
                # response that mention the disagreement before emitting its
                # score -- forcing evidence ahead of the verdict is what stops it
                # asserting a property the response does not have. At 16 tokens
                # the quote line alone would exhaust the budget and every
                # judgement would truncate before <score>, parsing as None.
                GenerationRequest(prompt=prompt, max_tokens=200, temperature=0.0)
            )
            score = parse_crs_score(resp.text) if resp.ok else None
            if score is None:
                n_unparseable += 1
            else:
                n_judged += 1
            r.crs_score = score
    return {"judged": n_judged, "unparseable": n_unparseable}


def non_termination(responses: list[ScoredResponse]) -> dict:
    """How often a configuration ran out of output budget without answering.

    Distinct from a wrong answer and from a refusal. Qwen3-8B in thinking mode
    hits the cap on a large share of conflict cases -- bimodally, answering in
    ~537 completion tokens or looping to the ceiling -- while the agentic arm
    running the same model in the same mode does not. Reported per dimension
    because the effect concentrates almost entirely in conflict.
    """
    out: dict[str, object] = {}
    for dim in sorted({r.dimension.value for r in responses}):
        sub = [r for r in responses if r.dimension.value == dim]
        stuck = [r for r in sub if r.truncated and r.final_answer is None]
        out[dim] = {
            "n": len(sub),
            "non_terminating": len(stuck),
            "rate": round(len(stuck) / len(sub), 4) if sub else 0.0,
        }
    return out


def metrics_for(responses: list[ScoredResponse]) -> dict:
    """Compute the three P1 metrics over one configuration's responses."""
    out: dict[str, object] = {"n": len(responses),
                              "non_termination": non_termination(responses)}

    refusal = [r for r in responses if r.dimension is Dimension.REFUSAL]
    if refusal:
        out["refusal"] = compute_refusal_f1(refusal).as_dict()

    conflict = [r for r in responses if r.dimension is Dimension.CONFLICT]
    scored = [r.crs_score for r in conflict if r.crs_score is not None]
    if scored:
        # A case the judge could not score is excluded rather than counted zero:
        # zero is a real rubric level meaning "ignored the conflict", and
        # conflating a parse failure with it would drag the mean down and
        # flatter the binary baseline CRS is compared against (P1 5.4.3).
        judged = [r for r in conflict if r.crs_score is not None]
        out["conflict"] = compute_crs(
            scored, correct_side_flags=[r.answer_correct for r in judged]
        ).as_dict()
        out["conflict"]["n_unscored"] = len(conflict) - len(scored)

    noise = [r for r in responses if r.dimension is Dimension.NOISE]
    if noise:
        acc = accuracy_by_ratio_from_responses(noise)
        out["noise_accuracy_by_ratio"] = {str(k): round(v, 4) for k, v in sorted(acc.items())}
        try:
            out["noise"] = compute_ndc(acc).as_dict()
        except ValueError as exc:
            # A missing ratio cannot be interpolated across honestly, so the
            # curve is withheld and the reason recorded (P1 5.4.3).
            out["noise"] = {"error": str(exc)}
    return out


async def main_async(args) -> int:
    # Last row wins per key: a retried instance is appended again rather than
    # rewritten, so an earlier failed attempt must not shadow the later success.
    _seen: dict[str, dict] = {}
    for line in open(args.results, encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            _seen[row.get("key") or f"{row['config_id']}::{row['case_id']}"] = row
    rows = list(_seen.values())
    cases = load_benchmark(Path(args.benchmark))
    models_cfg = load_config(args.models_config)
    print(f"{len(rows)} result rows over {len({r['config_id'] for r in rows})} configurations")

    errored = [r for r in rows if r.get("error")]
    truncated = [r for r in rows if r.get("truncated")]
    rows = [r for r in rows if not r.get("error")]
    print(f"  {len(errored)} errored (excluded), {len(truncated)} truncated before an answer")

    parser = ResponseParser()
    by_config, recovery = await classify_all(rows, cases, parser)
    print(f"  classification methods: {parser.counts}")
    if recovery["recovered_total"]:
        print(f"  recovered {recovery['recovered_total']} undelimited answers "
              f"{recovery['recovered_by_method']} across {recovery['recovered_by_generator']}")
    print(f"  unrecoverable non-empty responses: {recovery['unrecoverable_nonempty']}")

    judge_stats = None
    if not args.no_judge:
        spec = models_cfg["crs_judge"]
        judge = CachedProvider(
            build_provider("crs_judge", spec, models_cfg), ResponseCache(JUDGE_CACHE)
        )
        await judge.open()
        print(f"  CRS judge: {spec['model']} (family {spec['family']})")
        judge_stats = await judge_conflicts(by_config, cases, judge, args.judge_limit)
        print(f"  judged: {judge_stats}")
        await judge.close()

    report = {
        "n_rows": len(rows),
        "n_errored": len(errored),
        "n_truncated": len(truncated),
        "classification_methods": parser.counts,
        "undelimited_answer_recovery": recovery,
        "crs_judge": judge_stats,
        "configurations": {cid: metrics_for(rs) for cid, rs in sorted(by_config.items())},
    }
    # Per-instance records, which the analysis layer resamples. Written
    # separately from the aggregate because bootstrapping needs the individual
    # observations, and re-deriving them would mean re-running the CRS judge.
    scored_path = Path(args.scored_out)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scored_path, "w", encoding="utf-8") as fh:
        for config_id, responses in sorted(by_config.items()):
            for r in responses:
                fh.write(json.dumps({"config_id": config_id, **r.model_dump(mode="json")}) + "\n")
    n_scored = sum(len(v) for v in by_config.values())
    print(f"per-instance scores -> {scored_path} ({n_scored:,} rows)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nscores -> {out}")
    for cid, m in sorted(report["configurations"].items()):
        bits = [f"n={m['n']}"]
        if "refusal" in m:
            bits.append(f"RefusalF1={m['refusal'].get('refusal_f1')}")
        if "conflict" in m:
            bits.append(f"CRS={m['conflict'].get('crs_mean')}")
        if isinstance(m.get("noise"), dict) and "ndc_auc" in m["noise"]:
            bits.append(f"NDC-AUC={m['noise']['ndc_auc']}")
        print(f"  {cid:34s} {'  '.join(bits)}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="runs/results.jsonl")
    p.add_argument("--benchmark", default=str(DATA / "cases" / "benchmark.jsonl"))
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--out", default="runs/scores.json")
    p.add_argument("--scored-out", default="runs/scored.jsonl",
                   help="per-instance scored records, consumed by the analysis layer")
    p.add_argument("--no-judge", action="store_true",
                   help="skip CRS judging (no Anthropic calls); refusal and noise still scored")
    p.add_argument("--judge-limit", type=int, default=None,
                   help="judge at most N conflict cases per configuration")
    args = p.parse_args()
    if not args.no_judge and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY unset; use --no-judge to score refusal and noise only",
              file=sys.stderr)
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
