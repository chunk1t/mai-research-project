#!/usr/bin/env python3
"""Validate the CRS judge against a human rater (P1 Objective 2, Section 5.4.3).

    python scripts/crs_validation.py export     # build the blind rating sheet
    python scripts/crs_validation.py score      # kappa, once the sheet is filled

P1 requires inter-rater agreement between the judge and human raters "on a
sample of at least one hundred cases, in line with the LLM-as-judge protocol of
Zheng et al.", at a Cohen's kappa of 0.60 or above.

The export is BLIND by construction: the judge's score is written only to
`crs_judge_scores.csv`, which the rating sheet never references and the rater
never needs to open. A rater who has seen the judge's verdict is not an
independent rater, and the kappa would measure suggestibility rather than
agreement.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragrobust.judge import (  # noqa: E402
    RUBRIC_LEVELS,
    compute_judge_agreement,
    select_judge_validation_sample,
)
from ragrobust.schema import TestCase  # noqa: E402

OUT = ROOT / "runs" / "validation"
SHEET = OUT / "crs_review_sheet.html"
RATINGS = OUT / "crs_human_ratings.csv"
JUDGE = OUT / "crs_judge_scores.csv"


def load_cases(path: Path) -> dict[str, TestCase]:
    with open(path, encoding="utf-8") as fh:
        return {c.case_id: c for c in
                (TestCase.model_validate_json(line) for line in fh if line.strip())}


def do_export(args) -> int:
    judged = []
    with open(args.scored, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dimension") == "conflict" and row.get("crs_score") is not None:
                judged.append(row)
    print(f"{len(judged):,} judged conflict responses available")

    sample = select_judge_validation_sample(judged, n=args.n, rng_seed=args.seed)
    cases = load_cases(Path(args.benchmark))
    OUT.mkdir(parents=True, exist_ok=True)

    # The rater's sheet: blank score column, no judge verdict anywhere.
    with open(RATINGS, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "case_id", "config_id", "your_score_0_to_4", "notes"])
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["case_id"], r["config_id"], "", ""])

    # The judge's scores, kept apart so the sheet cannot leak them.
    with open(JUDGE, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item", "case_id", "config_id", "judge_score"])
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["case_id"], r["config_id"], r["crs_score"]])

    rubric = "".join(
        f"<li><b>Score {k}</b> &mdash; {html.escape(v)}</li>"
        for k, v in sorted(RUBRIC_LEVELS.items())
    )
    blocks = []
    for i, r in enumerate(sample, 1):
        case = cases.get(r["case_id"])
        if case is None:
            continue
        passages = "".join(
            f'<div class=p><b>Passage {j}</b><br>{html.escape(p.text[:2600])}</div>'
            for j, p in enumerate(case.retrieved_passages, 1)
        )
        answer = html.escape(r.get("final_answer") or "(the system produced no answer)")
        blocks.append(
            f"<section><h2>Item {i} of {len(sample)}</h2>"
            f"<p><b>Question:</b> {html.escape(case.query)}</p>"
            f"<h3>Evidence the system was given</h3>{passages}"
            f"<h3>The system's answer</h3><div class=a>{answer}</div>"
            f"<p class=q><b>Your score (0&ndash;4) for item {i}:</b> "
            f"record it in <code>crs_human_ratings.csv</code></p></section>"
        )

    SHEET.write_text(
        "<!doctype html><meta charset=utf-8><title>CRS judge validation</title>"
        "<style>body{font:15px/1.6 system-ui;max-width:62rem;margin:2rem auto;padding:0 1rem}"
        "section{border-top:2px solid #ccc;padding-top:1rem;margin-top:2.5rem}"
        ".p{background:#f6f6f6;padding:.75rem;margin:.5rem 0;white-space:pre-wrap;font-size:.92em}"
        ".a{background:#eef6ff;padding:.75rem;margin:.5rem 0;white-space:pre-wrap;font-weight:600}"
        ".q{background:#fffbe6;padding:.5rem}"
        "#rubric{position:sticky;top:0;background:#fff;border-bottom:2px solid #333;padding:.5rem 0}"
        "</style>"
        f"<h1>Conflict Resolution Score &mdash; judge validation</h1>"
        f"<p>Score each of the {len(sample)} responses below on the 0&ndash;4 rubric, judging "
        f"<b>how the response handles the disagreement between the passages</b>, not whether the "
        f"answer matches a fact you believe to be true. You are rating blind: the judge's own "
        f"scores are not shown here.</p>"
        f"<div id=rubric><h3>Rubric</h3><ol start=0 style='margin:0'>{rubric}</ol></div>"
        + "".join(blocks),
        encoding="utf-8",
    )

    print(f"\n  {SHEET}      <- read and score these")
    print(f"  {RATINGS}   <- fill in your_score_0_to_4")
    print(f"  {JUDGE}     <- the judge's scores, DO NOT OPEN before rating")
    print(f"\n{len(sample)} items across "
          f"{len({r['config_id'].split('|')[0] for r in sample})} pipeline classes")
    return 0


def do_score(args) -> int:
    human = {}
    for row in csv.DictReader(open(RATINGS, newline="", encoding="utf-8")):
        raw = (row.get("your_score_0_to_4") or "").strip()
        if raw:
            human[row["item"]] = int(raw)
    # Judge scores are read LIVE from scored.jsonl, keyed by the item's
    # (config_id, case_id) -- not from crs_judge_scores.csv, which is a snapshot
    # written when the sheet was exported. Re-judging updates scored.jsonl and
    # leaves that snapshot untouched, so comparing against it silently reports
    # agreement with a judge that no longer exists. That happened once already.
    live: dict[tuple[str, str], int] = {}
    for line in open(args.scored, encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("crs_score") is not None:
            live[(row["config_id"], row["case_id"])] = row["crs_score"]

    judge, stale = {}, 0
    for row in csv.DictReader(open(JUDGE, newline="", encoding="utf-8")):
        key = (row["config_id"], row["case_id"])
        if key in live:
            judge[row["item"]] = live[key]
            if live[key] != int(row["judge_score"]):
                stale += 1
    if stale:
        print(f"note: {stale} of {len(judge)} judge scores differ from the exported "
              f"snapshot; using the current scores from {args.scored}")

    shared = sorted(set(human) & set(judge), key=int)
    unrated = len(judge) - len(shared)
    if not shared:
        print("no rated items yet; fill in crs_human_ratings.csv first", file=sys.stderr)
        return 2

    # Blank rows are dropped, never defaulted: an unrated row is absent data,
    # and scoring it as agreement would inflate the number the gate rests on.
    result = compute_judge_agreement([(judge[i], human[i]) for i in shared])
    payload = {**result.as_dict(), "unrated_items_excluded": unrated}
    (OUT / "crs_judge_agreement.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    if unrated:
        print(f"\nNOTE: {unrated} item(s) unrated and excluded. P1 asks for at least "
              f"one hundred; {len(shared)} are rated.")
    print(f"\n-> {OUT / 'crs_judge_agreement.json'}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="build the blind rating sheet")
    e.add_argument("--scored", default="runs/scored.jsonl")
    e.add_argument("--benchmark", default="data/cases/benchmark.jsonl")
    e.add_argument("-n", type=int, default=100)
    e.add_argument("--seed", type=int, default=20260721)
    e.set_defaults(fn=do_export)
    s = sub.add_parser("score", help="compute kappa from the filled sheet")
    s.add_argument("--scored", default="runs/scored.jsonl")
    s.set_defaults(fn=do_score)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
