#!/usr/bin/env python3
"""Bake the curated cases into a self-contained HTML explorer.

    python scripts/build_demo_data.py            # writes demo/index.html
    python scripts/build_demo_data.py --json     # also writes demo/demo_data.json

The terminal demonstration needs the repository, the 36 MB benchmark and the
72 MB response cache. This does not: it is one file with the data inside it,
for the case where the room has no network, the laptop is not the presenter's,
or a reviewer wants to look at the evidence a week later.

Everything here comes from the stored records in `runs/results.jsonl` and
`runs/scored.jsonl` -- what the run produced and what the scorer made of it --
rather than from a replay. The two agree instance for instance (that is what
`demo.py verify` establishes), and taking the stored record means the page
shows the response the reported numbers were actually computed from, including
on the instances where a shared cache entry makes replay return the other
retriever arm's answer.

The curated set is `demo.SHOWCASE` plus `EXTRA` below, and it is deliberately
not a flattering selection: two entries exist to show the scorer understating
an arm, and one shows a pipeline returning the right answer and scoring zero.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import demo  # noqa: E402

from ragrobust.schema import NO_ANSWER  # noqa: E402

OUT_DIR = ROOT / "demo"
OUT_HTML = OUT_DIR / "index.html"
OUT_JSON = OUT_DIR / "demo_data.json"

# Beyond the eight showcase entries, chosen to widen the sample rather than to
# strengthen it: two more conflict cases where the class ordering flips again,
# two refusal cases on the second generator, and two noise cases at r=0.90.
EXTRA: tuple[tuple[str, str], ...] = (
    ("conflict-nq-1285",
     "The agentic arm reaches level 3 while reasoning abstains and naive answers blind."),
    ("conflict-nq-2375",
     "Reasoning names the disagreement; naive answers from the wrong passage."),
    ("conflict-tqa-2139",
     "A level-4 resolution on the second retriever arm."),
    ("refusal-unans-tqa-3388",
     "An unanswerable case the agentic arm answers anyway."),
    ("refusal-ctrl-tqa-700",
     "An answerable control that the reasoning arm refuses."),
    ("refusal-unans-nq-4910",
     "An unparseable response: neither a refusal nor an answer (Section 6.5.4)."),
    ("noise-tqa-4778-r90",
     "At r=0.90 the naive arm answers from a distractor."),
    ("noise-tqa-4013-r90",
     "Two defensible answers, one of which the string matcher rejects."),
)

CONFIGS = [
    f"{pclass}|bm25|{template.format(n=arm)}"
    for arm in (1, 2)
    for pclass, template in demo.ARM_CLASSES.items()
]


def curated_ids() -> list[tuple[str, str]]:
    """(case_id, why) for every case the page carries, in narrative order."""
    out: list[tuple[str, str]] = []
    for show in demo.SHOWCASE:
        parts = show.command.split()
        if parts[0] == "case":
            out.append((parts[1], show.headline))
        else:
            for r in demo.NOISE_RATIOS:
                out.append((f"noise-{parts[1]}-r{r}", show.headline))
    out.extend(EXTRA)
    seen: set[str] = set()
    return [(cid, why) for cid, why in out if not (cid in seen or seen.add(cid))]


def conflict_diff(case: Any) -> dict[str, str] | None:
    """The single claim that differs between the original and the contradiction."""
    import difflib

    original = next((p for p in case.retrieved_passages if p.is_answer_bearing), None)
    injected = next((p for p in case.retrieved_passages if p.is_injected), None)
    if original is None or injected is None:
        return None
    a, b = demo._sentences(original.text), demo._sentences(injected.text)
    diff = [x for x in difflib.unified_diff(a, b, n=0, lineterm="")
            if not x.startswith(("---", "+++", "@@"))]
    removed = [x[1:].strip() for x in diff if x.startswith("-")]
    added = [x[1:].strip() for x in diff if x.startswith("+")]
    if not removed or not added:
        return None
    return {"original": removed[0], "injected": added[0]}


def build_payload() -> dict[str, Any]:
    corpus = demo.load_corpus()
    cases: list[dict[str, Any]] = []
    for case_id, why in curated_ids():
        case = corpus.by_id.get(case_id)
        if case is None:
            raise SystemExit(f"curated case '{case_id}' is not in the benchmark")
        arms = []
        for config_id in CONFIGS:
            key = f"{config_id}::{case_id}"
            stored = corpus.stored.get(key)
            scored = corpus.scored.get(key)
            if stored is None or scored is None:
                continue
            steps = [
                {"node": s.get("node"), "query": s.get("query"),
                 "subqueries": s.get("subqueries"),
                 "n_passages": len(s.get("passage_ids") or [])}
                for s in (stored.get("steps") or [])
            ]
            arms.append({
                "config_id": config_id,
                "pipeline_class": config_id.split("|")[0],
                "generator": config_id.split("|")[2],
                "final_answer": scored.get("final_answer") or stored.get("final_answer"),
                "raw_text": stored.get("raw_text") or "",
                "calls": stored.get("generator_calls"),
                "steps": steps,
                "answer_correct": scored.get("answer_correct"),
                "category": scored.get("category"),
                "crs": scored.get("crs_score"),
                "recovery": scored.get("recovery_method"),
                "truncated": bool(stored.get("truncated")),
            })
        cases.append({
            "case_id": case_id,
            "why": why,
            "dimension": case.dimension.value,
            "perturbation": case.perturbation_type.value,
            "noise_ratio": case.noise_ratio,
            "query": case.query,
            "answer": None if case.answer == NO_ANSWER else case.answer,
            "seed_source": case.seed_source.value,
            "passages": [
                {"passage_id": p.passage_id, "text": p.text,
                 "answer_bearing": p.is_answer_bearing, "injected": p.is_injected,
                 "tokens": p.token_estimate()}
                for p in case.retrieved_passages
            ],
            "diff": conflict_diff(case),
            "arms": arms,
        })
    # Model identity travels from the config, never from the template: the
    # project convention is that swapping a generator is a configuration edit,
    # and a page that named the models itself would quietly go stale.
    generators = {
        key: {"model": spec["model"], "family": spec["family"],
              "thinking": bool(spec.get("thinking")), "cot": bool(spec.get("cot_prompt"))}
        for key, spec in (corpus.models_cfg.get("generators") or {}).items()
    }
    corpus.cache.close()
    return {
        "cases": cases,
        "rubric": demo.CRS_RUBRIC,
        "configs": CONFIGS,
        "generators": generators,
        "generated_from": {
            "results": "runs/results.jsonl",
            "scored": "runs/scored.jsonl",
            "benchmark": "data/cases/benchmark.jsonl",
        },
    }


def render(payload: dict[str, Any], template: Path) -> str:
    """Substitute the payload into the page template.

    A placeholder rather than string formatting, because the template is CSS and
    JavaScript full of braces. `</script>` inside the data would end the block
    early, so the closing angle bracket is escaped -- the only transformation
    applied to the JSON.
    """
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")
    text = template.read_text(encoding="utf-8")
    if "__DEMO_DATA__" not in text:
        raise SystemExit(f"{template}: no __DEMO_DATA__ placeholder")
    return text.replace("__DEMO_DATA__", blob)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", action="store_true", help="also write demo/demo_data.json")
    p.add_argument("--template", default=str(OUT_DIR / "template.html"))
    args = p.parse_args()

    payload = build_payload()
    OUT_DIR.mkdir(exist_ok=True)
    if args.json:
        OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {OUT_JSON} ({OUT_JSON.stat().st_size / 1024:.0f} KB)")

    OUT_HTML.write_text(render(payload, Path(args.template)), encoding="utf-8")
    n_arms = sum(len(c["arms"]) for c in payload["cases"])
    print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB): "
          f"{len(payload['cases'])} cases, {n_arms} stored records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
