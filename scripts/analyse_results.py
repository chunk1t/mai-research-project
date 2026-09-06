#!/usr/bin/env python3
"""Descriptive, inferential, and qualitative analysis of a scored run (P1 5.7).

    python scripts/score_results.py --results runs/results.jsonl   # writes scored.jsonl
    python scripts/analyse_results.py                              # this script

P1 specifies three layers and this produces all three:

  descriptive  mean and confidence interval per pipeline class per metric, with
               Refusal F1 decomposed into precision and recall and CRS reported
               alongside its distribution over the zero-to-four rubric;
  inferential  bootstrap confidence intervals on the DIFFERENCES between
               classes, tested pairwise at the case level;
  qualitative  a stratified export of cases where the classes diverge, for the
               failure-pattern discussion.

Reads `runs/scored.jsonl` rather than re-deriving scores, because re-deriving
them would mean re-running the CRS judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragrobust.analysis import (  # noqa: E402
    PIPELINE_CLASSES,
    InsufficientData,
    bootstrap_ci,
    crs_statistic,
    ndc_auc_statistic,
    paired_class_difference,
    refusal_statistic,
)
from ragrobust.analysis import (  # noqa: E402
    seed_source_of,
    unpaired_group_difference,
    within_class_variation,
)
from ragrobust.metrics.conflict import compute_crs  # noqa: E402
from ragrobust.metrics.noise import (  # noqa: E402
    accuracy_by_ratio_from_responses,
    compute_ndc,
)
from ragrobust.metrics.refusal import compute_refusal_f1  # noqa: E402
from ragrobust.metrics.scored import ScoredResponse  # noqa: E402
from ragrobust.schema import Dimension  # noqa: E402


def load_scored(path: Path) -> dict[str, list[ScoredResponse]]:
    """Group per-instance scored records by pipeline class."""
    by_class: dict[str, list[ScoredResponse]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            pclass = row["config_id"].split("|")[0]
            row.pop("config_id", None)
            by_class[pclass].append(ScoredResponse.model_validate(row))
    return by_class


def load_scored_by_configuration(path: Path) -> dict[str, list[ScoredResponse]]:
    """Group per-instance scored records by the full configuration id.

    Separate from `load_scored`, which groups by pipeline class and discards the
    configuration. P1 5.9's third threat is mitigated by "including multiple
    retrievers and multiple generators within each pipeline class, so that
    within-class variation is observable", and that variation is invisible once
    the four configurations of a class have been pooled into one number.
    """
    by_config: dict[str, list[ScoredResponse]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            config_id = row.pop("config_id")
            by_config[config_id].append(ScoredResponse.model_validate(row))
    return by_config


def of_dim(responses, dim: Dimension) -> list[ScoredResponse]:
    return [r for r in responses if r.dimension is dim]


def load_ragas_faithfulness(path: Path) -> dict[str, float]:
    """Mean RAGAS faithfulness on conflict instances, per pipeline class.

    P1 5.4: "The Conflict Resolution Score is compared against a binary
    correct-side accuracy and against RAGAS faithfulness." This supplies the
    second comparator from the secondary analysis of 5.7, so the CRS row
    carries both baselines P1 asks for rather than one.

    Computed over the held-out RAGAS subset, not the full run, and over the
    instances where faithfulness is DEFINED -- it is 0/0 on an abstention, and
    roughly half of all conflict responses abstain. The full picture of that is
    in runs/ragas_analysis.json; this is the single comparator number.
    """
    if not path.exists():
        return {}
    sums: dict[str, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dimension") != "conflict":
                continue
            if row.get("faithfulness") is not None:
                sums[row["pipeline_class"]].append(float(row["faithfulness"]))
    return {k: sum(v) / len(v) for k, v in sums.items() if v}


def describe(by_class, cfg, ragas_faithfulness: dict[str, float] | None = None) -> dict:
    """Layer one: point estimates with intervals, per class."""
    n, conf, seed = cfg["n_bootstrap"], cfg["confidence"], cfg["rng_seed"]
    ragas_faithfulness = ragas_faithfulness or {}
    out: dict[str, dict] = {}
    for pclass in PIPELINE_CLASSES:
        responses = by_class.get(pclass, [])
        if not responses:
            continue
        entry: dict[str, object] = {"n_instances": len(responses)}

        refusal = of_dim(responses, Dimension.REFUSAL)
        if refusal:
            res = compute_refusal_f1(refusal)
            entry["refusal"] = res.as_dict()
            try:
                ci = bootstrap_ci(sorted({r.case_id for r in refusal}),
                                  refusal_statistic(refusal),
                                  n_resamples=n, confidence=conf, seed=seed)
                entry["refusal"]["f1_ci"] = ci.as_dict()
            except InsufficientData as exc:
                entry["refusal"]["f1_ci"] = {"error": str(exc)}

        conflict = of_dim(responses, Dimension.CONFLICT)
        judged = [r for r in conflict if r.crs_score is not None]
        scores = [r.crs_score for r in judged]
        if scores:
            # P1 5.4 requires CRS to be reported against a binary correct-side
            # accuracy. `answer_correct` already carries that judgement per
            # instance -- True on the true side, False on the contradicted one,
            # None where the pipeline committed to no answer -- so the baseline
            # is aligned to the same cases the rubric was scored on.
            entry["conflict"] = compute_crs(
                scores,
                correct_side_flags=[r.answer_correct for r in judged],
                ragas_faithfulness=ragas_faithfulness.get(pclass),
            ).as_dict()
            # Non-terminating responses are reported separately so a reader can
            # tell "resolved the conflict badly" from "never finished".
            stuck = [r for r in conflict if r.truncated and r.final_answer is None]
            entry["conflict"]["non_terminating"] = len(stuck)
            entry["conflict"]["non_terminating_rate"] = round(len(stuck) / len(conflict), 4)
            for label, exclude in (("crs_ci", False), ("crs_ci_excluding_non_terminating", True)):
                try:
                    ci = bootstrap_ci(sorted({r.case_id for r in conflict}),
                                      crs_statistic(conflict, exclude_non_terminating=exclude),
                                      n_resamples=n, confidence=conf, seed=seed)
                    entry["conflict"][label] = ci.as_dict()
                except InsufficientData as exc:
                    entry["conflict"][label] = {"error": str(exc)}

        noise = of_dim(responses, Dimension.NOISE)
        if noise:
            acc = accuracy_by_ratio_from_responses(noise)
            entry["noise_accuracy_by_ratio"] = {str(k): round(v, 4) for k, v in sorted(acc.items())}
            try:
                ndc = compute_ndc(acc)
                entry["noise"] = ndc.as_dict()
            except ValueError as exc:
                entry["noise"] = {"error": str(exc)}
            try:
                ci = bootstrap_ci(sorted({r.seed_id or r.case_id for r in noise}),
                                  ndc_auc_statistic(noise),
                                  n_resamples=n, confidence=conf, seed=seed)
                entry["noise"]["auc_ci"] = ci.as_dict()
            except InsufficientData as exc:
                entry["noise"]["auc_ci"] = {"error": str(exc)}

        out[pclass] = entry
    return out


def compare(by_class, cfg) -> list[dict]:
    """Layer two: paired differences between classes, on the same cases."""
    n, conf, seed = cfg["n_bootstrap"], cfg["confidence"], cfg["rng_seed"]
    thresholds = cfg["practical_significance"]
    pairs = [("reasoning", "naive"), ("agentic", "reasoning"), ("agentic", "naive")]
    specs = [
        ("refusal_f1", Dimension.REFUSAL, refusal_statistic,
         lambda r: r.case_id, thresholds["refusal_f1"]),
        ("crs", Dimension.CONFLICT, crs_statistic,
         lambda r: r.case_id, thresholds["crs"]),
        ("ndc_auc", Dimension.NOISE, ndc_auc_statistic,
         lambda r: r.seed_id or r.case_id, thresholds["ndc_auc"]),
    ]
    out: list[dict] = []
    for metric, dim, factory, unit_of, threshold in specs:
        for a, b in pairs:
            ra, rb = of_dim(by_class.get(a, []), dim), of_dim(by_class.get(b, []), dim)
            if not ra or not rb:
                continue
            try:
                cmp = paired_class_difference(
                    ra, rb, metric=metric, class_a=a, class_b=b, threshold=threshold,
                    statistic_factory=factory, unit_of=unit_of,
                    n_resamples=n, confidence=conf, seed=seed,
                )
                out.append(cmp.as_dict())
            except InsufficientData as exc:
                out.append({"metric": metric, "comparison": f"{a} vs {b}", "error": str(exc)})
    return out


def divergent_cases(by_class, limit_per_dim: int = 20) -> list[dict]:
    """Layer three: cases where the classes disagree, stratified by dimension.

    P1 5.7's qualitative layer examines "a stratified sample of cases on which
    the pipeline classes diverge" to explain the quantitative result.
    """
    by_case: dict[tuple, dict[str, ScoredResponse]] = defaultdict(dict)
    for pclass, responses in by_class.items():
        for r in responses:
            by_case[(r.dimension.value, r.case_id)][pclass] = r

    picked: dict[str, list[dict]] = defaultdict(list)
    for (dim, case_id), arms in sorted(by_case.items()):
        if len(arms) < 2:
            continue
        if dim == "conflict":
            vals = {c: r.crs_score for c, r in arms.items() if r.crs_score is not None}
            if len(set(vals.values())) < 2 or max(vals.values()) - min(vals.values()) < 2:
                continue
        else:
            vals = {c: (r.answer_correct if dim == "noise" else r.category.value)
                    for c, r in arms.items()}
            if len(set(map(str, vals.values()))) < 2:
                continue
        if len(picked[dim]) >= limit_per_dim:
            continue
        picked[dim].append({
            "case_id": case_id, "dimension": dim,
            "by_class": {c: str(v) for c, v in vals.items()},
        })
    return [row for rows in picked.values() for row in rows]


# --------------------------------------------------------------------------
# P1 Section 5.9, threat 2: does the benchmark over-fit its seed datasets?
# --------------------------------------------------------------------------

SEED_SOURCES = ("natural_questions", "trivia_qa")


def _source_components(groups: dict, dim: Dimension) -> dict:
    """The per-source detail behind a seed-source difference."""
    out: dict[str, dict] = {}
    for src, rows in groups.items():
        if not rows:
            continue
        if dim is Dimension.REFUSAL:
            res = compute_refusal_f1(rows)
            out[src] = {
                "precision": round(res.precision, 4),
                "recall": round(res.recall, 4),
                "f1": round(res.f1, 4),
                "tp": res.tp, "fp": res.fp, "fn": res.fn, "tn": res.tn,
                # Derived from the confusion matrix rather than stored: an
                # unanswerable case is either correctly refused (tp) or missed
                # (fn), and an answerable control is either wrongly refused (fp)
                # or correctly answered (tn).
                "n_unanswerable": res.tp + res.fn,
                "n_answerable": res.fp + res.tn,
            }
        elif dim is Dimension.CONFLICT:
            judged = [r for r in rows if r.crs_score is not None]
            if judged:
                res = compute_crs([r.crs_score for r in judged],
                                  correct_side_flags=[r.answer_correct for r in judged])
                out[src] = {k: v for k, v in res.as_dict().items()
                            if k != "ragas_faithfulness"}
        else:
            acc = accuracy_by_ratio_from_responses(rows)
            out[src] = {"accuracy_by_ratio": {str(k): round(v, 4)
                                              for k, v in sorted(acc.items())}}
    return out


def by_seed_source(by_class, cfg) -> dict:
    """Every metric split by seed dataset, with an interval on the difference.

    P1 5.9 mitigates the over-fitting threat "by using two seed datasets with
    different question styles and by reporting metric scores separately for each
    seed source so that any systematic difference can be detected". Two point
    estimates side by side cannot detect a difference; an interval can, so each
    split carries a two-sample bootstrap on the gap.

    Unpaired, not paired. A Natural Questions case and a TriviaQA case are
    different cases and no case appears in both groups, so there is nothing to
    pair -- see `analysis.unpaired_group_difference`.

    A difference found here is a Chapter 8 finding, not a defect to suppress:
    P1 asks for the check precisely so that a systematic difference becomes
    visible rather than averaging away inside a headline number.
    """
    n, conf, seed = cfg["n_bootstrap"], cfg["confidence"], cfg["rng_seed"]
    thresholds = cfg["practical_significance"]
    out: dict[str, dict] = {}

    specs = (
        ("refusal_f1", Dimension.REFUSAL, refusal_statistic,
         lambda r: r.case_id, thresholds["refusal_f1"]),
        ("crs", Dimension.CONFLICT, crs_statistic,
         lambda r: r.case_id, thresholds["crs"]),
        ("ndc_auc", Dimension.NOISE, ndc_auc_statistic,
         lambda r: r.seed_id or r.case_id, thresholds["ndc_auc"]),
    )

    for pclass in PIPELINE_CLASSES:
        responses = by_class.get(pclass, [])
        if not responses:
            continue
        entry: dict[str, object] = {}
        for metric, dim, factory, unit_of, threshold in specs:
            pool = of_dim(responses, dim)
            groups = {
                src: [r for r in pool if seed_source_of(r) == src]
                for src in SEED_SOURCES
            }
            unknown = [r for r in pool if seed_source_of(r) == "unknown"]
            if not all(groups.values()):
                entry[metric] = {"error": "one seed source has no cases in this dimension"}
                continue
            try:
                cmp_ = unpaired_group_difference(
                    groups["natural_questions"], groups["trivia_qa"],
                    metric=metric, group_a="natural_questions", group_b="trivia_qa",
                    statistic_factory=factory, unit_of=unit_of,
                    n_resamples=n, confidence=conf, seed=seed,
                )
                row = cmp_.as_dict()
                # The practical threshold is the same one declared in advance for
                # the between-class comparisons. Reused deliberately: a seed-source
                # gap that would count as a real effect between architectures
                # should count as one here too.
                row["practical_threshold"] = threshold
                row["exceeds_practical_threshold"] = (
                    abs(cmp_.difference.observed) >= threshold
                )
                if unknown:
                    row["instances_with_unrecognised_seed_source"] = len(unknown)
                # Components per source. A scalar gap says THAT the sources
                # differ; the components say WHERE, and on this data the whole
                # refusal gap turns out to sit in precision -- pipelines refuse
                # answerable TriviaQA controls far more often than answerable NQ
                # ones. Without this breakdown the finding is unusable.
                row["components"] = _source_components(groups, dim)
                entry[metric] = row
            except InsufficientData as exc:
                entry[metric] = {"error": str(exc)}
        out[pclass] = entry
    return out


# --------------------------------------------------------------------------
# Figures (P1 Section 5.7)
#
# Every figure is drawn from an artefact on disk, never from a number typed in
# here, so a figure cannot disagree with the text that cites it. The builders
# below are PRESENTATIONAL: they read values and draw them, and any arithmetic
# they would need has already been computed and tested upstream.
#
# Each figure is emitted twice. The report variant is sized for a page; the
# slides variant enlarges type and thickens lines for a projector, where a
# 9-point tick label is unreadable from the back of a room.
# --------------------------------------------------------------------------

# Okabe-Ito, which stays distinguishable under the common forms of colour
# vision deficiency. The class -> colour mapping is fixed across every figure so
# a reader who learns it once can carry it through the chapter.
CLASS_COLOUR = {"naive": "#0072B2", "reasoning": "#D55E00", "agentic": "#009E73"}
NEUTRAL = "#555555"

REPORT_STYLE = {
    "figure.figsize": (6.5, 4.2), "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 8, "lines.linewidth": 1.6,
    "lines.markersize": 5, "savefig.dpi": 150,
}
SLIDE_STYLE = {
    "figure.figsize": (10.0, 6.0), "font.size": 15, "axes.titlesize": 17,
    "axes.labelsize": 15, "legend.fontsize": 13, "lines.linewidth": 2.8,
    "lines.markersize": 9, "savefig.dpi": 140,
}

# Figures that place a legend inside the plot area size it in points rather
# than with the "small" keyword, so it stays clear of the data at both variants.
# The slide multiplier keeps such a legend readable on a projector without
# letting it grow back over the lines it was moved off.
LEGEND_PT = {"report": 7, "slides": 11}


def _load_optional(path: Path) -> dict | None:
    """Read an artefact a figure needs, or None if that stage has not been run."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _footnote(fig, text: str) -> None:
    """Put a caption under the axes and reserve room for it.

    Captions drawn with `ax.text` inside the axes overlap the data whenever the
    plot fills its box, which both the forest plot and the seed-source chart
    did on the bottom row.
    """
    fig.subplots_adjust(bottom=0.16)
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize="small", color=NEUTRAL)


def _fig_ndc(plt, analysis, _extra):
    """The P1 Figure 5.3 shape, drawn from real data."""
    fig, ax = plt.subplots()
    for pclass in PIPELINE_CLASSES:
        acc = (analysis["descriptive"].get(pclass) or {}).get("noise_accuracy_by_ratio")
        if not acc:
            continue
        xs = sorted(float(k) for k in acc)
        ax.plot(xs, [acc[str(x)] for x in xs], marker="o", label=pclass,
                color=CLASS_COLOUR.get(pclass))
    ax.axhline(0.5, ls="--", lw=1.0, color=NEUTRAL)
    ax.text(0.01, 0.51, "NDC-50 threshold", color=NEUTRAL, fontsize="small", va="bottom")
    ax.set_xlabel("noise ratio"); ax.set_ylabel("answer accuracy")
    ax.set_title("Noise Degradation Curve by pipeline class")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=0.3)
    return fig


def _fig_crs_distribution(plt, analysis, _extra):
    fig, ax = plt.subplots()
    width = 0.25
    for i, pclass in enumerate(PIPELINE_CLASSES):
        dist = ((analysis["descriptive"].get(pclass) or {}).get("conflict") or {}).get("distribution")
        if not dist:
            continue
        levels = list(range(5))
        total = sum(dist.get(str(k), dist.get(k, 0)) for k in levels) or 1
        vals = [dist.get(str(k), dist.get(k, 0)) / total for k in levels]
        ax.bar([k + (i - 1) * width for k in levels], vals, width, label=pclass,
               color=CLASS_COLOUR.get(pclass))
    ax.set_xlabel("CRS rubric level"); ax.set_ylabel("share of conflict cases")
    ax.set_title("Conflict Resolution Score distribution")
    ax.set_xticks(range(5)); ax.legend(); ax.grid(alpha=0.3, axis="y")
    return fig


def _fig_refusal_components(plt, analysis, _extra):
    fig, ax = plt.subplots()
    width = 0.25
    comps = ("refusal_precision", "refusal_recall", "refusal_f1")
    shades = ("#7FB3D5", "#2E86C1", "#1B4F72")
    for i, (comp, shade) in enumerate(zip(comps, shades)):
        vals = []
        for pclass in PIPELINE_CLASSES:
            ref = (analysis["descriptive"].get(pclass) or {}).get("refusal")
            if ref:
                vals.append(ref.get(comp, 0.0))
        ax.bar([j + (i - 1) * width for j in range(len(vals))], vals, width,
               label=comp.replace("refusal_", ""), color=shade)
    ax.set_xticks(range(len(PIPELINE_CLASSES))); ax.set_xticklabels(PIPELINE_CLASSES)
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Refusal F1 and its components"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    return fig


def _fig_paired_comparisons(plt, analysis, _extra):
    """Forest plot of all nine paired differences.

    The one figure that carries the whole inferential layer. Two reference
    marks, because P1 5.7 asks two different questions of every comparison: the
    zero line answers "is the difference real", and the shaded band answers "is
    it large enough to matter". A difference can clear the first and not the
    second, which is what most of them do, and that is only visible when both
    are drawn.
    """
    rows = [r for r in analysis.get("paired_comparisons", []) if "difference" in r]
    if not rows:
        return None
    rows = list(reversed(rows))  # first comparison at the top

    fig, ax = plt.subplots()
    ys = list(range(len(rows)))
    for y, row in zip(ys, rows):
        d = row["difference"]
        colour = CLASS_COLOUR.get(row["comparison"].split(" vs ")[0], NEUTRAL)
        ax.plot([d["ci_lo"], d["ci_hi"]], [y, y], color=colour, solid_capstyle="butt")
        ax.plot([d["observed"]], [y], marker="o", color=colour)

    # Practical-significance bands, one per metric, drawn behind the intervals.
    for y, row in zip(ys, rows):
        t = row.get("practical_threshold")
        if t:
            ax.plot([-t, t], [y, y], color="#DDDDDD", lw=8, zorder=0, solid_capstyle="butt")

    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['metric']}  {r['comparison']}" for r in rows])
    ax.set_xlabel("difference (positive favours the first class)")
    ax.set_title("Paired differences with 95% bootstrap intervals")
    ax.grid(alpha=0.3, axis="x")
    # Below the axes, not inside them: at nine rows the bottom interval sits
    # where an in-axes caption lands, and the caption struck through it.
    _footnote(fig, "grey band = pre-declared practical threshold")
    return fig


def _fig_metric_disagreement(plt, analysis, _extra):
    """The three conflict metrics rank the three classes differently.

    CRS is divided by 4 to share the [0, 1] axis with the other two. That is a
    presentational rescaling of the rubric, not a normalisation of meaning, and
    the axis label says so.
    """
    series = {
        "CRS / 4 (purpose-built)": [],
        "binary correct-side": [],
        "RAGAS faithfulness": [],
    }
    classes = []
    for pclass in PIPELINE_CLASSES:
        conflict = (analysis["descriptive"].get(pclass) or {}).get("conflict")
        if not conflict:
            continue
        classes.append(pclass)
        series["CRS / 4 (purpose-built)"].append(conflict["crs_mean"] / 4.0)
        series["binary correct-side"].append(conflict.get("binary_correct_side_accuracy") or 0.0)
        series["RAGAS faithfulness"].append(conflict.get("ragas_faithfulness") or 0.0)
    if not classes:
        return None

    fig, ax = plt.subplots()
    width = 0.25
    shades = ("#0072B2", "#E69F00", "#CC79A7")
    for i, ((label, vals), shade) in enumerate(zip(series.items(), shades)):
        positions = [j + (i - 1) * width for j in range(len(classes))]
        ax.bar(positions, vals, width, label=label, color=shade)
        # Rank within this metric, 1 = best. The figure's whole claim is that
        # the three metrics order the classes differently, and three sets of
        # bar heights do not make an ordering obvious; the numbers do.
        order = sorted(range(len(vals)), key=lambda k: -vals[k])
        ranks = [0] * len(vals)
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank
        for x, v, rank in zip(positions, vals, ranks):
            ax.text(x, v + 0.02, str(rank), ha="center", va="bottom",
                    fontsize="small",
                    fontweight="bold" if rank == 1 else "normal",
                    color="black" if rank == 1 else NEUTRAL)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_ylabel("score on a 0-1 scale")
    ax.set_ylim(0, 1.05)
    ax.set_title("Three conflict metrics, three different rankings")
    ax.legend(loc="upper left", framealpha=0.95); ax.grid(alpha=0.3, axis="y")
    _footnote(fig, "numbers are the rank within each metric, 1 = best")
    return fig


def _fig_agreement_statistics(plt, _analysis, extra):
    """Both kappa gates and the statistics that explain them.

    Drawn so the paradox is visible rather than argued: the dataset gate sits
    at 89% observed agreement with a kappa below zero, and the bar for the
    prevalence index -- the diagnosis -- is the tallest one in that group.
    """
    label_ag, judge_ag = extra.get("label_agreement"), extra.get("crs_judge_agreement")
    if not label_ag or not judge_ag:
        return None
    ld = label_ag.get("paradox_diagnostics") or {}
    jd = judge_ag.get("paradox_diagnostics") or {}

    names = ["Cohen's\nkappa", "observed\nagreement", "prevalence\nindex",
             "bias\nindex", "PABAK", "Gwet's\nAC1"]
    dataset = [ld.get("cohens_kappa"), ld.get("observed_agreement"),
               ld.get("prevalence_index"), ld.get("bias_index"),
               ld.get("pabak"), ld.get("gwet_ac1")]
    judge = [jd.get("cohens_kappa"), jd.get("observed_agreement"),
             None, None, None, jd.get("gwet_ac1")]

    fig, ax = plt.subplots()
    width = 0.38
    xs = range(len(names))
    ax.bar([x - width / 2 for x in xs], [v if v is not None else 0 for v in dataset],
           width, label="benchmark validation (2 annotators)", color="#0072B2")
    ax.bar([x + width / 2 for x in xs], [v if v is not None else 0 for v in judge],
           width, label="CRS judge vs human", color="#009E73")
    for x, v in zip(xs, judge):
        if v is None:
            ax.text(x + width / 2, 0.02, "n/a", ha="center", fontsize="small", color=NEUTRAL)

    ax.axhline(0.60, ls="--", color="#D55E00", lw=1.4)
    # Anchored over the bias-index column, the only low group on the chart.
    # Right-aligned it landed on top of the AC1 bar, which is the one value a
    # reader most needs to read off this figure.
    ax.text(3.0, 0.63, "P1 gate 0.60", color="#D55E00", ha="center",
            fontsize="small")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(list(xs)); ax.set_xticklabels(names)
    ax.set_ylabel("value")
    ax.set_ylim(-0.12, 1.28)
    ax.set_title("Agreement statistics: why one gate fails and one passes")
    ax.legend(loc="upper center", ncol=2, framealpha=0.95)
    ax.grid(alpha=0.3, axis="y")
    return fig


def _fig_seed_source(plt, analysis, _extra):
    """P1 5.9 threat 2, drawn. Points are the two sources; the bar is the gap."""
    data = analysis.get("by_seed_source") or {}
    rows = []
    for pclass in PIPELINE_CLASSES:
        for metric in ("refusal_f1", "crs", "ndc_auc"):
            d = (data.get(pclass) or {}).get(metric)
            if not d or "error" in d:
                continue
            rows.append((pclass, metric, d["natural_questions"], d["trivia_qa"],
                         d["difference"], d.get("exceeds_practical_threshold")))
    if not rows:
        return None

    fig, ax = plt.subplots()
    ys = list(range(len(rows)))[::-1]
    for y, (pclass, metric, nq, tqa, diff, over) in zip(ys, rows):
        ax.plot([nq, tqa], [y, y], color="#CCCCCC", lw=2, zorder=0)
        ax.plot([nq], [y], marker="o", color="#0072B2",
                label="Natural Questions" if y == ys[0] else None)
        ax.plot([tqa], [y], marker="s", color="#D55E00",
                label="TriviaQA" if y == ys[0] else None)
        if diff.get("excludes_zero"):
            ax.text(max(nq, tqa) + 0.015, y, "*" + ("*" if over else ""),
                    va="center", fontsize="small", color="black")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{p}  {m}" for p, m, *_ in rows])
    ax.set_xlabel("metric value")
    ax.set_title("Metric scores by seed dataset (P1 5.9, threat 2)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3, axis="x")
    _footnote(fig, "*  interval excludes zero      "
                   "**  also exceeds the pre-declared practical threshold")
    return fig


def _fig_within_class(plt, analysis, _extra):
    """P1 5.9 threat 3: is the architecture effect bigger than the model effect?

    The dashed line is the largest between-class difference for that metric. A
    bar rising above it means two configurations of ONE class disagree by more
    than any two classes do.
    """
    wcv = analysis.get("within_class_variation") or {}
    by_class, verdicts = wcv.get("by_class") or {}, wcv.get("between_versus_within") or {}
    metrics = [m for m in ("refusal_f1", "crs", "ndc_auc") if m in verdicts]
    if not metrics:
        return None

    fig, axes = plt.subplots(1, len(metrics), sharey=False)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        classes, retr, gen = [], [], []
        for pclass in PIPELINE_CLASSES:
            entry = (by_class.get(pclass) or {}).get(metric)
            if not entry:
                continue
            classes.append(pclass)
            retr.append(entry.get("retriever_effect") or 0.0)
            gen.append(entry.get("generator_effect") or 0.0)
        xs = range(len(classes))
        width = 0.36
        ax.bar([x - width / 2 for x in xs], retr, width, label="retriever", color="#56B4E9")
        ax.bar([x + width / 2 for x in xs], gen, width, label="generator", color="#D55E00")
        between = verdicts[metric]["largest_between_class_difference"]
        ax.axhline(between, ls="--", color="black", lw=1.2)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_title(metric)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("effect size within a class")
    # Below the panels via fig.legend. Inside the last axes it covered the
    # generator bar -- the bar the whole figure exists to show -- and above
    # them it displaced the suptitle.
    handles, labels = axes[0].get_legend_handles_labels()
    # Reserve the band FIRST, then anchor inside it. Anchoring at a negative
    # offset works at report font size and collides with the rotated tick
    # labels at slide size, where the labels are two-thirds taller.
    fig.subplots_adjust(bottom=0.30)
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Within-class variation against the largest between-class "
                 "difference (dashed)")
    return fig


def _fig_ragas_coverage_gap(plt, _analysis, extra):
    """What a reference-free tool does with the failures the metrics catch."""
    ragas = extra.get("ragas_analysis")
    if not ragas:
        return None
    gaps = ragas.get("coverage_gap") or {}
    dims = [d for d in ("refusal", "conflict", "noise") if d in gaps]
    if not dims:
        return None

    flagged, missed, undefined = [], [], []
    for d in dims:
        row = (gaps[d] or {}).get("faithfulness")
        if not row:
            flagged.append(0); missed.append(0); undefined.append(0); continue
        n_fail = row["n_failures_by_purpose_built_metric"]
        n_missed = row["n_failures_ragas_scores_as_passing"]
        n_undef = row["n_failures_ragas_cannot_score"]
        missed.append(n_missed); undefined.append(n_undef)
        flagged.append(max(0, n_fail - n_missed - n_undef))

    fig, ax = plt.subplots()
    xs = range(len(dims))
    ax.bar(xs, flagged, label="RAGAS also scores it low", color="#009E73")
    ax.bar(xs, missed, bottom=flagged, label="RAGAS scores it as passing", color="#D55E00")
    bottoms = [f + m for f, m in zip(flagged, missed)]
    ax.bar(xs, undefined, bottom=bottoms, label="RAGAS cannot score it", color="#999999")
    # The miss rate printed on the segment it describes. The dimensions differ
    # fivefold in absolute size, so the bars alone invite a comparison of
    # heights when the comparable quantity is the proportion.
    for x, (f, m, u) in enumerate(zip(flagged, missed, undefined)):
        scored = f + m
        if scored:
            ax.text(x, f + m / 2.0, f"{m / scored:.0%}\nmissed", ha="center",
                    va="center", color="white", fontsize="small", fontweight="bold")
    ax.set_xticks(list(xs)); ax.set_xticklabels(dims)
    ax.set_ylabel("failures identified by the purpose-built metric")
    ax.set_title("RAGAS faithfulness on the failures the purpose-built metrics catch")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3, axis="y")
    return fig


def _fig_ragas_vs_noise(plt, analysis, extra, *, legend_pt: int = 7):
    """RAGAS across the noise axis the NDC is built on.

    The purpose-built curve is drawn beside it deliberately. RAGAS context
    relevance is a ratio of selected to TOTAL sentences, so adding distractors
    inflates its denominator and it has a mechanical reason to respond -- which
    makes a flat or non-monotonic curve the stronger negative result.
    """
    ragas = extra.get("ragas_analysis")
    if not ragas:
        return None
    by_ratio = ((ragas.get("noise_ratio_response") or {}).get("by_ratio")) or {}
    if not by_ratio:
        return None
    xs = sorted(float(k) for k in by_ratio)

    fig, ax = plt.subplots()
    styles = {"faithfulness": ("#0072B2", "o"), "answer_relevance": ("#E69F00", "s"),
              "context_relevance": ("#CC79A7", "^")}
    for metric, (colour, marker) in styles.items():
        ys = [by_ratio[f"{x:.2f}"].get(metric) for x in xs]
        if all(y is None for y in ys):
            continue
        ax.plot(xs, [y if y is not None else float("nan") for y in ys],
                marker=marker, color=colour, label=metric.replace("_", " "))

    acc = (analysis["descriptive"].get("naive") or {}).get("noise_accuracy_by_ratio")
    if acc:
        ax.plot(sorted(float(k) for k in acc),
                [acc[str(x)] for x in sorted(float(k) for k in acc)],
                ls="--", color=NEUTRAL, marker="x",
                label="answer accuracy (naive)")

    ax.set_xlabel("noise ratio"); ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Does RAGAS track the noise ratio?")
    # The legend goes in the empty band between the context-relevance line
    # (below 0.08 throughout) and the accuracy and answer-relevance lines
    # (above 0.49 throughout). Anchored in axes coordinates so the same gap is
    # found at projector size, and made compact -- shorter labels, tighter
    # spacing, smaller type -- because a legend sized for the default padding
    # covered two of the four lines this figure exists to compare.
    ax.legend(loc="lower left", bbox_to_anchor=(0.015, 0.13),
              fontsize=legend_pt, framealpha=0.9, labelspacing=0.3,
              handlelength=1.6, handletextpad=0.5, borderpad=0.4,
              borderaxespad=0.0, title="RAGAS metrics, and the NDC axis",
              title_fontsize=legend_pt)
    ax.grid(alpha=0.3)
    return fig


FIGURE_BUILDERS = {
    "ndc_by_class": _fig_ndc,
    "crs_distribution": _fig_crs_distribution,
    "refusal_components": _fig_refusal_components,
    "paired_comparisons": _fig_paired_comparisons,
    "metric_disagreement": _fig_metric_disagreement,
    "agreement_statistics": _fig_agreement_statistics,
    "seed_source_comparison": _fig_seed_source,
    "within_class_variation": _fig_within_class,
    "ragas_coverage_gap": _fig_ragas_coverage_gap,
    "ragas_vs_noise_ratio": _fig_ragas_vs_noise,
}


def plot(analysis: dict, out_dir: Path, *, runs_dir: Path | None = None) -> dict:
    """Write every figure at report size and again at projector size.

    A builder returning None means the artefact that figure needs has not been
    produced yet. That is reported as a skip rather than an error, so the
    analysis still runs before the RAGAS or validation stages have been.
    """
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    runs = runs_dir or (ROOT / "runs")
    extra = {
        "ragas_analysis": _load_optional(runs / "ragas_analysis.json"),
        "label_agreement": _load_optional(runs / "validation" / "label_agreement.json"),
        "crs_judge_agreement": _load_optional(runs / "validation" / "crs_judge_agreement.json"),
        "generation_artefacts": _load_optional(runs / "generation_artefacts.json"),
    }

    written: list[str] = []
    skipped: list[str] = []
    for variant, style, target in (
        ("report", REPORT_STYLE, out_dir),
        ("slides", SLIDE_STYLE, out_dir / "slides"),
    ):
        target.mkdir(parents=True, exist_ok=True)
        for name, builder in FIGURE_BUILDERS.items():
            with plt.rc_context(style):
                kwargs = ({"legend_pt": LEGEND_PT[variant]}
                          if builder is _fig_ragas_vs_noise else {})
                fig = builder(plt, analysis, extra, **kwargs)
                if fig is None:
                    if variant == "report":
                        skipped.append(name)
                    continue
                path = target / f"{name}.png"
                # bbox_inches="tight" rather than tight_layout(): the latter
                # recomputes the margins and discards the room reserved for a
                # figure-level footnote.
                fig.savefig(path, bbox_inches="tight")
                plt.close(fig)
                if variant == "report":
                    written.append(str(path))
    return {"report": written, "slides_dir": str(out_dir / "slides"),
            "skipped_missing_artefact": skipped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", default="runs/scored.jsonl")
    ap.add_argument("--dataset-config", default="configs/dataset.yaml")
    ap.add_argument("--out", default="runs/analysis.json")
    ap.add_argument("--plot-dir", default="runs/figures")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--ragas-scores", default="runs/ragas_scores.jsonl",
                    help="per-instance RAGAS scores, for the P1 5.4 faithfulness "
                         "comparator on the CRS row")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.dataset_config).read_text())["analysis"]
    by_class = load_scored(Path(args.scored))
    print(f"loaded {sum(len(v) for v in by_class.values()):,} scored instances "
          f"across {len(by_class)} pipeline classes")
    print(f"bootstrap: {cfg['n_bootstrap']} resamples at {cfg['confidence']:.0%}, "
          f"seed {cfg['rng_seed']}")
    print(f"practical-significance thresholds (declared in advance): {cfg['practical_significance']}")

    ragas_f = load_ragas_faithfulness(Path(args.ragas_scores))
    if ragas_f:
        print(f"RAGAS faithfulness comparator available for {sorted(ragas_f)}")
    else:
        print("no RAGAS scores found; the CRS row's faithfulness comparator "
              "will be null (run scripts/ragas_analysis.py)")

    analysis = {
        "config": cfg,
        "descriptive": describe(by_class, cfg, ragas_f),
        "paired_comparisons": compare(by_class, cfg),
        # P1 5.9 threat 2 -- reported per class, per metric, per seed dataset.
        "by_seed_source": by_seed_source(by_class, cfg),
        "divergent_cases": divergent_cases(by_class),
    }
    # P1 5.9 threat 3 -- needs the paired comparisons, so it is attached after.
    analysis["within_class_variation"] = within_class_variation(
        load_scored_by_configuration(Path(args.scored)), analysis["paired_comparisons"]
    )
    if not args.no_plots:
        analysis["figures"] = plot(analysis, Path(args.plot_dir))

    Path(args.out).write_text(json.dumps(analysis, indent=2))
    print(f"\nanalysis -> {args.out}")

    print("\n=== paired comparisons ===")
    for row in analysis["paired_comparisons"]:
        if "error" in row:
            print(f"  {row['metric']:10s} {row['comparison']:24s} {row['error'][:60]}")
            continue
        d = row["difference"]
        flag = ("PRACTICAL" if row["practically_significant"]
                else "significant" if row["statistically_significant"] else "-")
        print(f"  {row['metric']:10s} {row['comparison']:24s} "
              f"diff={d['observed']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
