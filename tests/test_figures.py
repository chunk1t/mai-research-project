"""The figure suite (P1 Section 5.7).

Figure builders are presentational -- they read values already computed and
tested upstream -- so these tests do not re-check arithmetic. They check the
two things that actually break: that a builder whose artefact has not been
produced yet SKIPS rather than crashing, and that every declared figure is
written at both sizes.

The first matters because `analyse_results.py` has to run before the RAGAS and
validation stages have. A builder that raised on a missing file would make the
main analysis depend on optional side analyses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

pytest.importorskip("matplotlib", reason="figures are an optional extra")

from analyse_results import FIGURE_BUILDERS, plot  # noqa: E402


def _minimal_analysis() -> dict:
    """The smallest analysis dict the class-level figures can be drawn from."""
    per_class = {
        "refusal": {"refusal_precision": 0.8, "refusal_recall": 0.9, "refusal_f1": 0.85},
        "conflict": {"crs_mean": 0.2, "distribution": {"0": 8, "1": 1, "2": 1, "3": 0, "4": 0},
                     "binary_correct_side_accuracy": 0.3, "ragas_faithfulness": 0.7},
        "noise_accuracy_by_ratio": {"0.0": 0.6, "0.25": 0.58, "0.5": 0.55,
                                    "0.75": 0.52, "0.9": 0.5},
    }
    return {
        "descriptive": {c: dict(per_class) for c in ("naive", "reasoning", "agentic")},
        "paired_comparisons": [
            {"metric": "crs", "comparison": "reasoning vs naive",
             "difference": {"observed": 0.18, "ci_lo": 0.12, "ci_hi": 0.24,
                            "excludes_zero": True},
             "practical_threshold": 0.25},
        ],
        "by_seed_source": {
            "naive": {"refusal_f1": {
                "natural_questions": 0.93, "trivia_qa": 0.80,
                "difference": {"observed": 0.13, "ci_lo": 0.04, "ci_hi": 0.23,
                               "excludes_zero": True},
                "exceeds_practical_threshold": True}}},
        "within_class_variation": {
            "by_class": {"naive": {"crs": {"retriever_effect": 0.005,
                                           "generator_effect": 0.025}}},
            "between_versus_within": {"crs": {"largest_between_class_difference": 0.178}},
        },
    }


def test_figures_that_need_a_missing_artefact_are_skipped_not_fatal(tmp_path):
    """The main analysis must not depend on the optional side analyses.

    `runs_dir` points at an empty directory, so the RAGAS and validation
    artefacts are absent. Those figures must be reported as skipped and the
    rest must still be written.
    """
    empty = tmp_path / "runs"
    empty.mkdir()
    out = plot(_minimal_analysis(), tmp_path / "figures", runs_dir=empty)

    assert "ragas_coverage_gap" in out["skipped_missing_artefact"]
    assert "agreement_statistics" in out["skipped_missing_artefact"]
    assert any("ndc_by_class" in p for p in out["report"])
    assert (tmp_path / "figures" / "ndc_by_class.png").exists()


def test_every_written_figure_appears_at_both_sizes(tmp_path):
    runs = tmp_path / "runs"
    (runs / "validation").mkdir(parents=True)
    (runs / "ragas_analysis.json").write_text(json.dumps({
        "coverage_gap": {"conflict": {"faithfulness": {
            "n_failures_by_purpose_built_metric": 10,
            "n_failures_ragas_scores_as_passing": 4,
            "n_failures_ragas_cannot_score": 3}}},
        "noise_ratio_response": {"by_ratio": {
            "0.00": {"faithfulness": 0.8, "answer_relevance": 0.5,
                     "context_relevance": 0.03},
            "0.90": {"faithfulness": 0.7, "answer_relevance": 0.5,
                     "context_relevance": 0.01}}},
    }))
    (runs / "validation" / "label_agreement.json").write_text(json.dumps({
        "paradox_diagnostics": {"cohens_kappa": -0.03, "observed_agreement": 0.89,
                                "prevalence_index": 0.89, "bias_index": 0.07,
                                "pabak": 0.78, "gwet_ac1": 0.88}}))
    (runs / "validation" / "crs_judge_agreement.json").write_text(json.dumps({
        "paradox_diagnostics": {"cohens_kappa": 0.83, "observed_agreement": 0.98,
                                "gwet_ac1": 0.98}}))

    out = plot(_minimal_analysis(), tmp_path / "figures", runs_dir=runs)
    assert out["skipped_missing_artefact"] == []

    for name in FIGURE_BUILDERS:
        assert (tmp_path / "figures" / f"{name}.png").exists(), name
        assert (tmp_path / "figures" / "slides" / f"{name}.png").exists(), name


def test_the_slides_variant_is_a_different_rendering_not_a_copy(tmp_path):
    # Larger type and a wider canvas, so the two files must differ in bytes.
    # A silent fall-through to the same rcParams would give an identical file
    # and an unreadable projection.
    out = plot(_minimal_analysis(), tmp_path / "figures", runs_dir=tmp_path / "nope")
    report = (tmp_path / "figures" / "ndc_by_class.png").read_bytes()
    slides = (tmp_path / "figures" / "slides" / "ndc_by_class.png").read_bytes()
    assert report != slides
    assert len(slides) > len(report)
    assert out["slides_dir"].endswith("slides")
