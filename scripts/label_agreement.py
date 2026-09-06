#!/usr/bin/env python3
"""Inter-rater agreement over the manual validation sheets (P1 Objective 1, 5.5.3).

    python scripts/label_agreement.py

P1 5.5.3 requires "a sub-sample of at least fifty cases reviewed by a second
annotator", with agreement reported as Cohen's kappa against a 0.60 gate.

This script exists because that number was previously computed ad hoc. The gate
is load-bearing for Objective 1, so it has to come out of a committed,
re-runnable artefact that Chapter 6 can cite -- not out of a session transcript.

It reports Cohen's kappa AND the statistics that explain it. On this benchmark
almost every sampled case passes review, which is a good property of the dataset
and a bad one for kappa: expected agreement approaches observed agreement and
the coefficient collapses regardless of how well the raters actually agree. The
prevalence and bias indices of Byrt, Bishop and Carlin (1993) separate those two
explanations, and Gwet's AC1 (2008) is the chance-corrected coefficient that
does not degrade under the skew. See `metrics/conflict.agreement_diagnostics`.

The gate itself stays keyed on Cohen's kappa. None of the extra statistics can
move it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragrobust.dataset.validation import compute_label_agreement  # noqa: E402

OUT = ROOT / "runs" / "validation"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--primary", default="runs/validation/primary_labels.csv")
    p.add_argument("--second", default="runs/validation/second_annotator_labels.csv")
    p.add_argument(
        "--rater-kind",
        default="inter",
        choices=["inter", "intra"],
        help="'inter' = two annotators (the P1 5.5.3 gate); 'intra' = one "
             "annotator labelling twice, which is test-retest reliability and "
             "must NOT be reported as the gate",
    )
    p.add_argument("--out", default="runs/validation/label_agreement.json")
    args = p.parse_args()

    a, b = ROOT / args.primary, ROOT / args.second
    for path in (a, b):
        if not path.exists():
            print(f"missing label sheet: {path}", file=sys.stderr)
            return 2

    result = compute_label_agreement(a, b, rater_kind=args.rater_kind)
    payload = result.as_dict()

    d = result.diagnostics
    if d is not None and d.paradox_detected:
        payload["gate_interpretation"] = (
            f"Cohen's kappa is {d.kappa:.4f} at {d.observed_agreement:.1%} observed "
            f"agreement and a prevalence index of {d.prevalence_index:.2f}. That "
            f"combination is the kappa paradox (Feinstein & Cicchetti 1990): the "
            f"raters agree on almost every item and almost always in the same "
            f"category, so expected agreement is near-total and kappa has no room "
            f"to be positive. Gwet's AC1 = {d.gwet_ac1:.4f} and PABAK = "
            f"{d.pabak:.4f} on the same data. REPORT ALL OF THEM: the P1 gate is "
            f"not met on its stated statistic, and the reason is a property of "
            f"the sample, not evidence that the annotators disagreed."
        )
    else:
        payload["gate_interpretation"] = (
            "Cohen's kappa is interpretable here; the prevalence diagnostics do "
            "not indicate a paradox."
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
