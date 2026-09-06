#!/usr/bin/env python3
"""Assert every number in the writing pack against the artefact it came from.

    python scripts/verify_pack.py

The pack in `for_cowork/WRITING_PACK.md` is the factual basis for Chapters
6-9, and it will be read by a writing session that cannot open the JSON
artefacts to check anything. A transcription slip there becomes a wrong number
in the submitted report with nothing downstream to catch it.

So the pack carries a machine-checkable ledger (its section 8), one row per
headline figure:

    | id | value | artefact | json path |

This script parses that table, resolves each JSON path, and compares. Run it
after regenerating any artefact, and before handing the pack over.

It deliberately checks the LEDGER rather than scanning the prose for numerals.
Scanning would flag every sample size, section number and year, and a check
that cries wolf is a check nobody runs. The ledger is the contract: a figure
quoted in the prose must appear there, and section 0 of the pack tells the
writing session to use nothing else.

One exception is checked in the prose: counts of the nine paired comparisons.
A ledger row holds a value, so a row cannot express "six of nine differences
are statistically significant" -- and that sentence was wrong for a day, saying
five, while the table beside it correctly marked six. It is the one headline
claim that is a count over the artefact rather than a value in it, so it gets
its own check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACK = ROOT / "for_cowork" / "WRITING_PACK.md"

# A ledger row: four pipe-delimited cells, the third ending in .json.
_ROW = re.compile(
    r"^\|\s*([A-Za-z0-9_.]+)\s*\|\s*(-?[0-9a-f.]+)\s*\|\s*(\S+\.json)\s*\|\s*([A-Za-z0-9_.]+)\s*\|\s*$"
)


def resolve(blob: dict, path: str):
    """Walk a dotted JSON path, tolerating dict keys that are numeric strings."""
    node = blob
    for part in path.split("."):
        if isinstance(node, dict):
            if part in node:
                node = node[part]
                continue
            raise KeyError(part)
        raise TypeError(f"cannot index {type(node).__name__} with {part!r}")
    return node


_NUMBER_WORDS = {
    "zero": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
}

# "Six of nine differences are ...", "Five of nine comparisons exclude zero".
_OF_NINE = re.compile(r"\b([A-Za-z]+|[0-9]+) of nine\b", re.IGNORECASE)


def _spelled(token: str) -> int | None:
    """Read 'six' or '6' as 6; anything else is not a count claim."""
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token.lower())


def check_of_nine_claims(text: str, comparisons: list[dict]) -> list[str]:
    """Check every 'N of nine' sentence against the paired-comparison records.

    Both counts live in the same nine records, so which one a sentence means is
    decided by its wording: a claim about the practical threshold is compared
    against `practically_significant`, everything else against
    `statistically_significant`. Returns one message per mismatch.
    """
    n_stat = sum(1 for c in comparisons if c["statistically_significant"])
    n_prac = sum(1 for c in comparisons if c["practically_significant"])

    failures: list[str] = []
    for m in _OF_NINE.finditer(text):
        stated = _spelled(m.group(1))
        if stated is None:
            continue
        # Both counts are usually named in the same sentence ("six of nine are
        # statistically significant; none clears the practical threshold"), so
        # a keyword-anywhere test picks the wrong one. Whichever marker comes
        # FIRST after the count is the one the count is about.
        # Whitespace-normalised: these documents hard-wrap, so a marker
        # phrase is routinely split across two lines.
        window = " ".join(text[m.start(): m.start() + 200].lower().split())
        stat_at = min((i for i in (window.find("statistically significant"),
                                   window.find("exclude zero"),
                                   window.find("excluding zero")) if i >= 0),
                      default=len(window))
        prac_at = min((i for i in (window.find("practical"),
                                   window.find("threshold")) if i >= 0),
                      default=len(window))
        practical = prac_at < stat_at
        expected = n_prac if practical else n_stat
        if stated != expected:
            kind = "practically" if practical else "statistically"
            failures.append(
                f"{m.group(0)!r}: says {stated} {kind} significant, "
                f"analysis.json has {expected}"
            )
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(PACK))
    ap.add_argument("--runs", default=str(ROOT))
    args = ap.parse_args()

    pack = Path(args.pack)
    if not pack.exists():
        print(f"missing pack: {pack}", file=sys.stderr)
        return 2

    rows = []
    for line in pack.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            rows.append(m.groups())
    if not rows:
        print("no ledger rows found; has section 8's table format changed?",
              file=sys.stderr)
        return 2

    cache: dict[str, dict] = {}
    failures: list[str] = []
    checked = 0

    for ident, stated, artefact, path in rows:
        full = Path(args.runs) / artefact
        if artefact not in cache:
            if not full.exists():
                failures.append(f"{ident}: artefact {artefact} not found")
                continue
            cache[artefact] = json.loads(full.read_text())
        try:
            actual = resolve(cache[artefact], path)
        except (KeyError, TypeError) as exc:
            failures.append(f"{ident}: path {path} not in {artefact} ({exc})")
            continue

        checked += 1
        if isinstance(actual, str):
            ok = actual == stated
        else:
            try:
                # Compared at the pack's own precision. The pack rounds to four
                # decimals; requiring exact float equality would fail on values
                # the artefact stores at full precision.
                ok = abs(float(actual) - float(stated)) < 5e-5
            except (TypeError, ValueError):
                ok = False
        if not ok:
            failures.append(
                f"{ident}: pack says {stated}, {artefact}:{path} says {actual}"
            )

    # The prose claim, checked over the same artefact the table is built from.
    analysis = Path(args.runs) / "runs" / "analysis.json"
    claims_checked = 0
    if analysis.exists():
        comparisons = json.loads(analysis.read_text())["paired_comparisons"]
        prose = pack.read_text(encoding="utf-8")
        for extra in (pack.parent / "README.md",):
            if extra.exists():
                prose += "\n" + extra.read_text(encoding="utf-8")
        claim_failures = check_of_nine_claims(prose, comparisons)
        claims_checked = len(_OF_NINE.findall(prose))
        failures.extend(claim_failures)

    print(f"checked {checked} of {len(rows)} ledger rows against their artefacts")
    print(f"checked {claims_checked} 'N of nine' prose claims against runs/analysis.json")
    if failures:
        print(f"\n{len(failures)} MISMATCH(ES):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("all ledger values match their artefacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
