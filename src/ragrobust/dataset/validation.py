"""Manual validation sampling (P1 Section 5.5.3).

P1 specifies a two-tier protocol and the tiers are easy to conflate:

    "A random sample of approximately fifteen to twenty percent of the generated
    cases, drawn proportionally from the three testbeds, is reviewed by hand to
    confirm that the perturbation behaves as intended. [...] Inter-rater
    agreement on the manual validation is established by having a sub-sample of
    at least fifty cases reviewed by a second annotator."

So the PRIMARY sample (15-20%, ~158 cases) is reviewed by the researcher, and a
SUB-SAMPLE of it (>= 50) is reviewed by the second annotator as well. Cohen's
kappa is computed only on the overlap, because kappa needs two independent
labels for the same item. Sampling the second annotator's cases independently of
the primary sample would leave no overlap and no computable kappa at all.

The per-dimension checklists below are P1's own review criteria, kept verbatim
so the annotator applies the specified test rather than a paraphrase.
"""

from __future__ import annotations

import csv
import html
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..schema import Dimension, TestCase

if TYPE_CHECKING:  # pragma: no cover
    from ..metrics.conflict import AgreementDiagnostics

# P1 5.5.3, verbatim. Each is answered yes/no per case; any "no" fails review
# and the case is repaired or discarded, with the decision reported.
CHECKLISTS: dict[Dimension, tuple[str, ...]] = {
    Dimension.REFUSAL: (
        "No information in the retrieved context can be used to derive the "
        "answer, including by combining multiple passages.",
    ),
    Dimension.CONFLICT: (
        "The two passages do indeed contradict each other on the target claim.",
        "The contradiction is plausible (not absurd).",
        "Both passages are on the original topic.",
    ),
    Dimension.NOISE: (
        "The distractor passages are topically related to the query.",
        "The distractor passages do not contain the answer.",
    ),
}

# Answerable refusal controls are the mirror test: the answer MUST be derivable,
# or the control is not a control and Refusal F1 precision is measured against a
# broken baseline.
ANSWERABLE_CONTROL_CHECK = (
    "The retrieved context DOES support the answer (this is an answerable control).",
)


def _round_half_up(x: float) -> int:
    """Deterministic rounding.

    `round()` is banker's rounding, so round(87.5) == 88 but round(86.5) == 86.
    A sample size that depends on the parity of a testbed count is not something
    anyone should have to reason about when reproducing the sample.
    """
    return int(x + 0.5)


@dataclass
class ValidationSample:
    primary: list[TestCase]
    second_annotator: list[TestCase]
    stats: dict[str, object] = field(default_factory=dict)


def stratified_validation_sample(
    cases: list[TestCase],
    *,
    fraction: float = 0.175,
    second_annotator_n: int = 50,
    rng_seed: int = 20260721,
    exclude_zero_noise: bool = True,
) -> ValidationSample:
    """Draw the P1 5.5.3 primary sample and its second-annotator sub-sample.

    Sampling is over BASE CASES, not evaluation instances. P1 describes the
    benchmark as "approximately nine hundred base test cases" made of "two
    hundred conflict cases, and five hundred noise cases", so a noise seed is
    one case whose five noise ratios are expansions of it. Sampling instances
    instead would draw the same seed up to five times and quadruple the
    reviewer's workload for near-duplicate judgements -- P1's noise question
    ("are the distractors topically related and answer-free?") has essentially
    the same answer at every ratio for a given seed.

    One representative instance is emitted per sampled noise seed, with the
    ratio rotated across the sample so the reviewer still sees the full range.

    Proportional by dimension, as P1 requires ("drawn proportionally from the
    three testbeds"), so the sample mirrors the benchmark rather than
    over-weighting the small testbeds.

    `exclude_zero_noise` drops r=0 instances from the pool of representatives.
    P1's noise check is vacuous when there are no distractors, and the r=0
    integrity is already machine-checked by the verify stage. Kept as a flag
    rather than hardcoded so the choice is visible and reversible.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between 0 and 1")

    # Group into base cases: one per (dimension, seed_id). Noise contributes
    # several instances per group, every other dimension exactly one.
    groups: dict[Dimension, dict[str, list[TestCase]]] = {}
    for c in cases:
        groups.setdefault(c.dimension, {}).setdefault(c.seed_id, []).append(c)

    rng = random.Random(rng_seed)
    primary: list[TestCase] = []
    per_dim_counts: dict[str, int] = {}
    n_base_total = 0
    for dim in sorted(groups, key=lambda d: d.value):
        keys = sorted(groups[dim])           # stable order before shuffling
        n_base_total += len(keys)
        rng.shuffle(keys)
        take = min(_round_half_up(fraction * len(keys)), len(keys))
        for i, key in enumerate(keys[:take]):
            members = sorted(groups[dim][key], key=lambda c: (c.noise_ratio or 0.0, c.case_id))
            if exclude_zero_noise:
                nonzero = [m for m in members if (m.noise_ratio or 0.0) > 0.0]
                # Refusal and conflict carry no ratio at all, so they keep their
                # single member; only noise is actually filtered here.
                if nonzero:
                    members = nonzero
            # Rotate the representative so the sample spans the ratio range
            # instead of always showing the reviewer the same one.
            primary.append(members[i % len(members)])
        per_dim_counts[dim.value] = take

    second: list[TestCase] = []
    second_counts: dict[str, int] = {}
    total_primary = len(primary)
    if second_annotator_n > total_primary:
        raise ValueError(
            f"second_annotator_n={second_annotator_n} exceeds the primary sample "
            f"of {total_primary}; kappa needs both raters on the same cases"
        )
    prim_by_dim: dict[Dimension, list[TestCase]] = {}
    for c in primary:
        prim_by_dim.setdefault(c.dimension, []).append(c)

    # Proportional allocation, then a largest-remainder top-up.
    #
    # Rounding each dimension independently can land BELOW the requested total:
    # on the pruned benchmark (32 refusal, 35 conflict, 88 noise) the shares are
    # 10.32, 11.29 and 28.39, which round to 10 + 11 + 28 = 49. P1 5.5.3 asks
    # for "a sub-sample of AT LEAST fifty cases", so 49 silently misses the
    # requirement -- and the kappa is reported against that count.
    dims = sorted(prim_by_dim, key=lambda d: d.value)
    exact = {d: second_annotator_n * len(prim_by_dim[d]) / total_primary for d in dims}
    take = {d: min(int(exact[d]), len(prim_by_dim[d])) for d in dims}

    # Whole cases go to the largest fractional remainders first, which is what
    # keeps the topped-up sub-sample as close to proportional as integers allow.
    order = sorted(dims, key=lambda d: (-(exact[d] - int(exact[d])), d.value))
    i = 0
    while sum(take.values()) < second_annotator_n and any(
        take[d] < len(prim_by_dim[d]) for d in dims
    ):
        d = order[i % len(order)]
        if take[d] < len(prim_by_dim[d]):
            take[d] += 1
        i += 1

    for dim in dims:
        second.extend(prim_by_dim[dim][: take[dim]])
        second_counts[dim.value] = take[dim]

    return ValidationSample(
        primary=primary,
        second_annotator=second,
        stats={
            "base_cases_total": n_base_total,
            "sampling_unit": "base_case",
            "fraction_requested": fraction,
            "fraction_achieved": round(len(primary) / n_base_total, 4) if n_base_total else 0.0,
            "primary_n": len(primary),
            "primary_by_dimension": per_dim_counts,
            "second_annotator_n": len(second),
            "second_annotator_by_dimension": second_counts,
            "rng_seed": rng_seed,
            "zero_noise_excluded": exclude_zero_noise,
        },
    )


def read_review_failures(paths: Iterable[Path]) -> dict[str, list[str]]:
    """Collect the cases a human annotator failed, from filled label CSVs.

    P1 Section 5.5.3: "Cases that fail review are repaired or discarded. The
    repair-or-discard decision and the final acceptance rate are reported with
    the benchmark release." The machine half of that already runs in the prune
    stage; this is the half that carries the *human* verdict, which previously
    had no way of reaching it at all.

    A case fails if ANY of its checklist questions is answered "no" -- the
    checklists are conjunctive, since each question tests a different way the
    perturbation can be wrong. Blank answers mean "not yet reviewed" and are not
    failures; treating them as such would discard the whole benchmark on the
    first run.

    Returns case_id -> the reasons it failed, so the discard report can say why
    rather than just how many. Later files win nothing and lose nothing: reasons
    accumulate, because two annotators failing the same case for different
    reasons is a stronger finding, not a conflict to resolve.
    """
    failures: dict[str, list[str]] = {}
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("answer_yes_no") or "").strip().lower() != "no":
                    continue
                note = (row.get("notes") or "").strip()
                reason = row.get("question", "").strip()
                if note:
                    reason = f"{reason} [{note}]"
                bucket = failures.setdefault(row["case_id"], [])
                if reason not in bucket:
                    bucket.append(reason)
    return failures


def read_reviewed_case_ids(paths: Iterable[Path]) -> set[str]:
    """Every case id carrying a human verdict, pass or fail."""
    seen: set[str] = set()
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if (row.get("answer_yes_no") or "").strip():
                    seen.add(row["case_id"])
    return seen


def stale_review_case_ids(
    paths: Iterable[Path],
    benchmark_case_ids: Iterable[str],
    *,
    discarded_case_ids: Iterable[str] = (),
) -> list[str]:
    """Reviewed cases that are no longer in the benchmark.

    `discarded_case_ids` are the cases a reviewer deliberately failed and the
    prune stage then removed. They are absent from the benchmark BY DESIGN, so
    counting them as stale would make the warning fire loudest exactly when the
    repair-or-discard loop worked correctly -- and a warning that always fires
    gets ignored, which is how the real thing would slip past.

    A label describes the case as it was when it was judged. Seeds are streamed
    from Natural Questions and TriviaQA, so an upstream change to either corpus
    -- or any rebuild that re-draws the sample -- can leave a verdict pointing at
    a case id the benchmark no longer contains.

    That failure is silent and it has already happened once here: a noise case
    keeps its case_id while its distractors are replaced wholesale, so verdicts
    carried forward by id alone described distractors that no longer existed.
    Surfacing the mismatch is the difference between a stale label being noticed
    and a stale label being reported as a kappa.
    """
    return sorted(
        read_reviewed_case_ids(paths) - set(benchmark_case_ids) - set(discarded_case_ids)
    )


def checklist_for(case: TestCase) -> tuple[str, ...]:
    """The P1 5.5.3 review questions that apply to one case."""
    if case.dimension is Dimension.REFUSAL and case.is_answerable:
        return ANSWERABLE_CONTROL_CHECK
    return CHECKLISTS[case.dimension]


# --------------------------------------------------------------------------
# Review sheet rendering
#
# Seed evidence is *prepared, not filtered* (HANDOFF locked decision), so NQ
# passages keep the raw Wikipedia markup token stream -- <P>, <H2>, <Table>,
# <Tr>, <Th colspan="2">, <Ul>, <Li>. Escaping that into the review sheet turned
# a wide infobox into several hundred literal "<Td>" tokens, which is unreadable
# and makes the annotator's job (is the answer derivable? do the passages
# contradict?) harder than the judgement itself warrants.
#
# So the sheet renders that markup as real HTML. The transform is presentation
# only: every character of passage text survives it, and the sheet carries a
# raw-markup toggle so the annotator can always see the exact string the
# pipelines receive. Nothing here touches the cases themselves.
# --------------------------------------------------------------------------

# The complete tag inventory of the built benchmark, measured rather than
# guessed. Anything outside this set is escaped and shown as literal text, so a
# stray "<" in passage prose can never inject markup into the sheet.
_ALLOWED_TAGS: frozenset[str] = frozenset(
    {"p", "h1", "h2", "h3", "h4", "h5", "h6",
     "table", "tr", "td", "th", "ul", "ol", "li", "dl", "dt", "dd"}
)

# Passages are windows centred on the answer, so they routinely start or end
# mid-table and arrive with unbalanced markup. A browser silently DROPS an
# orphan <td>, taking its text with it, so the missing ancestors are supplied
# here instead. First entry is the one to open when none is present.
_PARENTS: dict[str, tuple[str, ...]] = {
    "td": ("tr",), "th": ("tr",), "tr": ("table",),
    "li": ("ul", "ol"), "dt": ("dl",), "dd": ("dl",),
}

# Elements whose end tag Wikipedia markup may omit: opening one of these closes
# any of the listed siblings still on the stack.
_CLOSES: dict[str, frozenset[str]] = {
    "td": frozenset({"td", "th"}),
    "th": frozenset({"td", "th"}),
    "tr": frozenset({"td", "th", "tr"}),
    "li": frozenset({"li"}),
    "dt": frozenset({"dt", "dd"}),
    "dd": frozenset({"dt", "dd"}),
    "p": frozenset({"p"}),
}

# Only spanning attributes are carried over; every other attribute is dropped.
# The values are re-emitted from a digit match, never from the source string.
_TAG_RE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9]*)((?:[^<>]*)?)>")
_SPAN_RE = re.compile(r"\b(colspan|rowspan)\s*=\s*\"?(\d{1,3})\"?", re.IGNORECASE)


def render_passage_html(text: str) -> str:
    """Render one passage's Wikipedia markup as safe, well-formed HTML.

    Whitelisted tags become real elements; everything else -- unknown tags,
    stray angle brackets, the passage prose itself -- is escaped. Unbalanced
    markup is repaired (implied parents opened, stray end tags dropped, open
    elements closed at the end) so a passage that begins mid-table still shows
    all of its text in document order.
    """
    out: list[str] = []
    stack: list[str] = []

    def open_tag(name: str, attrs: str = "") -> None:
        for parent in _PARENTS.get(name, ()):
            if parent in stack:
                break
        else:  # no acceptable ancestor is open -- supply the first one
            if name in _PARENTS:
                open_tag(_PARENTS[name][0])
        while stack and stack[-1] in _CLOSES.get(name, frozenset()):
            out.append(f"</{stack.pop()}>")
        keep = "".join(
            f' {k.lower()}="{v}"' for k, v in _SPAN_RE.findall(attrs)
        ) if name in ("td", "th") else ""
        out.append(f"<{name}{keep}>")
        stack.append(name)

    def emit_text(raw: str) -> None:
        # Text sitting directly inside a table or row is foster-parented by the
        # browser (hoisted out above the table), which silently reorders the
        # passage. Give it a cell of its own instead.
        if raw.strip() and stack and stack[-1] in ("table", "tr"):
            open_tag("td")
        out.append(html.escape(raw))

    pos = 0
    for m in _TAG_RE.finditer(text):
        emit_text(text[pos:m.start()])
        pos = m.end()
        closing, name, attrs = bool(m.group(1)), m.group(2).lower(), m.group(3)
        if name not in _ALLOWED_TAGS:
            out.append(html.escape(m.group(0)))
            continue
        if not closing:
            open_tag(name, attrs)
        elif name in stack:
            while stack:
                popped = stack.pop()
                out.append(f"</{popped}>")
                if popped == name:
                    break
        # else: an end tag for an element the window never opened -- drop it.
    emit_text(text[pos:])
    while stack:
        out.append(f"</{stack.pop()}>")
    return "".join(out)


_SHEET_CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#6b6b6b;--line:#d8d8d8;--panel:#f6f6f6;
      --accent:#1d4ed8;--warn:#b45309;--sel:#e8effd}
*{box-sizing:border-box}
body{font:15px/1.6 system-ui,-apple-system,sans-serif;color:var(--fg);
     background:var(--bg);margin:0}
.wrap{max-width:64rem;margin:0 auto;padding:0 1rem 4rem}
h1{font-size:1.4rem;margin:1.5rem 0 .25rem}
.lede{color:var(--muted);margin:0 0 1rem}
.bar{position:sticky;top:0;z-index:5;background:var(--bg);
     border-bottom:1px solid var(--line);padding:.6rem 0;margin-bottom:1rem}
.bar .row{display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;
          max-width:64rem;margin:0 auto;padding:.15rem 1rem}
.bar label.grp{font-size:.75rem;text-transform:uppercase;letter-spacing:.04em;
               color:var(--muted);margin-right:.1rem}
button.chip{font:inherit;font-size:.85rem;padding:.2rem .6rem;border-radius:999px;
            border:1px solid var(--line);background:var(--bg);cursor:pointer}
button.chip[aria-pressed=true]{background:var(--sel);border-color:var(--accent);
                               color:var(--accent);font-weight:600}
input[type=search]{font:inherit;font-size:.9rem;padding:.25rem .5rem;flex:1;
                   min-width:12rem;border:1px solid var(--line);border-radius:4px}
.count{font-size:.85rem;color:var(--muted);margin-left:auto;white-space:nowrap}
article{border-top:2px solid var(--line);padding-top:1rem;margin-top:2rem}
article.hide{display:none}
h2{font-size:1.05rem;margin:0 0 .4rem;font-family:ui-monospace,monospace}
.tag{display:inline-block;font-size:.72rem;font-family:system-ui;font-weight:600;
     text-transform:uppercase;letter-spacing:.04em;padding:.1rem .45rem;
     border-radius:3px;background:var(--panel);color:var(--muted);
     border:1px solid var(--line);vertical-align:2px;margin-left:.35rem}
.tag.second{background:#fdf3e3;color:var(--warn);border-color:#f0d9ad}
.meta{margin:.2rem 0}
.meta b{font-weight:600}
ol.check{background:#fffdf5;border:1px solid #f0e4c0;border-radius:4px;
         padding:.6rem .6rem .6rem 2rem;margin:.6rem 0}
h3{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;
   color:var(--muted);margin:1.2rem 0 .3rem}
.p{background:var(--panel);border:1px solid var(--line);border-radius:4px;
   padding:.75rem;margin:.5rem 0;overflow-x:auto}
.phead{font-family:ui-monospace,monospace;font-size:.8rem;color:var(--muted);
       border-bottom:1px solid var(--line);padding-bottom:.35rem;margin-bottom:.5rem}
.phead .ab{color:#166534;font-weight:600}
.phead .inj{color:var(--warn);font-weight:600}
.body p{margin:.5rem 0}
.body h1,.body h2,.body h3{font-size:1rem;font-family:inherit;text-transform:none;
                           letter-spacing:0;color:var(--fg);margin:.8rem 0 .3rem}
.body table{border-collapse:collapse;margin:.5rem 0;font-size:.9rem}
.body td,.body th{border:1px solid var(--line);padding:.2rem .45rem;
                  text-align:left;vertical-align:top;background:var(--bg)}
.body th{background:#ededed}
.body ul,.body ol,.body dl{margin:.4rem 0;padding-left:1.4rem}
.body li{margin:.1rem 0}
.body dd{margin-left:1.2rem}
pre.raw{display:none;white-space:pre-wrap;word-break:break-word;margin:0;
        font:12px/1.55 ui-monospace,monospace;color:#444}
body.showraw .body{display:none}
body.showraw pre.raw{display:block}
.empty{display:none;color:var(--muted);padding:2rem 0}
body.noresults .empty{display:block}
@media print{.bar{position:static}article.hide{display:none}}
"""

_SHEET_JS = """
(function(){
 var arts=[].slice.call(document.querySelectorAll('article')),
     q=document.getElementById('q'),
     count=document.getElementById('count'),
     dim='all', who='all';
 function apply(){
   var t=q.value.trim().toLowerCase(), n=0;
   arts.forEach(function(a){
     var ok=(dim==='all'||a.dataset.dim===dim)
          &&(who==='all'||a.dataset[who]==='1')
          &&(!t||a.dataset.find.indexOf(t)>-1);
     a.classList.toggle('hide',!ok); if(ok)n++;
   });
   count.textContent=n+' of '+arts.length+' cases';
   document.body.classList.toggle('noresults',n===0);
 }
 function group(sel,set){
   var btns=[].slice.call(document.querySelectorAll(sel));
   btns.forEach(function(b){b.addEventListener('click',function(){
     btns.forEach(function(o){o.setAttribute('aria-pressed',String(o===b))});
     set(b.dataset.v); apply();
   })});
 }
 group('[data-f=dim]',function(v){dim=v});
 group('[data-f=who]',function(v){who=v});
 q.addEventListener('input',apply);
 document.getElementById('raw').addEventListener('change',function(e){
   document.body.classList.toggle('showraw',e.target.checked);
 });
 apply();
})();
"""


def render_review_sheet(sample: ValidationSample) -> str:
    """Build the self-contained HTML review sheet for a validation sample.

    One file, no assets, no network: it has to survive being emailed to the
    second annotator. The filters (dimension, and whether a case is in the
    inter-rater sub-sample) exist because the two annotators work different
    scopes -- the second annotator reviews only the 50-case sub-sample, and
    filtering to it beats scrolling past 108 cases that are not theirs.
    """
    second_ids = {c.case_id for c in sample.second_annotator}
    by_dim: dict[str, int] = {}
    for c in sample.primary:
        by_dim[c.dimension.value] = by_dim.get(c.dimension.value, 0) + 1

    articles: list[str] = []
    for c in sample.primary:
        is_second = c.case_id in second_ids
        tags = [f'<span class="tag">{html.escape(c.dimension.value)}</span>']
        if c.noise_ratio is not None:
            tags.append(f'<span class="tag">r = {c.noise_ratio:g}</span>')
        if c.dimension is Dimension.REFUSAL:
            tags.append(
                f'<span class="tag">{"answerable control" if c.is_answerable else "unanswerable"}</span>'
            )
        if is_second:
            tags.append('<span class="tag second">2nd annotator</span>')

        checks = "".join(f"<li>{html.escape(q)}</li>" for q in checklist_for(c))
        passages = []
        for p in c.retrieved_passages:
            marks = ""
            if p.is_answer_bearing:
                marks += ' <span class="ab">answer-bearing</span>'
            if p.is_injected:
                marks += ' <span class="inj">injected</span>'
            passages.append(
                f'<div class="p"><div class="phead">{html.escape(p.passage_id)}{marks}</div>'
                f'<div class="body">{render_passage_html(p.text)}</div>'
                f'<pre class="raw">{html.escape(p.text)}</pre></div>'
            )
        # Searchable haystack: the question and the case id, lower-cased once
        # here so the filter never has to walk the (large) passage text.
        find = html.escape(f"{c.case_id} {c.query}".lower(), quote=True)
        articles.append(
            f'<article id="{html.escape(c.case_id, quote=True)}" '
            f'data-dim="{c.dimension.value}" data-second="{"1" if is_second else "0"}" '
            f'data-primaryonly="{"0" if is_second else "1"}" data-find="{find}">'
            f'<h2>{html.escape(c.case_id)}{"".join(tags)}</h2>'
            f'<p class="meta"><b>Question:</b> {html.escape(c.query)}</p>'
            f'<p class="meta"><b>Ground truth:</b> {html.escape(c.answer)}</p>'
            f"<h3>Check</h3><ol class=\"check\">{checks}</ol>"
            f"<h3>Retrieved passages</h3>{''.join(passages)}</article>"
        )

    def chip(f: str, value: str, label: str, pressed: bool = False) -> str:
        return (f'<button class="chip" type="button" data-f="{f}" data-v="{value}" '
                f'aria-pressed="{str(pressed).lower()}">{html.escape(label)}</button>')

    dim_chips = chip("dim", "all", f"all {len(sample.primary)}", True) + "".join(
        chip("dim", d, f"{d} {by_dim[d]}") for d in sorted(by_dim)
    )
    who_chips = (
        chip("who", "all", "both annotators", True)
        + chip("who", "second", f"2nd annotator {len(sample.second_annotator)}")
        + chip("who", "primaryonly", f"1st annotator only {len(sample.primary) - len(second_ids)}")
    )

    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Validation review sheet</title>"
        f"<style>{_SHEET_CSS}</style></head><body>"
        '<div class="wrap"><h1>Manual validation review sheet</h1>'
        f'<p class="lede">P1 Section 5.5.3. {len(sample.primary)} sampled cases; the '
        f"{len(sample.second_annotator)} marked <b>2nd annotator</b> are the inter-rater "
        "sub-sample. Answer every check for each case in your label CSV. Wikipedia markup "
        "is rendered for readability only &mdash; tick <b>raw markup</b> to see the exact "
        "text the pipelines receive.</p></div>"
        '<div class="bar"><div class="row"><label class="grp">testbed</label>'
        f'{dim_chips}</div><div class="row"><label class="grp">reviewer</label>{who_chips}</div>'
        '<div class="row"><input type="search" id="q" placeholder="filter by question or case id">'
        '<label><input type="checkbox" id="raw"> raw markup</label>'
        '<span class="count" id="count"></span></div></div>'
        f'<div class="wrap"><p class="empty">No cases match these filters.</p>'
        f"{''.join(articles)}</div>"
        f"<script>{_SHEET_JS}</script></body></html>"
    )


# --------------------------------------------------------------------------
# Inter-rater agreement over the label sheets (P1 Section 5.5.3)
# --------------------------------------------------------------------------


@dataclass
class AgreementResult:
    """Agreement between two label sheets over the rows they share.

    `rater_kind` records WHAT was compared, because the arithmetic is identical
    for two people and for one person labelling twice while the claim is not.
    P1 5.5.3 asks for "a sub-sample of at least fifty cases reviewed by a second
    annotator", which is inter-rater agreement; the same person labelling two
    sheets measures intra-rater (test-retest) reliability instead. Reporting the
    latter as the former would overstate the result, since one rater carries the
    same reading of every borderline case into both passes.
    """

    kappa: float
    n_items: int
    n_cases: int
    observed_agreement: float
    rater_kind: str  # "inter" or "intra"
    by_dimension: dict[str, float]
    skipped_blank: int
    meets_p1_gate: bool
    # Cohen's kappa is the P1 gate, but it is not self-interpreting on a
    # testbed where almost every item passes review. These are the statistics
    # that say WHY it landed where it did; see metrics.conflict.
    diagnostics: "AgreementDiagnostics | None" = None

    def as_dict(self) -> dict[str, object]:
        return {
            "cohens_kappa": round(self.kappa, 4),
            "rater_kind": self.rater_kind,
            "interpretation": (
                "inter-rater agreement (P1 5.5.3)" if self.rater_kind == "inter"
                else "intra-rater / test-retest reliability -- NOT the P1 5.5.3 "
                     "inter-rater gate, which requires a second annotator"
            ),
            "n_items_compared": self.n_items,
            "n_cases_compared": self.n_cases,
            "observed_agreement": round(self.observed_agreement, 4),
            "kappa_by_dimension": {k: round(v, 4) for k, v in sorted(self.by_dimension.items())},
            "blank_rows_skipped": self.skipped_blank,
            "meets_p1_gate": self.meets_p1_gate,
            "p1_threshold": 0.60,
            # The gate stays keyed on Cohen's kappa. Nothing below can move it:
            # a threshold that is re-pointed at whichever statistic clears it is
            # not a threshold. These are reported ALONGSIDE the gate, and the
            # divergence between them is itself the finding.
            "paradox_diagnostics": (
                None if self.diagnostics is None else self.diagnostics.as_dict()
            ),
        }


def _read_labels(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    """Map (case_id, question_no) -> (label, dimension) for non-blank rows."""
    import csv  # noqa: PLC0415

    out: dict[tuple[str, str], tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            label = (row.get("answer_yes_no") or "").strip().lower()
            if label not in ("yes", "no"):
                continue  # blank means "not reviewed", never "failed"
            out[(row["case_id"], row["question_no"])] = (label, row.get("dimension", "?"))
    return out


def compute_label_agreement(
    path_a: Path,
    path_b: Path,
    *,
    rater_kind: str,
    threshold: float = 0.60,
) -> AgreementResult:
    """Cohen's kappa between two filled label sheets, over shared rows only.

    Aligned on (case_id, question_no) rather than row order: the two sheets are
    drawn from different sample sizes, so position carries no meaning and a
    positional zip would silently compare unrelated questions.

    Rows blank in either sheet are excluded. A blank is "not yet reviewed", and
    scoring it as agreement would inflate kappa with work nobody did.

    `rater_kind` must be stated by the caller because nothing in the files
    records who filled them; see AgreementResult.
    """
    from ..metrics.conflict import agreement_diagnostics, cohens_kappa  # noqa: PLC0415

    if rater_kind not in ("inter", "intra"):
        raise ValueError("rater_kind must be 'inter' (two annotators) or 'intra' (one)")

    a, b = _read_labels(path_a), _read_labels(path_b)
    shared = sorted(set(a) & set(b))
    blank = len(set(a) ^ set(b))
    if not shared:
        raise ValueError(
            "the two sheets share no completed rows; kappa needs both raters on "
            "the same items"
        )

    codes = {"no": 0, "yes": 1}
    ra = [codes[a[k][0]] for k in shared]
    rb = [codes[b[k][0]] for k in shared]
    kappa = cohens_kappa(ra, rb)

    by_dim: dict[str, float] = {}
    dims = {a[k][1] for k in shared}
    for d in dims:
        keys = [k for k in shared if a[k][1] == d]
        if len(keys) < 2:
            continue
        try:
            by_dim[d] = cohens_kappa([codes[a[k][0]] for k in keys],
                                     [codes[b[k][0]] for k in keys])
        except ValueError:
            continue

    observed = sum(1 for x, y in zip(ra, rb) if x == y) / len(ra)
    return AgreementResult(
        diagnostics=agreement_diagnostics(ra, rb),
        kappa=kappa,
        n_items=len(shared),
        n_cases=len({k[0] for k in shared}),
        observed_agreement=observed,
        rater_kind=rater_kind,
        by_dimension=by_dim,
        skipped_blank=blank,
        meets_p1_gate=(rater_kind == "inter" and kappa >= threshold),
    )
