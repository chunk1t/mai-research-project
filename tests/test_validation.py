"""Manual validation sampling (P1 5.5.3).

Expected sample sizes are computed by hand from the real benchmark shape --
200 refusal, 200 conflict, 2,500 noise instances of which 500 are r=0 -- and
asserted against the implementation, not read back from it.
"""

from __future__ import annotations

import re
from html import unescape as html_unescape

import pytest

from ragrobust.dataset.validation import (
    ANSWERABLE_CONTROL_CHECK,
    CHECKLISTS,
    ValidationSample,
    checklist_for,
    read_review_failures,
    read_reviewed_case_ids,
    render_passage_html,
    render_review_sheet,
    stale_review_case_ids,
    stratified_validation_sample,
)
from ragrobust.schema import Dimension, NO_ANSWER, Passage, PerturbationType, SeedSource, TestCase


def mk(dim: Dimension, i: int, *, answerable: bool = True, ratio: float | None = None,
       seed: int | None = None) -> TestCase:
    if dim is Dimension.REFUSAL:
        pert = (PerturbationType.ANSWER_PASSAGE_RETAINED if answerable
                else PerturbationType.ANSWER_PASSAGE_REMOVED)
    elif dim is Dimension.CONFLICT:
        pert = PerturbationType.CONTRADICTION_INJECTED
    else:
        pert = PerturbationType.DISTRACTORS_ADDED
    passages = [Passage(passage_id=f"{dim.value}-{i}-p0", text="evidence text",
                        is_answer_bearing=answerable)]
    seed_no = i if seed is None else seed
    return TestCase(
        case_id=f"{dim.value}-{i}",
        query=f"question {i}",
        retrieved_passages=passages,
        answer="1889" if answerable else NO_ANSWER,
        dimension=dim,
        perturbation_type=pert,
        noise_ratio=ratio,
        seed_source=SeedSource.NATURAL_QUESTIONS,
        seed_id=f"{dim.value}-seed-{seed_no}",
    )


def real_shape() -> list[TestCase]:
    """The benchmark as actually built: 200 / 200 / 2,500.

    Crucially, a noise seed contributes FIVE instances that all share one
    seed_id -- that is what makes it one base case, and it is what the sampler
    groups on.
    """
    cases = [mk(Dimension.REFUSAL, i, answerable=i % 2 == 0) for i in range(200)]
    cases += [mk(Dimension.CONFLICT, i) for i in range(200)]
    for r in (0.0, 0.25, 0.5, 0.75, 0.9):
        cases += [mk(Dimension.NOISE, int(r * 100) * 1000 + i, ratio=r, seed=i)
                  for i in range(500)]
    return cases


def test_primary_sample_sizes_are_proportional():
    """Hand-computed against P1's 15-20 percent of BASE cases, proportionally.

    P1 frames the benchmark as ~900 base test cases: 200 refusal, 200 conflict,
    500 noise (each noise seed expanding to 5 ratios).
      refusal  : 0.175 x 200 =  35.0 -> 35
      conflict : 0.175 x 200 =  35.0 -> 35
      noise    : 0.175 x 500 =  87.5 -> 88   (half-up, not banker's)
      total    = 158, i.e. 158/900 = 17.6 percent, inside P1's 15-20 band.
    """
    s = stratified_validation_sample(real_shape(), fraction=0.175, second_annotator_n=50)
    assert s.stats["base_cases_total"] == 900
    assert s.stats["primary_by_dimension"] == {"refusal": 35, "conflict": 35, "noise": 88}
    assert s.stats["primary_n"] == 158
    assert 0.15 <= s.stats["fraction_achieved"] <= 0.20


def test_each_noise_seed_appears_at_most_once():
    """A noise seed is ONE base case; drawing it five times is repeated work."""
    s = stratified_validation_sample(real_shape(), fraction=0.175, second_annotator_n=50)
    noise_seeds = [c.seed_id for c in s.primary if c.dimension is Dimension.NOISE]
    assert len(noise_seeds) == len(set(noise_seeds))


def test_noise_representatives_span_the_ratio_range():
    """The reviewer should see several ratios, not 88 copies of the same one."""
    s = stratified_validation_sample(real_shape(), fraction=0.175, second_annotator_n=50)
    ratios = {c.noise_ratio for c in s.primary if c.dimension is Dimension.NOISE}
    assert ratios == {0.25, 0.5, 0.75, 0.9}


def test_second_annotator_subsample_is_drawn_from_the_primary():
    """Kappa needs two labels on the SAME cases, so the sub-sample must overlap.

    Hand-computed from a 158-case primary sample at n=50:
      refusal  : 50 x 35/158 = 11.08 -> 11
      conflict : 50 x 35/158 = 11.08 -> 11
      noise    : 50 x 88/158 = 27.85 -> 28
      total    = 50, and P1 requires "at least fifty".
    """
    s = stratified_validation_sample(real_shape(), fraction=0.175, second_annotator_n=50)
    assert s.stats["second_annotator_by_dimension"] == {"refusal": 11, "conflict": 11, "noise": 28}
    assert len(s.second_annotator) == 50

    primary_ids = {c.case_id for c in s.primary}
    assert all(c.case_id in primary_ids for c in s.second_annotator), \
        "second annotator got cases the primary reviewer never sees -- kappa uncomputable"


def test_zero_noise_instances_are_excluded():
    """An r=0 case has no distractors, so P1's noise question is vacuous there."""
    s = stratified_validation_sample(real_shape(), fraction=0.175, second_annotator_n=50)
    assert all((c.noise_ratio or 0.0) > 0.0
               for c in s.primary if c.dimension is Dimension.NOISE)
    # ... unless the caller asks otherwise.
    s2 = stratified_validation_sample(real_shape(), fraction=0.175, second_annotator_n=50,
                                      exclude_zero_noise=False)
    assert 0.0 in {c.noise_ratio for c in s2.primary if c.dimension is Dimension.NOISE}


def test_sample_is_reproducible_and_seed_sensitive():
    a = stratified_validation_sample(real_shape(), rng_seed=7)
    b = stratified_validation_sample(real_shape(), rng_seed=7)
    c = stratified_validation_sample(real_shape(), rng_seed=8)
    assert [x.case_id for x in a.primary] == [x.case_id for x in b.primary]
    assert [x.case_id for x in a.primary] != [x.case_id for x in c.primary]


def test_rounding_is_half_up_not_bankers():
    """round(87.5) is 88 but round(86.5) is 86; sample size must not hinge on parity.

    50 conflict cases at fraction 0.25 gives exactly 12.5, which banker's
    rounding sends to 12 and half-up sends to 13.
    """
    cases = [mk(Dimension.CONFLICT, i) for i in range(50)]
    s = stratified_validation_sample(cases, fraction=0.25, second_annotator_n=5)
    assert s.stats["primary_by_dimension"]["conflict"] == 13


def test_second_annotator_cannot_exceed_the_primary_sample():
    cases = [mk(Dimension.CONFLICT, i) for i in range(50)]
    with pytest.raises(ValueError, match="exceeds the primary sample"):
        stratified_validation_sample(cases, fraction=0.10, second_annotator_n=40)


def test_answerable_control_gets_the_mirror_checklist():
    """A control must be checked for the OPPOSITE property to an unanswerable case."""
    control = mk(Dimension.REFUSAL, 1, answerable=True)
    unanswerable = mk(Dimension.REFUSAL, 2, answerable=False)
    assert checklist_for(control) == ANSWERABLE_CONTROL_CHECK
    assert checklist_for(unanswerable) == CHECKLISTS[Dimension.REFUSAL]
    assert checklist_for(mk(Dimension.CONFLICT, 3)) == CHECKLISTS[Dimension.CONFLICT]


# --------------------------------------------------------------------------
# Review sheet rendering. Expected HTML is written out by hand from the input
# markup, character for character, then asserted -- not read back from the
# renderer.
# --------------------------------------------------------------------------


def test_wikipedia_table_becomes_a_real_table():
    src = ('<Table> <Tr> <Th colspan="2"> Swine flu </Th> </Tr> '
           "<Tr> <Th> Specialty </Th> <Td> Infectious disease </Td> </Tr> </Table>")
    assert render_passage_html(src) == (
        '<table> <tr> <th colspan="2"> Swine flu </th> </tr> '
        "<tr> <th> Specialty </th> <td> Infectious disease </td> </tr> </table>"
    )


def test_prose_markup_is_rendered_and_case_folded():
    assert render_passage_html("<P> Beijing , China </P> <H2> Other notes </H2>") == (
        "<p> Beijing , China </p> <h2> Other notes </h2>"
    )


def test_passage_starting_mid_element_drops_the_stray_end_tag():
    # A window centred on the answer routinely opens with "</P>". Emitting that
    # unmatched close tag would end an element the sheet itself opened.
    # The tag goes; both spaces that surrounded it are text and stay.
    assert render_passage_html("Beijing . </P> <H2> Notes </H2>") == (
        "Beijing .  <h2> Notes </h2>"
    )


def test_passage_ending_mid_element_is_closed():
    assert render_passage_html("<P> unterminated") == "<p> unterminated</p>"


def test_orphan_cell_gets_the_ancestors_a_browser_would_otherwise_drop():
    # <td> outside a table is discarded by the HTML parser, taking its text
    # with it -- the one failure mode that would hide evidence from the
    # annotator rather than merely look untidy.
    assert render_passage_html("<Td> 1932 </Td>") == "<table><tr><td> 1932 </td></tr></table>"


def test_text_directly_inside_a_table_keeps_its_position():
    # Foster parenting would hoist "The Hunger Games" above the table.
    # The stray text gets a full implied row+cell (a bare <td> would itself be
    # an orphan), which the real <Tr> that follows then closes.
    assert render_passage_html("<Table> The Hunger Games <Tr> <Td> 2012 </Td> </Tr> </Table>") == (
        "<table><tr><td> The Hunger Games </td></tr>"
        "<tr> <td> 2012 </td> </tr> </table>"
    )


def test_unknown_tags_and_prose_angle_brackets_are_escaped():
    assert render_passage_html("5 < 6 and <Script> x </Script> & <B> b </B>") == (
        "5 &lt; 6 and &lt;Script&gt; x &lt;/Script&gt; &amp; &lt;B&gt; b &lt;/B&gt;"
    )


def test_only_span_attributes_survive():
    assert render_passage_html('<Td style="x" colspan="3" onclick="evil()"> a </Td>') == (
        '<table><tr><td colspan="3"> a </td></tr></table>'
    )


def test_rendering_preserves_every_character_of_passage_text():
    src = ('Swine influenza <Table> <Tr> <Th colspan="2"> Swine flu </Th> </Tr> </Table> '
           "<P> caused by 5 < 6 viruses </P>")
    rendered = render_passage_html(src)
    visible = html_unescape(re.sub(r"<[^>]*>", "", rendered))
    assert visible == re.sub(r"<(/?)(Table|Tr|Th|P)(\s[^>]*)?>", "", src)


def test_review_sheet_filters_carry_dimension_and_annotator_scope():
    cases = [mk(Dimension.REFUSAL, i) for i in range(4)]
    cases += [mk(Dimension.CONFLICT, i) for i in range(2)]
    sample = ValidationSample(primary=cases, second_annotator=cases[:2])
    sheet = render_review_sheet(sample)

    # 6 primary cases: 4 refusal + 2 conflict; 2 of them also second annotator.
    assert sheet.count("<article ") == 6
    assert sheet.count('data-dim="refusal"') == 4
    assert sheet.count('data-dim="conflict"') == 2
    assert sheet.count('data-second="1"') == 2
    assert sheet.count('data-primaryonly="1"') == 4
    # Chip labels carry the hand-counted totals.
    assert 'data-v="all" aria-pressed="true">all 6<' in sheet
    assert 'data-v="refusal" aria-pressed="false">refusal 4<' in sheet
    assert 'data-v="second" aria-pressed="false">2nd annotator 2<' in sheet
    assert 'data-v="primaryonly" aria-pressed="false">1st annotator only 4<' in sheet


def test_review_sheet_search_index_is_lowercased_query_and_id():
    case = mk(Dimension.CONFLICT, 7)
    sheet = render_review_sheet(ValidationSample(primary=[case], second_annotator=[]))
    assert 'data-find="conflict-7 question 7"' in sheet


def test_review_sheet_shows_both_rendered_and_raw_passage_text():
    case = mk(Dimension.CONFLICT, 0)
    case.retrieved_passages[0].text = "<P> a & b </P>"
    sheet = render_review_sheet(ValidationSample(primary=[case], second_annotator=[]))
    assert '<div class="body"><p> a &amp; b </p></div>' in sheet
    assert '<pre class="raw">&lt;P&gt; a &amp; b &lt;/P&gt;</pre>' in sheet


# --------------------------------------------------------------------------
# Reading the human verdict back (P1 5.5.3 repair-or-discard)
# --------------------------------------------------------------------------

_HEADER = "case_id,dimension,question_no,question,answer_yes_no,notes\n"


def _write_labels(tmp_path, name: str, body: str):
    path = tmp_path / name
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


def test_only_no_answers_are_failures(tmp_path):
    # Hand-checked: of these four rows only c3's counts. A blank means "not yet
    # reviewed" -- treating it as a failure would discard the whole benchmark
    # on the first run, before anyone has annotated anything.
    path = _write_labels(tmp_path, "primary_labels.csv", (
        "c1,conflict,1,Do they contradict?,yes,\n"
        "c2,conflict,1,Do they contradict?,,\n"
        "c3,conflict,1,Do they contradict?,no,passages agree\n"
        "c4,noise,1,Are distractors related?,YES,\n"
    ))
    failures = read_review_failures([path])
    assert set(failures) == {"c3"}
    assert failures["c3"] == ["Do they contradict? [passages agree]"]


def test_any_failed_question_fails_the_case():
    # The checklists are conjunctive: each question tests a different way the
    # perturbation can be wrong, so one "no" is enough.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "primary_labels.csv"
        path.write_text(_HEADER + (
            "c1,conflict,1,Do they contradict?,yes,\n"
            "c1,conflict,2,Is it plausible?,no,absurd substitution\n"
            "c1,conflict,3,Same topic?,yes,\n"
        ), encoding="utf-8")
        failures = read_review_failures([path])
    assert list(failures) == ["c1"]
    assert failures["c1"] == ["Is it plausible? [absurd substitution]"]


def test_reasons_accumulate_across_annotators_without_duplicating(tmp_path):
    # Two annotators failing the same case for DIFFERENT reasons is a stronger
    # finding, so both are kept; the identical reason is not repeated.
    a = _write_labels(tmp_path, "primary_labels.csv",
                      "c1,noise,2,Answer absent?,no,distractor 4 states it\n")
    b = _write_labels(tmp_path, "second_annotator_labels.csv", (
        "c1,noise,2,Answer absent?,no,distractor 4 states it\n"
        "c1,noise,1,Topically related?,no,unrelated subject\n"
    ))
    failures = read_review_failures([a, b])
    assert failures["c1"] == [
        "Answer absent? [distractor 4 states it]",
        "Topically related? [unrelated subject]",
    ]


def test_a_missing_ledger_is_not_an_error(tmp_path):
    # The ledger only exists once something has failed review.
    assert read_review_failures([tmp_path / "review_failures.csv"]) == {}


def test_second_annotator_subsample_meets_the_minimum_after_pruning():
    """P1 5.5.3 requires "at least fifty"; proportional rounding can miss it.

    The pruned benchmark has 183 refusal, 200 conflict and 500 noise base cases,
    so the primary sample is hand-computed as:
      refusal  : 0.175 x 183 = 32.025 -> 32
      conflict : 0.175 x 200 = 35.0   -> 35
      noise    : 0.175 x 500 = 87.5   -> 88
      total    = 155
    At n=50 the exact shares are 10.32, 11.29 and 28.39. Rounding each one down
    gives 10 + 11 + 28 = 49, one short. The largest remainder is noise (.39), so
    the top-up lands there: 10 + 11 + 29 = 50.
    """
    cases = [mk(Dimension.REFUSAL, i, answerable=i % 2 == 0) for i in range(183)]
    cases += [mk(Dimension.CONFLICT, i) for i in range(200)]
    for r in (0.0, 0.25, 0.5, 0.75, 0.9):
        cases += [mk(Dimension.NOISE, int(r * 100) * 1000 + i, ratio=r, seed=i)
                  for i in range(500)]

    s = stratified_validation_sample(cases, fraction=0.175, second_annotator_n=50)
    assert s.stats["primary_by_dimension"] == {"refusal": 32, "conflict": 35, "noise": 88}
    assert s.stats["second_annotator_by_dimension"] == {
        "refusal": 10, "conflict": 11, "noise": 29,
    }
    assert len(s.second_annotator) == 50, "P1 5.5.3 requires at least fifty"

    primary_ids = {c.case_id for c in s.primary}
    assert all(c.case_id in primary_ids for c in s.second_annotator)


def test_stale_labels_are_the_reviewed_cases_the_benchmark_no_longer_has(tmp_path):
    """A verdict outlives the case it judged; that must be visible, not silent.

    Hand-computed: c1 and c3 carry verdicts, c2 is blank (not yet reviewed).
    The benchmark holds c1 and c4. So the only stale verdict is c3 -- c2 is not
    stale because it was never judged, and c4 is simply unreviewed.
    """
    path = tmp_path / "primary_labels.csv"
    path.write_text(_HEADER + (
        "c1,conflict,1,Do they contradict?,yes,\n"
        "c2,conflict,1,Do they contradict?,,\n"
        "c3,noise,1,Are distractors related?,no,off topic\n"
    ), encoding="utf-8")

    assert read_reviewed_case_ids([path]) == {"c1", "c3"}
    assert stale_review_case_ids([path], {"c1", "c4"}) == ["c3"]
    # Nothing stale when the benchmark still holds every reviewed case.
    assert stale_review_case_ids([path], {"c1", "c2", "c3"}) == []


def test_stale_check_tolerates_a_missing_sheet(tmp_path):
    assert stale_review_case_ids([tmp_path / "nope.csv"], {"c1"}) == []


def test_deliberately_discarded_cases_are_not_stale(tmp_path):
    """The discard ledger names cases the benchmark no longer has, by design.

    Counting those as stale would fire the warning loudest precisely when the
    repair-or-discard loop worked, and a warning that always fires is ignored.
    """
    ledger = tmp_path / "review_failures.csv"
    ledger.write_text(_HEADER + "c9,conflict,2,Is it plausible?,no,absurd\n", encoding="utf-8")
    sheet = tmp_path / "primary_labels.csv"
    sheet.write_text(_HEADER + "c7,noise,1,Are distractors related?,yes,\n", encoding="utf-8")

    benchmark = {"c1"}  # holds neither c7 nor c9
    # Without the exemption both look stale.
    assert stale_review_case_ids([sheet, ledger], benchmark) == ["c7", "c9"]
    # With it, only c7 -- an absence nobody asked for.
    assert stale_review_case_ids(
        [sheet, ledger], benchmark, discarded_case_ids={"c9"}
    ) == ["c7"]


# --------------------------------------------------------------------------
# Label-sheet agreement (P1 5.5.3)
# --------------------------------------------------------------------------


def _sheet(tmp_path, name, rows):
    import csv
    p = tmp_path / name
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "dimension", "question_no", "question", "answer_yes_no", "notes"])
        for cid, dim, q, lab in rows:
            w.writerow([cid, dim, q, "q?", lab, ""])
    return p


def test_kappa_matches_a_hand_computed_value(tmp_path):
    """Ten shared items, hand-computed.

      rater A: 5 yes, 5 no      rater B: 5 yes, 5 no
      agreements: 4 yes/yes + 4 no/no = 8, so observed = 0.80
      expected   = (5/10 * 5/10) + (5/10 * 5/10) = 0.50
      kappa      = (0.80 - 0.50) / (1 - 0.50) = 0.600
    """
    from ragrobust.dataset.validation import compute_label_agreement

    a_lab = ["yes"] * 5 + ["no"] * 5
    b_lab = ["yes"] * 4 + ["no", "yes"] + ["no"] * 4
    rows_a = [(f"c{i}", "noise", "1", a_lab[i]) for i in range(10)]
    rows_b = [(f"c{i}", "noise", "1", b_lab[i]) for i in range(10)]
    r = compute_label_agreement(_sheet(tmp_path, "a.csv", rows_a),
                                _sheet(tmp_path, "b.csv", rows_b), rater_kind="inter")
    assert r.kappa == pytest.approx(0.600, abs=1e-6)
    assert r.observed_agreement == pytest.approx(0.80, abs=1e-6)
    assert r.n_items == 10 and r.n_cases == 10
    assert r.meets_p1_gate is True  # 0.600 >= 0.60


def test_blank_rows_are_excluded_not_counted_as_agreement(tmp_path):
    """A blank means 'not reviewed'. Counting it as agreement inflates kappa
    with work nobody did."""
    from ragrobust.dataset.validation import compute_label_agreement

    rows_a = [("c1", "noise", "1", "yes"), ("c2", "noise", "1", "no"), ("c3", "noise", "1", "yes")]
    rows_b = [("c1", "noise", "1", "yes"), ("c2", "noise", "1", "no"), ("c3", "noise", "1", "")]
    r = compute_label_agreement(_sheet(tmp_path, "a.csv", rows_a),
                                _sheet(tmp_path, "b.csv", rows_b), rater_kind="inter")
    assert r.n_items == 2, "the blank row must not be compared"


def test_alignment_is_by_case_and_question_not_row_order(tmp_path):
    """The sheets have different sizes, so position carries no meaning."""
    from ragrobust.dataset.validation import compute_label_agreement

    rows_a = [("c1", "noise", "1", "yes"), ("c2", "noise", "1", "no"),
              ("c3", "noise", "1", "yes"), ("c4", "noise", "1", "no")]
    rows_b = [("c3", "noise", "1", "yes"), ("c1", "noise", "1", "yes")]
    r = compute_label_agreement(_sheet(tmp_path, "a.csv", rows_a),
                                _sheet(tmp_path, "b.csv", rows_b), rater_kind="inter")
    assert r.n_items == 2
    assert r.observed_agreement == pytest.approx(1.0)  # c1 and c3 both agree


def test_intra_rater_never_satisfies_the_p1_gate(tmp_path):
    """P1 5.5.3 requires a SECOND annotator. One person labelling twice measures
    self-consistency, and the arithmetic cannot tell the difference -- so the
    distinction has to be carried explicitly."""
    from ragrobust.dataset.validation import compute_label_agreement

    rows = [(f"c{i}", "noise", "1", "yes" if i % 2 else "no") for i in range(10)]
    r = compute_label_agreement(_sheet(tmp_path, "a.csv", rows),
                                _sheet(tmp_path, "b.csv", rows), rater_kind="intra")
    assert r.kappa == pytest.approx(1.0)
    assert r.meets_p1_gate is False, "perfect self-agreement is still not the P1 gate"
    assert "NOT the P1 5.5.3" in r.as_dict()["interpretation"]


def test_no_shared_rows_raises(tmp_path):
    from ragrobust.dataset.validation import compute_label_agreement

    a = _sheet(tmp_path, "a.csv", [("c1", "noise", "1", "yes")])
    b = _sheet(tmp_path, "b.csv", [("c9", "noise", "1", "yes")])
    with pytest.raises(ValueError, match="share no completed rows"):
        compute_label_agreement(a, b, rater_kind="inter")
