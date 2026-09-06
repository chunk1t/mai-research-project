#!/usr/bin/env python3
"""Build the real benchmark from Natural Questions and TriviaQA (P1 Section 5.5).

Staged, because the stages have very different prerequisites and very different
costs. Seeds and embeddings are slow but local. Refusal and noise need nothing
but those. Conflict needs the Gemini case generator and an NLI model, so it is
isolated: everything else can be built while an API key is still outstanding.

Every stage caches to `data/`, so a failure part way through never costs the
work already done.

    python scripts/build_dataset.py --stage seeds      # download, filter, cache
    python scripts/build_dataset.py --stage embed      # bge embeddings, cache
    python scripts/build_dataset.py --stage refusal
    python scripts/build_dataset.py --stage noise
    python scripts/build_dataset.py --stage conflict --limit 8   # pilot first
    python scripts/build_dataset.py --stage conflict   # needs GOOGLE_API_KEY
    python scripts/build_dataset.py --stage verify     # re-derive the invariants
    python scripts/build_dataset.py --stage assemble   # merge + manifest
    python scripts/build_dataset.py --stage validation # export the P1 5.5.3 sample

Seed reuse across dimensions is deliberate and follows P1's own arithmetic:
5.5.1 samples "approximately three hundred seed examples [...] from each
dataset" (600) while 5.6.5 specifies "approximately nine hundred base test
cases", so the testbeds must draw from a shared pool. Within the refusal
testbed the unanswerable and control partitions stay disjoint, which is the
separation Refusal F1 actually depends on. The overlap between dimensions is
counted and written into the manifest rather than left implicit.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragrobust.dataset.conflict import (  # noqa: E402
    CONTRADICTION_PROMPT,
    ConflictCandidate,
    build_conflict_case,
    summarise_candidates,
    verify_contradiction,
)
from ragrobust.dataset.noise import build_noise_testbed  # noqa: E402
from ragrobust.dataset.refusal import build_refusal_testbed  # noqa: E402
from ragrobust.dataset.seeds import (  # noqa: E402
    Seed,
    filter_seeds,
    load_natural_questions,
    load_trivia_qa,
    sample_seeds,
    strip_markup_for_embedding,
    verify_seed_integrity,
)
from ragrobust.cache import ResponseCache  # noqa: E402
from ragrobust.providers.base import GenerationRequest  # noqa: E402
from ragrobust.providers.cached import CachedProvider  # noqa: E402
from ragrobust.providers.factory import build_provider, load_config  # noqa: E402
from ragrobust.corpus import SandboxedCorpus  # noqa: E402
from ragrobust.matching import contains_on_word_boundary  # noqa: E402
from ragrobust.schema import (  # noqa: E402
    NO_ANSWER,
    Benchmark,
    Passage,
    SeedSource,
    TestCase,
)

DATA = ROOT / "data"
SEEDS_PATH = DATA / "seeds" / "seeds.json"
EMB_PATH = DATA / "seeds" / "embeddings.npz"
CASES = DATA / "cases"
CACHE_PATH = DATA / "cache" / "case_generator.sqlite"

# Generation budget for one contradiction rewrite. Evidence is capped at 800
# tokens (configs/dataset.yaml) and the prompt asks for the SAME length back, so
# the answer alone can approach that. Gemini 3.x thinks by default and its
# thoughts are drawn from the same budget, so a budget sized to the answer alone
# silently returns a truncated passage -- which NLI then scores as a weak
# contradiction and the stage rejects, burning free-tier quota to produce a
# testbed biased toward short passages.
#
# 8192 rather than 4096: the 12-seed pilot still lost one candidate to
# truncation at 4096 (thoughts ran ~1,300-2,000 tokens on top of an ~850-token
# rewrite). Output tokens are not separately billed on the free tier, so the
# headroom is free and each recovered candidate is one less seed consumed.
CONTRADICTION_MAX_TOKENS = 8192


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_seeds(seeds: list[Seed], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "seed_id": s.seed_id,
            "source": s.source.value,
            "query": s.query,
            "answer": s.answer,
            "passages": [p.model_dump() for p in s.passages],
            "extra_passages": [p.model_dump() for p in s.extra_passages],
        }
        for s in seeds
    ]
    path.write_text(json.dumps(payload))


def load_seeds(path: Path) -> list[Seed]:
    rows = json.loads(path.read_text())
    return [
        Seed(
            seed_id=r["seed_id"],
            source=SeedSource(r["source"]),
            query=r["query"],
            answer=r["answer"],
            passages=[Passage(**p) for p in r["passages"]],
            # Absent in seed files written before the distractor corpus was
            # enlarged; an old cache stays loadable, just without neighbours.
            extra_passages=[Passage(**p) for p in r.get("extra_passages", [])],
        )
        for r in rows
    ]


def save_cases(cases: list[TestCase], name: str) -> None:
    CASES.mkdir(parents=True, exist_ok=True)
    path = CASES / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(c.model_dump_json() + "\n")
    print(f"  wrote {len(cases)} cases -> {path}")


def load_cases(name: str) -> list[TestCase]:
    path = CASES / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [TestCase.model_validate_json(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def stage_seeds(cfg: dict) -> None:
    seeds_cfg = cfg["seeds"]
    filters = seeds_cfg["filters"]
    print("loading Natural Questions...")
    nq, nq_stats = load_natural_questions(limit=6000)
    print(f"  {dict(nq_stats)}")
    print("loading TriviaQA...")
    tqa, tqa_stats = load_trivia_qa(limit=6000)
    print(f"  {dict(tqa_stats)}")

    rng_seed = cfg.get("rng_seed", 20260721)
    nq = sample_seeds(nq, seeds_cfg["natural_questions"], rng_seed)
    tqa = sample_seeds(tqa, seeds_cfg["trivia_qa"], rng_seed + 1)

    combined = nq + tqa
    kept, fstats = filter_seeds(
        combined,
        drop_ambiguous_answers=filters["drop_ambiguous_answers"],
        drop_multi_answer=filters["drop_multi_answer"],
        min_evidence_tokens=filters["min_evidence_tokens"],
        max_evidence_tokens=filters["max_evidence_tokens"],
    )
    print(f"filter: {dict(fstats)}")

    # Integrity gate. A mislabelled answer flag corrupts every perturbation
    # downstream and the schema cannot detect it, so it fails here or nowhere.
    problems: list[str] = []
    for s in kept:
        problems.extend(verify_seed_integrity(s))
    if problems:
        print(f"\nINTEGRITY FAILURES: {len(problems)}")
        for p in problems[:10]:
            print(f"  {p}")
        raise SystemExit("refusing to build on seeds whose answer flags do not match their text")

    print(f"\n{len(kept)} seeds pass integrity ({sum(1 for s in kept if s.source is SeedSource.NATURAL_QUESTIONS)} NQ, "
          f"{sum(1 for s in kept if s.source is SeedSource.TRIVIA_QA)} TriviaQA)")
    save_seeds(kept, SEEDS_PATH)
    print(f"cached -> {SEEDS_PATH}")


def stage_embed(cfg: dict) -> None:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    seeds = load_seeds(SEEDS_PATH)
    model_name = cfg["embedding_model"]
    n_signal = sum(len(s.passages) for s in seeds)
    n_extra = sum(len(s.extra_passages) for s in seeds)
    print(f"embedding {n_signal + n_extra} passages ({n_signal} evidence + "
          f"{n_extra} distractor candidates) with {model_name}...")
    model = SentenceTransformer(model_name)

    ids: list[str] = []
    texts: list[str] = []
    for s in seeds:
        # Extras are distractor candidates only, but they must be embedded too:
        # the topical filter compares every candidate against the seed anchor.
        for p in [*s.passages, *s.extra_passages]:
            ids.append(p.passage_id)
            # Similarity must reflect topic, not Wikipedia table markup. The
            # stored passage text is untouched; only the vector changes.
            texts.append(strip_markup_for_embedding(p.text))

    vectors = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB_PATH, ids=np.array(ids), vectors=vectors)
    print(f"cached {len(ids)} embeddings -> {EMB_PATH}")


def load_embeddings() -> dict[str, np.ndarray]:
    blob = np.load(EMB_PATH, allow_pickle=True)
    return dict(zip(blob["ids"].tolist(), blob["vectors"]))


def _partition(seeds: list[Seed], n: int, rng_seed: int) -> list[Seed]:
    pool = list(seeds)
    random.Random(rng_seed).shuffle(pool)
    return pool[:n]


def stage_refusal(cfg: dict) -> None:
    seeds = load_seeds(SEEDS_PATH)
    emb = load_embeddings()
    tb = cfg["testbeds"]["refusal"]
    total = tb["unanswerable"] + tb["answerable_control"]

    chosen = _partition(seeds, min(total * 2, len(seeds)), 101)
    cases, stats = build_refusal_testbed(
        chosen,
        emb,
        n_unanswerable=tb["unanswerable"],
        n_answerable=tb["answerable_control"],
        similarity_floor=tb["similarity_floor"],
        similarity_ceiling=tb["similarity_ceiling"],
    )
    print(f"refusal: {stats}")
    if stats.get("SHORTFALL"):
        print("  WARNING: shortfall -- the testbed is unbalanced, which makes "
              "Refusal F1 precision and recall incomparable (P1 5.4.3)")
    save_cases(cases, "refusal")


def stage_noise(cfg: dict) -> None:
    seeds = load_seeds(SEEDS_PATH)
    emb = load_embeddings()
    tb = cfg["testbeds"]["noise"]

    chosen = _partition(seeds, min(tb["n_cases"], len(seeds)), 202)
    # Distractors come "from the same corpus" (P1 5.5.2): every seed's passages
    # plus the same-document windows that stock the corpus with genuine topical
    # neighbours. select_distractor_pool() excludes the seed's own evidence and
    # anything carrying the answer.
    distractor_source = [p for s in seeds for p in [*s.passages, *s.extra_passages]]

    cases, stats = build_noise_testbed(
        chosen,
        distractor_source,
        emb,
        n_cases=tb["n_cases"],
        similarity_floor=tb["distractor_similarity_floor"],
        max_pool=tb["distractor_pool_size"],
    )
    print(f"noise: { {k: v for k, v in stats.items() if k != 'warnings'} }")
    for w in stats["warnings"][:5]:
        print(f"  {w}")
    save_cases(cases, "noise")


async def _preflight(generator, model: str) -> bool:
    """Confirm the case generator actually serves before spending the day's quota.

    Appearing in ListModels does not mean a model answers. `gemini-3.7-flash`
    is catalogued on this key but hangs for ~4 minutes and then returns 503
    UNAVAILABLE; because 503 is transient the retry policy retries it, so the
    first seed alone burns ~16 minutes and the build looks merely slow rather
    than broken. One cheap call up front turns that into an immediate, legible
    failure.

    Deliberately not cached under the build's own key: it is a different prompt,
    so it neither pollutes nor is satisfied by the contradiction cache.
    """
    print(f"preflight: calling {model}...")
    resp = await generator.generate(
        GenerationRequest(prompt="Reply with exactly: OK", max_tokens=64, temperature=0.0)
    )
    if not resp.ok:
        print(
            f"\nPREFLIGHT FAILED: {model} did not answer -- {resp.error}\n"
            f"  The model may be listed but not serving. Check availability with a\n"
            f"  direct generateContent call before editing configs/models.yaml.",
            file=sys.stderr,
        )
        return False
    print(f"  ok ({resp.latency_s:.1f}s)")
    return True


async def stage_conflict(cfg: dict, models_cfg: dict, limit: int | None = None) -> None:
    from transformers import pipeline  # noqa: PLC0415

    seeds = load_seeds(SEEDS_PATH)
    tb = cfg["testbeds"]["conflict"]
    target = tb["n_cases"] if limit is None else limit
    # Take the whole seed pool and let the loop stop at `target`.
    #
    # Sizing the pool by a fixed multiple of the target repeatedly guessed
    # wrong: 1.6x assumed ~100% acceptance, and 2.0x was set from a 12-seed
    # pilot that measured 58%. The real rate over the first 25 seeds of the full
    # build was 48%, which needs ~417 seeds for 200 cases -- more than the 400
    # that 2.0x allows. Since the loop breaks as soon as `target` is reached, an
    # oversized pool costs nothing when acceptance is good and is the only thing
    # that rescues the run when it is not.
    #
    # Wider seed reuse across dimensions is expected and already reported:
    # P1 5.5.1 samples ~600 seeds for ~900 cases, and stage_assemble() writes
    # the per-dimension overlap into the manifest.
    #
    # The pool is a deterministic prefix (same shuffle seed), so widening it
    # keeps every already-cached response valid -- a re-run re-issues nothing.
    chosen = _partition(seeds, len(seeds), 303)

    spec = models_cfg["case_generator"]
    # Wrapped above the rate limiter, so a resumed build re-issues nothing and
    # costs no free-tier quota (P1 5.8). This stage is the only one that spends
    # a daily cap, so resumability here is the difference between a failed run
    # costing minutes and costing a day.
    #
    # timeout_s is cut from the 180s default because a healthy call on this
    # prompt returns in 12-19s. The default exists for slow generators; here it
    # only lengthens the failure path, and the failure path is what bit us.
    generator = CachedProvider(
        build_provider("case_generator", spec, models_cfg, timeout_s=90.0),
        ResponseCache(CACHE_PATH),
    )
    await generator.open()
    print(f"case generator: {spec['model']} (family {spec['family']})")

    if not await _preflight(generator, spec["model"]):
        await generator.close()
        raise SystemExit(2)

    print(f"loading NLI model {tb['nli_model']}...")
    nli_pipe = pipeline("text-classification", model=tb["nli_model"], top_k=None)

    class NLI:
        def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
            # A real text pair, not a "</s></s>"-joined string: that is the
            # RoBERTa convention and DeBERTa scores it as one malformed
            # sequence, which would quietly degrade every contradiction score.
            out = nli_pipe({"text": premise, "text_pair": hypothesis})
            rows = out[0] if isinstance(out[0], list) else out
            return {row["label"].lower(): float(row["score"]) for row in rows}

    nli = NLI()
    candidates: list[ConflictCandidate] = []
    cases: list[TestCase] = []
    n_cached = 0
    seed_value = (models_cfg.get("controls") or {}).get("seed")

    for seed in chosen:
        if len(cases) >= target:
            break
        original = seed.answer_passages[0]
        prompt = CONTRADICTION_PROMPT.format(
            query=seed.query, answer=seed.answer, passage=original.text
        )
        resp = await generator.generate(
            GenerationRequest(
                prompt=prompt,
                max_tokens=CONTRADICTION_MAX_TOKENS,
                temperature=0.0,
                seed=seed_value,
            )
        )
        n_cached += int(resp.cached)
        if not resp.ok:
            candidates.append(
                ConflictCandidate(seed, "", 0.0, False, f"generation_failed: {resp.error}")
            )
            continue

        # Two failure modes the NLI gate cannot distinguish from a weak
        # contradiction, kept as their own reject reasons so the acceptance
        # breakdown reported with the benchmark says what actually went wrong.
        if not resp.text.strip():
            candidates.append(ConflictCandidate(seed, "", 0.0, False, "empty_response"))
            continue
        if resp.meta.get("finish_reason") == "MAX_TOKENS":
            candidates.append(ConflictCandidate(seed, resp.text, 0.0, False, "truncated"))
            continue

        ok, score, reason = verify_contradiction(
            original.text, resp.text, seed.answer, nli,
            threshold=tb["contradiction_threshold"],
        )
        candidates.append(ConflictCandidate(seed, resp.text, score, ok, reason))
        if ok:
            cases.append(
                build_conflict_case(
                    seed, resp.text, score,
                    gen_params={"case_generator": spec["model"], "family": spec["family"]},
                )
            )
        if len(candidates) % 25 == 0:
            print(f"  {len(candidates)} attempted, {len(cases)} accepted, {n_cached} cache hits")

    summary = summarise_candidates(candidates)
    print(f"conflict: {summary}")
    print(f"  {n_cached} of {len(candidates)} served from cache (no quota spent)")
    if len(cases) < target:
        print(f"  SHORTFALL: {len(cases)} of {target} requested")
    save_cases(cases, "conflict" if limit is None else "conflict_pilot")

    # The acceptance breakdown is a reportable property of the benchmark
    # (P1 5.5.2 discards rather than repairs), not just console noise.
    report = ROOT / "runs" / ("conflict_candidates.json" if limit is None
                              else "conflict_pilot_candidates.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "model": spec["model"],
        "threshold": tb["contradiction_threshold"],
        "summary": summary,
        "cache_hits": n_cached,
        "rejected": [
            {"seed_id": c.seed.seed_id, "reason": c.reject_reason,
             "score": c.contradiction_score}
            for c in candidates if not c.accepted
        ],
    }, indent=2))
    print(f"  candidate report -> {report}")
    await generator.close()


def stage_validation(cfg: dict) -> None:
    """Export the P1 5.5.3 manual validation sample for human review.

    Writes three things: the machine-readable sample, a readable review sheet
    the annotator works through, and a blank label CSV they fill in. The CSV is
    deliberately dumb -- case id, question, blank answer -- so a second annotator
    needs nothing but a spreadsheet and can start before any interface exists.
    """
    import csv  # noqa: PLC0415

    from ragrobust.dataset.validation import (  # noqa: PLC0415
        checklist_for,
        render_review_sheet,
        stratified_validation_sample,
    )

    vcfg = cfg["validation"]
    cases = load_cases("refusal") + load_cases("conflict") + load_cases("noise")
    if not cases:
        raise SystemExit("no cases built; run the build stages first")

    sample = stratified_validation_sample(
        cases,
        fraction=vcfg["sample_fraction"],
        second_annotator_n=vcfg["second_annotator_cases"],
        rng_seed=cfg.get("rng_seed", 20260721),
    )
    out = ROOT / "runs" / "validation"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "primary_sample.jsonl", "w", encoding="utf-8") as fh:
        for c in sample.primary:
            fh.write(c.model_dump_json() + "\n")

    for name, subset in (("primary", sample.primary),
                         ("second_annotator", sample.second_annotator)):
        with open(out / f"{name}_labels.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "dimension", "question_no", "question",
                        "answer_yes_no", "notes"])
            for c in subset:
                for i, q in enumerate(checklist_for(c), 1):
                    w.writerow([c.case_id, c.dimension.value, i, q, "", ""])

    # Readable review sheet. Self-contained so it can be emailed as one file.
    (out / "review_sheet.html").write_text(render_review_sheet(sample), encoding="utf-8")

    (out / "sample_stats.json").write_text(json.dumps(sample.stats, indent=2))
    print(json.dumps(sample.stats, indent=2))
    print(f"\nwrote -> {out}")
    print("  review_sheet.html        the annotator reads this")
    print("  second_annotator_labels.csv  they fill this in")
    print("  primary_labels.csv       you fill this in")


def stage_prune(cfg: dict) -> None:
    """Repair-or-discard pass over the built testbeds (P1 Section 5.5.3).

    P1: "Cases that fail review are repaired or discarded. The repair-or-discard
    decision and the final acceptance rate are reported with the benchmark
    release." This is the machine half of that: a case whose answer-bearing
    passage does not actually assert the gold answer on a word boundary cannot
    be repaired without re-deriving it from source, so it is discarded and the
    decision recorded.

    Noise cases are discarded by SEED, not by instance. Dropping a single ratio
    would leave that seed with partial ratio coverage and tilt the Noise
    Degradation Curve, which is fitted across ratios per seed.
    """
    from ragrobust.dataset.validation import read_review_failures  # noqa: PLC0415

    discarded: dict[str, list[str]] = {}
    bad_noise_seeds: set[str] = set()

    # The human half of the repair-or-discard decision (P1 5.5.3). Both
    # annotators' live sheets count, plus a standing ledger.
    #
    # The ledger exists because a verdict outlives the sample that drew it: the
    # validation sample is re-drawn whenever a testbed is rebuilt, so a case
    # failed in an earlier round can vanish from the sheet while still sitting
    # in the benchmark. Only THIS directory is read -- an archived sheet from
    # before a rebuild describes cases that no longer have that content, and a
    # noise case in particular keeps its case_id while its distractors are
    # replaced wholesale.
    vdir = ROOT / "runs" / "validation"
    review_failures = read_review_failures(
        sorted(vdir.glob("*_labels.csv")) + [vdir / "review_failures.csv"]
    )
    if review_failures:
        print(f"manual review failed {len(review_failures)} case(s)")

    parts = {name: load_cases(name) for name in ("refusal", "conflict", "noise")}
    for name, cases in parts.items():
        for c in cases:
            failed_review = c.case_id in review_failures
            if failed_review:
                discarded.setdefault(name, []).append(c.case_id)
                if name == "noise":
                    bad_noise_seeds.add(c.seed_id)
                continue
            if c.answer == NO_ANSWER:
                continue
            ap = [p for p in c.retrieved_passages if p.is_answer_bearing]
            if ap and not contains_on_word_boundary(c.answer, ap[0].text):
                discarded.setdefault(name, []).append(c.case_id)
                if name == "noise":
                    bad_noise_seeds.add(c.seed_id)

    # Cases already drawn into the manual validation sample are preferred for
    # retention: an annotator may already be part-way through labelling them,
    # and their labels are what the kappa gate rests on.
    protected: set[str] = set()
    vpath = ROOT / "runs" / "validation" / "primary_sample.jsonl"
    if vpath.exists():
        with open(vpath, encoding="utf-8") as fh:
            protected = {json.loads(line)["case_id"] for line in fh if line.strip()}

    kept_counts = {}
    rebalanced: list[str] = []
    for name, cases in parts.items():
        drop = set(discarded.get(name, []))
        if name == "noise":
            kept = [c for c in cases if c.seed_id not in bad_noise_seeds]
        else:
            kept = [c for c in cases if c.case_id not in drop]

        if name == "refusal":
            # Refusal F1 precision is only defined when both groups are present
            # in comparable numbers (P1 5.4.3), so discarding from one side
            # obliges a matching discard from the other.
            unans = [c for c in kept if not c.is_answerable]
            ans = [c for c in kept if c.is_answerable]
            target = min(len(unans), len(ans))
            def trim(group):
                # Deterministic, and protected cases go last so they survive.
                ordered = sorted(group, key=lambda c: (c.case_id in protected, c.case_id))
                return ordered[len(group) - target:] if len(group) > target else group
            if len(unans) != len(ans):
                dropped_now = ([c.case_id for c in unans] + [c.case_id for c in ans])
                unans, ans = trim(unans), trim(ans)
                surviving = {c.case_id for c in unans + ans}
                rebalanced = [cid for cid in dropped_now if cid not in surviving]
                kept = sorted(unans + ans, key=lambda c: c.case_id)

        kept_counts[name] = {"before": len(cases), "after": len(kept),
                             "discarded": len(cases) - len(kept)}
        save_cases(kept, name)

    report = {
        "reason": "answer-bearing passage does not assert the gold answer on a "
                  "word boundary, OR a human annotator failed the case on a "
                  "P1 5.5.3 checklist question (repair-or-discard)",
        "counts": kept_counts,
        "manual_review_failures": {
            cid: reasons for cid, reasons in sorted(review_failures.items())
        },
        "noise_seeds_discarded": sorted(bad_noise_seeds),
        "refusal_rebalance_discards": sorted(rebalanced),
        "refusal_rebalance_reason": "Refusal F1 precision requires both groups "
                                    "present in comparable numbers (P1 5.4.3)",
        "case_ids": {k: sorted(v) for k, v in discarded.items()},
    }
    out = ROOT / "runs" / "discard_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(kept_counts, indent=2))
    print(f"\ndiscard report -> {out}")


def stage_verify(cfg: dict) -> None:
    """Check the built testbeds against the P1 invariants, in the data.

    The build stages already report counters, but a counter says what the code
    believed it did. This stage re-derives every load-bearing property from the
    written cases, which is the only check that survives a bug in the builder
    itself. It found nothing on refusal and noise; it exists so that stays true
    after any future edit.

    Exits non-zero on failure so it can gate a release of the benchmark.
    """
    from ragrobust.dataset.noise import achieved_ratio  # noqa: PLC0415

    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    ref, con, noi = load_cases("refusal"), load_cases("conflict"), load_cases("noise")
    print(f"loaded: refusal={len(ref)} conflict={len(con)} noise={len(noi)}")

    # --- refusal: the balance Refusal F1 precision depends on (P1 5.4.3) ---
    if ref:
        unans = [c for c in ref if not c.is_answerable]
        ans = [c for c in ref if c.is_answerable]
        check(len(unans) == len(ans),
              f"refusal unbalanced: {len(unans)} unanswerable vs {len(ans)} answerable")
        check(all(c.answer == NO_ANSWER for c in unans),
              "an unanswerable case is missing the NO_ANSWER sentinel")
        check(not any(p.is_answer_bearing for c in unans for p in c.retrieved_passages),
              "an unanswerable case retains an answer-bearing passage")
        check(all(any(p.is_answer_bearing for p in c.retrieved_passages) for c in ans),
              "an answerable control lost its answer passage")
        print(f"  refusal: {len(unans)}/{len(ans)} balance, sentinel discipline OK")

    # --- conflict: two positions, no leak, randomised order (P1 5.5.2, 5.9) ---
    if con:
        thr = cfg["testbeds"]["conflict"]["contradiction_threshold"]
        for c in con:
            orig = [p for p in c.retrieved_passages if p.is_answer_bearing]
            contra = [p for p in c.retrieved_passages if p.is_injected]
            check(len(c.retrieved_passages) == 2 and len(orig) == 1 and len(contra) == 1,
                  f"{c.case_id}: not a clean two-passage conflict case")
            if orig and contra:
                # The contradiction must not restate the gold answer, or the
                # case asserts both positions and cannot be graded on the rubric.
                check(c.answer.lower() not in contra[0].text.lower(),
                      f"{c.case_id}: gold answer survives in the contradiction passage")
                check(c.answer.lower() in orig[0].text.lower(),
                      f"{c.case_id}: original passage does not contain the gold answer")
            check(float(c.gen_params.get("contradiction_score", 0)) >= thr,
                  f"{c.case_id}: contradiction score below the configured threshold")
            recorded = c.gen_params.get("true_passage_position")
            actual = [p.is_answer_bearing for p in c.retrieved_passages].index(True)
            check(recorded == actual,
                  f"{c.case_id}: recorded true_passage_position {recorded} != actual {actual}")
        first = sum(c.retrieved_passages[0].is_answer_bearing for c in con)
        # P1 5.9 requires randomized presentation order. A constant order is the
        # failure this catches; the band is wide because it is not a PRNG test.
        check(0.25 < first / len(con) < 0.75,
              f"presentation order not randomised: true passage first in {first}/{len(con)}")
        print(f"  conflict: two-passage, no answer leak, order {first}/{len(con)} first")

    # --- noise: the NDC x-axis must be exact (P1 5.5.2) ---
    if noi:
        worst = 0.0
        for c in noi:
            got = achieved_ratio(c.retrieved_passages)
            worst = max(worst, abs(got - (c.noise_ratio or 0.0)))
            check(any(p.is_answer_bearing for p in c.retrieved_passages),
                  f"{c.case_id}: noise case lost its answer passage")
        check(worst < 0.01, f"noise ratio drifts up to {worst:.4f} from target")
        print(f"  noise: worst ratio deviation {worst:.4f} across {len(noi)} instances")

    # --- answer flags must be supported by the text, on a word boundary ---
    # A bare substring flagged a passage answer-bearing because it contained
    # "King George III" while the gold answer was "King George I". The model
    # refused correctly and would have been scored wrong.
    for c in ref + con + noi:
        if c.answer == NO_ANSWER:
            continue
        for pas in c.retrieved_passages:
            if pas.is_answer_bearing and not contains_on_word_boundary(c.answer, pas.text):
                failures.append(
                    f"{c.case_id}/{pas.passage_id}: flagged answer-bearing but "
                    f"{c.answer!r} does not occur on a word boundary"
                )

    # --- the central control, on every case ---
    for c in ref + con + noi:
        try:
            SandboxedCorpus(c, distractor_pool=[]).assert_perturbation_holds()
        except Exception as exc:  # noqa: BLE001 - collected and reported
            failures.append(f"{c.case_id}: perturbation does not hold -- {exc}")

    if failures:
        print(f"\nVERIFICATION FAILED: {len(failures)} problem(s)")
        for f in failures[:20]:
            print(f"  {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        raise SystemExit(1)
    print("\nall invariants hold")

    # Pin the release. The label sheets reference case ids, but the cases
    # themselves are not in git -- they are rebuilt from streamed NQ/TriviaQA,
    # so an upstream change could silently produce a different benchmark under
    # the same ids. The manifest makes that visible instead of silent.
    from ragrobust.dataset.validation import (  # noqa: PLC0415
        read_review_failures,
        stale_review_case_ids,
    )

    all_cases = ref + con + noi
    # Same default as stage_assemble, or the two manifests would disagree on
    # version while describing identical cases.
    manifest = Benchmark(version=cfg.get("version", "1.0.0"), cases=all_cases).manifest()

    vdir = ROOT / "runs" / "validation"
    ledger = vdir / "review_failures.csv"
    # Cases a reviewer failed are meant to be gone; only an UNEXPLAINED absence
    # is a stale label.
    stale = stale_review_case_ids(
        sorted(vdir.glob("*_labels.csv")) + [ledger],
        {c.case_id for c in all_cases},
        discarded_case_ids=read_review_failures([ledger]),
    )
    manifest["stale_reviewed_case_ids"] = stale

    out = ROOT / "runs" / "benchmark_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"manifest {manifest['content_hash']} "
          f"({manifest['n_cases']} cases) -> {out}")
    if stale:
        # Not a build failure: the benchmark is sound, the LABELS are stale.
        print(f"  WARNING: {len(stale)} reviewed case(s) are no longer in the "
              f"benchmark, so their verdicts describe cases that no longer exist:")
        for cid in stale[:10]:
            print(f"    {cid}")
        if len(stale) > 10:
            print(f"    ... and {len(stale) - 10} more")


def stage_assemble(cfg: dict) -> None:
    parts = {name: load_cases(name) for name in ("refusal", "conflict", "noise")}
    missing = [n for n, c in parts.items() if not c]
    if missing:
        print(f"WARNING: no cases for {missing}; assembling what exists")

    all_cases = [c for cases in parts.values() for c in cases]
    bench = Benchmark(version=cfg.get("version", "1.0.0"), cases=all_cases)
    manifest = bench.manifest()

    # Seed reuse across dimensions is expected (P1 5.5.1 samples 600 seeds for
    # ~900 cases) but must be visible rather than implicit.
    by_dim = {name: {c.seed_id for c in cases} for name, cases in parts.items() if cases}
    overlaps = {
        f"{a}&{b}": len(by_dim[a] & by_dim[b])
        for i, a in enumerate(sorted(by_dim))
        for b in sorted(by_dim)[i + 1 :]
    }
    manifest["seed_overlap_between_dimensions"] = overlaps
    manifest["evaluation_instances"] = bench.evaluation_instances()

    CASES.mkdir(parents=True, exist_ok=True)
    bench.to_jsonl(str(CASES / "benchmark.jsonl"))
    (CASES / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== MANIFEST ===")
    print(json.dumps(manifest, indent=2))
    print(f"\ninstances/config: {manifest['evaluation_instances']} (P1 target ~2,900)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=["seeds", "embed", "refusal", "noise", "conflict", "verify",
                 "prune", "assemble", "validation", "local"],
        help="'local' runs everything that needs no API key",
    )
    p.add_argument("--dataset-config", default="configs/dataset.yaml")
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="conflict only: stop after N accepted cases and write them to "
             "conflict_pilot.jsonl instead of conflict.jsonl. For piloting the "
             "prompt and the NLI gate on a handful of seeds before committing "
             "the day's free-tier quota.",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.dataset_config).read_text())

    if args.stage == "conflict":
        if not os.environ.get("GOOGLE_API_KEY"):
            print("GOOGLE_API_KEY is unset; the conflict testbed needs the case "
                  "generator (P1 5.5.2).", file=sys.stderr)
            return 2
        asyncio.run(stage_conflict(cfg, load_config(args.models_config), args.limit))
        return 0

    stages = {
        "seeds": stage_seeds,
        "embed": stage_embed,
        "refusal": stage_refusal,
        "noise": stage_noise,
        "prune": stage_prune,
        "verify": stage_verify,
        "validation": stage_validation,
        "assemble": stage_assemble,
    }
    if args.stage == "local":
        for name in ("seeds", "embed", "refusal", "noise"):
            print(f"\n=== {name.upper()} ===")
            stages[name](cfg)
        return 0

    stages[args.stage](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
