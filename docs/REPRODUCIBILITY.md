# Reproducibility appendix

*Written to be used as a report appendix largely as-is. Appendices do not count
toward the 30,000-word limit.*

Benchmark `content_hash 903392537ebef4de` · code at commit `a5b88e7` ·
frozen 26 August 2026.

---

## A.1 What determinism means here, and where it stops

P1 Section 5.6.5 requires the version of every model to be recorded and the
decoding parameters held constant; Section 5.8 requires the experiment to be
"batched and resumable so that interrupted runs can be continued without
re-running completed cases". Both are treated as correctness properties rather
than conveniences, and the pipeline is built so that a rerun of an unchanged
configuration issues **no model calls at all**.

Three mechanisms:

- **Fixed seeds.** `rng_seed: 20260721` at every stage that samples: seed
  selection, distractor drawing, conflict passage presentation order, the
  validation sample, the CRS judge validation sample, the RAGAS held-out subset,
  and the bootstrap.
- **Temperature 0** on every generation, judge and analysis call.
- **Content-addressed response caching.** The cache key covers the model, the
  family, the full prompt, `max_tokens`, temperature, seed, the thinking flag
  and stop sequences. Identical requests are issued at most once, ever.

**Where determinism stops, stated honestly.** The evaluated generators are
served by vLLM on A10G GPUs. Even at temperature 0, batched GPU inference is not
bit-reproducible across differing batch compositions, so a *cold* rerun on
different hardware may not reproduce every generation token-for-token. The
response cache is therefore the reproducibility guarantee: the published cache
replays the run without a GPU. `scripts/demo.py verify` (Appendix A.6) checks
that directly, and states precisely how far the guarantee extends — including
the one case where it does not, which is that two configurations building an
identical prompt address a single cache entry.

Everything downstream of generation — parsing, scoring, metrics, bootstrap,
figures — is pure Python over stored records and **is** bit-reproducible. Reruns
of `analyse_results.py` on unchanged inputs produce byte-identical output; this
was asserted rather than assumed after each of the analysis additions.

---

## A.2 Environment

| component | version |
|---|---|
| Python | 3.13.5 |
| Platform | macOS 26.5.2, arm64 (Apple M3, 16 GB) — orchestration and analysis |
| GPU | NVIDIA A10G via Modal, up to 10 concurrent containers — generation only |
| pydantic | 2.9.2 |
| httpx | 0.27.2 |
| numpy | 2.4.2 |
| datasets | 5.0.1 |
| sentence-transformers | 5.7.0 |
| transformers | 5.15.0 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.11.1 |
| rank-bm25 | ≥ 0.2.2 |

Install: `pip install -e ".[dev,analysis]"` from the repository root.

Note the analysis layer deliberately avoids scipy: the percentile function
underlying every confidence interval is written out in
`src/ragrobust/analysis.py` so that published bounds cannot shift if a library
changes its default interpolation.

---

## A.3 Models

| role | model | family | host |
|---|---|---|---|
| Standard generator 1 | `Qwen/Qwen3-8B` (thinking off) | alibaba | Modal + vLLM |
| Standard generator 2 | `THUDM/glm-4-9b-chat` | zhipu | Modal + vLLM |
| Reasoning generator 1 | `Qwen/Qwen3-8B` (thinking on) | alibaba | Modal + vLLM |
| Reasoning generator 2 | `THUDM/glm-4-9b-chat` + CoT prompt | zhipu | Modal + vLLM |
| Agentic generators | the two reasoning generators (P1 5.6.3) | — | — |
| Case generator | `gemini-3.6-flash` | google | Google AI Studio |
| CRS judge | `claude-haiku-4-5-20251001` | anthropic | Anthropic API |
| Refusal classifier fallback | `claude-haiku-4-5-20251001` | anthropic | Anthropic API |
| RAGAS judge | `gemini-3.5-flash-lite` | google | Google AI Studio |
| Embeddings (DPR, similarity) | `BAAI/bge-base-en-v1.5` | — | local |
| NLI gate (conflict) | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | — | local |

**Anti-circularity (P1 5.9).** The case generator (google), the evaluated
generators (alibaba, zhipu) and the CRS judge (anthropic) occupy three disjoint
model families. `providers/factory.py` enforces this at construction and
`tests/test_dataset.py` asserts it against `configs/models.yaml`, so the control
is audited rather than asserted. All model ids are pinned, never floating
aliases, so the provenance of the testbed cannot change mid-build.

---

## A.4 Rebuilding from scratch

Run everything from the repository root. Stages are ordered; each depends on
the previous one.

```bash
# 0. Environment
pip install -e ".[dev,analysis]"
cp .env.example .env        # then add GOOGLE_API_KEY, ANTHROPIC_API_KEY,
                            # MODAL_VLLM_URL_QWEN, MODAL_VLLM_URL_GLM, MODAL_VLLM_KEY

# 1. Seeds and embeddings: stream NQ + TriviaQA, prepare evidence windows.
#    Slow but entirely local -- no API key needed.
python scripts/build_dataset.py --stage seeds
python scripts/build_dataset.py --stage embed

# 2. Testbeds. Refusal and noise need only the above; conflict needs
#    GOOGLE_API_KEY and the NLI model (~340 Gemini calls).
python scripts/build_dataset.py --stage refusal
python scripts/build_dataset.py --stage noise
python scripts/build_dataset.py --stage conflict --limit 8   # pilot first
python scripts/build_dataset.py --stage conflict

# 3. Re-derive every schema invariant against the built cases, then merge
python scripts/build_dataset.py --stage verify
python scripts/build_dataset.py --stage assemble

# 4. Manual validation sheets (17.5% stratified sample, P1 5.5.3)
python scripts/build_dataset.py --stage validation
#    -> runs/validation/review_sheet.html      annotator works through this
#    -> runs/validation/primary_labels.csv     researcher fills in
#    -> runs/validation/second_annotator_labels.csv
python scripts/label_agreement.py             # the Objective 1 kappa gate

# 5. Apply the repair-or-discard decisions and freeze the content hash
python scripts/build_dataset.py --stage prune

# 6. Deploy the generators, then run the full matrix (resumable)
modal deploy deploy/modal_vllm.py
python scripts/run_experiment.py --plan-only        # inspect the plan first
python scripts/run_experiment.py --concurrency 40

# 7. Score: parse, classify, judge conflicts (~2,375 Haiku calls)
python scripts/score_results.py --results runs/results.jsonl --out runs/scores.json

# 8. Validate the CRS judge against 100 human ratings (the Objective 2 gate)
python scripts/crs_validation.py export       # blind rating sheet
#    ... a human rates runs/validation/crs_human_ratings.csv ...
python scripts/crs_validation.py score

# 9. Secondary and validity analyses
python scripts/ragas_analysis.py select
python scripts/ragas_analysis.py score        # ~1,800 Gemini flash-lite calls
python scripts/ragas_analysis.py analyse
python scripts/generation_artefacts.py        # P1 5.9 artefact check, offline

# 10. Analysis and all ten figures
python scripts/analyse_results.py

# 11. Checks
pytest -q                                     # 377 tests
python scripts/verify_pack.py                 # every reported number vs artefact
python scripts/demo.py verify --n 200         # replay from cache, no network
```

**Cost of a cold rebuild:** ~56 GPU-hours on A10G (~USD 61 of Modal credit),
~USD 2 of Anthropic API for the CRS judge, and under USD 1 of Gemini for case
generation and RAGAS. With the published cache, ~USD 0.

**Wall clock:** the full matrix took roughly 8 hours at ten concurrent
containers. GPU-hours are conserved regardless of replica count, so parallelism
buys wall clock only.

---

## A.5 Configuration

All experimental parameters live in YAML and none is hard-coded:

- `configs/models.yaml` — every model, family, provider, rate limit and
  decoding control.
- `configs/dataset.yaml` — testbed sizes, noise ratios, similarity floors and
  ceilings, distractor pool size, validation fractions, the bootstrap
  parameters, and the practical-significance thresholds.

**The practical-significance thresholds — 0.05 Refusal F1, 0.25 CRS, 0.05
NDC-AUC — were committed to `configs/dataset.yaml` before the first analysis
run.** P1 5.7 requires the threshold "set in advance at a metric-specific
level", and the commit history establishes the ordering: a threshold chosen
after seeing the results would be a conclusion, not a threshold.

---

## A.6 Replaying the run without a GPU

Every model call is in the content-addressed cache, so the experiment can be
re-executed offline:

```bash
python scripts/demo.py list                    # the curated cases
python scripts/demo.py case conflict-nq-4643   # all three classes on one case
python scripts/demo.py noise tqa-2481          # one seed across five noise ratios
python scripts/demo.py verify --n 200          # provenance and coverage check
```

`verify` re-runs the real pipeline objects against the cache. `ReplayProvider`
raises on a cache miss rather than falling through to the network, and imports
no HTTP client at all, so a pass is proof that nothing was fetched.

What it asserts, and what it only reports, are deliberately different. It
asserts **provenance**: every response served is byte-identical to a response
this run recorded for that generator. It asserts byte-exact reproduction for
every instance whose cache entry is provably unshared — the two retriever arms
ranked the case differently, so each addressed its own entry. It *reports* the
overall reproduction rate, which is ~97 percent, because the cache is keyed on
the prompt rather than on the configuration: two arms that rank a case
identically issue one request and share one entry, and the survivor is whichever
finished last. Those instances are checked against both arms' records rather
than excused. On a sample of 400 single-call instances, 175 of 175 unshared
entries reproduced exactly and 225 of 225 shared entries returned one of the two
recorded responses.

The agentic decompose prompt is built from the query alone, so it is shared by
every instance carrying that query. A chain that starts from another instance's
decomposition asks a question this configuration never asked, which is the only
way a miss can occur; `verify` classifies those and fails on any it cannot
account for.

---

## A.7 Artefact manifest

SHA-256 prefixes (first 16 hex characters), re-verified 27 August 2026.
Every checksum is unchanged since the 26 August freeze; the size column was
previously reported in allocated blocks and now gives actual file size.

| artefact | size | sha256 (16) |
|---|---|---|
| `data/cases/benchmark.jsonl` | 34M | `d116bb6a1c92766f` |
| `runs/results.jsonl` | 98M | `08157dd688b4fb92` |
| `runs/scored.jsonl` | 13M | `fda0a2e8d8517a57` |
| `runs/analysis.json` | 40K | `5cdc62a74ac447b1` |
| `runs/scores.json` | 22K | `7225a805f10fd409` |
| `runs/ragas_analysis.json` | 10K | `26f23c779a859ac7` |
| `runs/ragas_scores.jsonl` | 195K | `486206e428f5f80f` |
| `runs/generation_artefacts.json` | 6.9K | `5a10281ea6e8f376` |
| `runs/validation/label_agreement.json` | 1.2K | `8303481f9bfe6e05` |
| `runs/validation/crs_judge_agreement.json` | 703B | `3a2976eadfc50163` |
| `runs/validation/crs_human_ratings.csv` | 14K | `a47e7b312301593d` |
| `runs/validation/primary_labels.csv` | 32K | `3ff943994016e1db` |
| `runs/validation/second_annotator_labels.csv` | 10K | `d4df7476a3145b3d` |
| `runs/benchmark_manifest.json` | 1.1K | `b11b017530d34ca0` |
| `runs/discard_report.json` | 6.1K | `a38873270a2644f9` |

**Human-produced records, not regenerable by any rerun:** the two annotator
label sheets, the 100 CRS human ratings, the repair-or-discard ledger
(`discard_report.json`, required by P1 5.5.3), and the pre-repair judge record
in `runs/validation/prefix_judge/`. These are tracked in version control for
that reason; the large generated artefacts are not, and are preserved in a
dated run snapshot instead.

---

## A.8 Testing

377 tests, no API keys, no network, no downloads: `pytest -q`.

The project convention is that **every expected value is computed by hand
before being asserted** — never read back from the implementation. Each test's
docstring carries the arithmetic. The discipline caught real defects rather
than merely documenting behaviour, including two in the generation-artefact
check that would have produced a false all-clear on a validity threat: gating
the template signal on absolute token count missed a stock phrase spread thinly
across every case, and without a cross-case minimum the check fired on its own
design, since the replaced claim is a generated-only token in exactly one case
by construction.

Three checks are worth naming as reproducibility guarantees in their own right:

- `tests/test_dataset.py` audits `configs/models.yaml` for the P1 5.9
  anti-circularity constraint, so the control cannot be silently broken by a
  config edit.
- `scripts/verify_pack.py` asserts all 66 reported headline figures against the
  artefacts they came from.
- `scripts/demo.py verify` replays the pipelines from cache with no network,
  asserts that every response served was recorded by this run, and reports the
  reproduction rate against the stored records.
