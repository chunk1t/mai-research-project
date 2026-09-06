# ragrobust

Benchmarking RAG robustness under imperfect retrieval across naive, reasoning,
and agentic pipelines. Implementation for Research Project Phase 2 (WQF7023).

Lee Chun Kit (23076218), Universiti Malaya.

**This repository ships the code and the frozen benchmark**, so the report can
be read against the evidence it was written from. `data/cases/benchmark.jsonl`
is the 2,858-case benchmark that was evaluated, `runs/` holds the artefacts
every cited number is read from, and `docs/` carries the reproducibility
appendix and the ledger tying each reported number to its artefact.

What is deliberately absent is the regenerable bulk: the raw generations
(~102 MB), the response cache, the seed corpora, the figures (they regenerate,
see below), and the intermediate files from superseded runs. The report and
presentation PDFs sit in the folder above this one.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
```

No API keys, no network, no downloads. Expect **377 passed**.

## What runs here, and what does not

The test suite is fully self-contained — every expected value is hand-computed
in the test rather than read back from the implementation, so the suite is a
specification rather than a snapshot.

The pipeline entry points in `scripts/` are included because they are the code
under review, but they need artefacts this repository does not ship:

| Script | Status here |
|---|---|
| `verify_pack.py` | **runs** — re-checks all 66 cited numbers, no arguments needed |
| `analyse_results.py` | **runs** — regenerates `runs/analysis.json` and the figures from the shipped `runs/scored.jsonl` |
| `smoke_build.py` | **runs** — builds all three testbeds end to end on mock seeds, no network |
| `run_experiment.py`, `score_results.py` | needs live model endpoints |
| `build_demo_data.py` | rebuilds `demo/index.html`; the built casebook already ships |
| `demo.py` | **runs** — replays the study offline from the shipped cache; no GPU, API key or network |

`tests/test_demo.py` runs its end-to-end check against the shipped cache.

## Verifying the frozen benchmark

The report identifies the evaluated benchmark by `content_hash 903392537ebef4de`.
That hash is a SHA-256 over each case's query, passage texts, answer and noise
ratio (`schema.Benchmark.manifest`), so it can be recomputed from the shipped
file rather than taken on trust:

```bash
python - <<'EOF'
import hashlib
from ragrobust.schema import TestCase
cases = [TestCase.model_validate_json(line)
         for line in open("data/cases/benchmark.jsonl", encoding="utf-8")]
print(len(cases), hashlib.sha256(
    "".join(sorted(c.content_hash() for c in cases)).encode()).hexdigest()[:16])
EOF
# 2858 903392537ebef4de  -- matching data/cases/manifest.json
```

Every figure quoted in the report is machine-checked against the artefact it was
read from:

```bash
python scripts/verify_pack.py
# checked 66 of 66 ledger rows against their artefacts
# checked 3 'N of nine' prose claims against runs/analysis.json
# all ledger values match their artefacts
```

`docs/NUMBERS_LEDGER.md` is the ledger it reads: one row per cited number,
naming the artefact and the JSON path that number comes from, so a reviewer can
also follow any single figure back by hand.

The analysis itself reproduces from the shipped per-instance scores:

```bash
python scripts/analyse_results.py       # runs/scored.jsonl -> runs/analysis.json + runs/figures/
```

This rewrites `runs/analysis.json` byte-identically and regenerates all ten
report figures into `runs/figures/`. The figures are not shipped, because this
command reproduces them from the shipped scores.

### Where the evidence lives

| Path | What it holds |
|---|---|
| `data/cases/benchmark.jsonl` | the 2,858 frozen cases: 200 refusal, 198 conflict, 2,460 noise |
| `data/cases/manifest.json` | case counts and the content hash |
| `runs/benchmark_manifest.json` | the same hash as run, plus the stale-review case ids |
| `runs/scored.jsonl` | per-instance scores for all twelve pipeline x retriever x model configurations |
| `runs/scores.json`, `runs/analysis.json` | aggregate metrics and the paired comparisons |
| `runs/ragas_analysis.json`, `runs/ragas_scores.jsonl` | the RAGAS convergent-validity check |
| `runs/generation_artefacts.json`, `runs/discard_report.json` | what the case build produced and rejected |
| `runs/validation/` | human annotation: primary and second-annotator labels, CRS judge agreement, review sheets |
| `docs/REPRODUCIBILITY.md` | environment, seeds, model versions, and the full rebuild-from-scratch sequence |
| `docs/NUMBERS_LEDGER.md` | all 66 reported numbers, each mapped to its artefact and JSON path |

The human annotation files carry no annotator identifiers — only case ids,
labels and free-text notes.

## Layout

```
src/ragrobust/       the library
  dataset/           testbed construction: seeds, refusal, noise, conflict
  metrics/           Refusal F1, Conflict Resolution Score, Noise Degradation Curve
  parsing/           three-way response classifier and answer extraction
  pipelines/         naive, reasoning, agentic, and the shared prompts
  providers/         model clients, response cache, replay provider
  corpus.py          the per-case sandboxed corpus (the central control)
scripts/             entry points for building, running, scoring, analysing
tests/               377 tests, hand-computed expectations
configs/             models.yaml and dataset.yaml — all model and threshold choices
deploy/              Modal vLLM serving script
demo/                self-contained offline casebook; open index.html directly
data/cases/          the frozen benchmark and its manifest
runs/                result artefacts, per-instance scores, human validation
docs/                reproducibility appendix and the numbers ledger
```

Every model choice and every threshold lives in `configs/`, never in code.

## Design invariants enforced in code

These are the properties that make the experiment valid. Each is asserted rather
than assumed, so a violation fails loudly instead of silently biasing a metric.

**Refusal cases are balanced and coherent.** An unanswerable case carries the
`NO_ANSWER` sentinel and contains no answer-bearing passage. An answerable
control does the opposite. Enforced in `schema.TestCase`. Without both classes
present, refusal precision equals one by construction and Refusal F1 collapses
into recall (P1 Section 5.4.3).

**The sandbox cannot be escaped.** `SandboxedCorpus` confines all retrieval,
including agentic re-retrieval, to the case's own passage set plus a controlled
distractor pool, and refuses a pool containing answer-bearing passages. This is
the control that makes any observed agentic gain attributable to orchestration
rather than to retrieval quality (P1 Section 5.6.5).

**Noise ratios are over tokens and are exact.** The final distractor is truncated
to land on the token budget, because whole-passage granularity overshoots badly
at low ratios. The ratio is the x axis of the Noise Degradation Curve, so an
inaccurate ratio puts a case at the wrong point on the curve and biases NDC-AUC
and NDC-50.

**Model families stay disjoint across roles.** `tests/test_dataset.py` audits
`configs/models.yaml` so the case generator, the evaluated generators, and the
CRS judge never share a family (P1 Section 5.9). `providers/factory.py` enforces
the same rule at construction, so a hand-edited config fails before any API call
is paid for rather than after a partial run.

**Only the final answer is scored.** Every provider splits the reasoning trace
into `GenerationResponse.reasoning_trace` and leaves `text` answer-only, so no
downstream consumer can score a trace by accident (P1 Section 5.6.4). Without
this, a verbose reasoning model would be scored partly on its trace and the
comparison between pipeline classes would be confounded by verbosity.

**Shortfalls are never silent.** Both testbed builders emit an explicit
`SHORTFALL` marker rather than quietly returning fewer cases than requested.

## Model allocation

Evaluated generators are self-hosted because 69,600 calls cannot fit the USD 20
budget on hosted APIs. See `configs/models.yaml`.

| Role | Model | Family | Host |
|---|---|---|---|
| Standard 1 | Qwen3-8B (thinking off) | alibaba | Modal, vLLM |
| Standard 2 | GLM-4-9B-Chat | zhipu | Modal, vLLM |
| Reasoning 1 | Qwen3-8B (thinking on) | alibaba | Modal, vLLM |
| Reasoning 2 | GLM-4-9B-Chat + CoT prompt | zhipu | Modal, vLLM |
| Agentic | subset of the reasoning pair | alibaba, zhipu | Modal, vLLM |
| Case generator | gemini-3.6-flash | google | AI Studio, billed Tier 1 key |
| RAGAS judge | gemini-3.5-flash-lite | google | AI Studio, same billed key |
| CRS judge | claude-haiku-4-5-20251001 | anthropic | API, ~USD 4 |

The Google roles need a billed key, not the free tier: the free tier allows
20 requests per day per model (measured 24 August 2026), and the conflict
build alone needs ~340 calls, so it cannot construct the RQ2 testbed at all.

Standard and reasoning classes share base models deliberately. Toggling thinking
mode on the same weights isolates the reasoning variable more tightly than
swapping in a different model would, which is what RQ1 to RQ3 actually ask. This
is a refinement on the P1 Table 5.3 examples and should be documented as such in
Chapter 6.
