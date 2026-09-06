# ragrobust

Benchmarking RAG robustness under imperfect retrieval across naive, reasoning,
and agentic pipelines. Implementation for Research Project Phase 2 (WQF7023).

Lee Chun Kit (23076218), Universiti Malaya.

**This repository is the source code only.** The benchmark data, the run
artefacts, the report and the presentation are not included: they are large,
regenerable or not code. What is here is everything needed to read, test and
understand the implementation.

## Quick start

```bash
pip install -e ".[dev]"
pytest -q
```

No API keys, no network, no downloads. Expect **376 passed, 1 skipped**.

## What runs here, and what does not

The test suite is fully self-contained — every expected value is hand-computed
in the test rather than read back from the implementation, so the suite is a
specification rather than a snapshot.

The pipeline entry points in `scripts/` are included because they are the code
under review, but they need artefacts this repository does not ship:

| Script | Needs |
|---|---|
| `run_experiment.py`, `score_results.py`, `analyse_results.py` | a built benchmark and live model endpoints |
| `demo.py`, `build_demo_data.py` | `data/cases/benchmark.jsonl` and the response cache |
| `verify_pack.py` | the writing pack, which is report material and not shipped here |

`tests/test_demo.py` skips its end-to-end check for the same reason. The skip is
a guard, not a failure.

`python scripts/smoke_build.py` does run standalone: it builds the three
testbeds end to end on mock seeds, with no network.

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
tests/               376 tests, hand-computed expectations
configs/             models.yaml and dataset.yaml — all model and threshold choices
deploy/              Modal vLLM serving script
demo/                self-contained offline casebook; open index.html directly
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
