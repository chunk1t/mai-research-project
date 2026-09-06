# The offline demo

Replays the finished experiment with no network, no GPU and no API key. Every
generator call in the run was content-addressed into `data/cache/generators.sqlite`,
so a finished configuration re-executes through the real `NaivePipeline`,
`ReasoningPipeline` and `AgenticPipeline` objects with `ReplayProvider`
(`src/ragrobust/providers/replay.py`) standing in for the model. The pipelines,
the sandboxed corpus, BM25 retrieval and the answer parsing all run for real.

Run everything from the repository root.

```bash
python scripts/demo.py list                     # the curated cases
python scripts/demo.py case conflict-nq-4643    # three pipeline classes, one case
python scripts/demo.py noise tqa-2481           # one seed across five noise ratios
python scripts/demo.py verify --n 200           # provenance and coverage check
```

Flags: `--generator {1,2}` picks the base model (1 = Qwen3-8B, 2 = GLM-4-9B,
per `configs/models.yaml`), `--retriever {bm25,dpr}` picks the arm, `--trace`
prints reasoning traces, `--full-text` prints passages in full, `--no-color`
strips ANSI for piping into a file or a slide.

**`--retriever dpr` downloads a sentence-transformers model on first use.** BM25
is the default for that reason: it needs nothing beyond the repository.

## Requirements

| | |
|---|---|
| `data/cases/benchmark.jsonl` | 36 MB, 2,858 cases — loaded in full even to run one case |
| `data/cache/generators.sqlite` | 72 MB, 32,355 responses |
| `runs/results.jsonl`, `runs/scored.jsonl` | the stored records and their verdicts |

The whole benchmark is loaded for a single case on purpose. `build_distractor_pool`
draws from every passage in it, and `run_experiment.py` records what happens when
it does not: a pool assembled from a subset builds a different sandbox, and the
same case under the same configuration and seed answered `1934` in one and
`INSUFFICIENT EVIDENCE` in the other. It costs about a tenth of a second.

## What `verify` establishes

It **asserts** three things and fails on any violation:

1. Every response served is byte-identical to a response this run recorded for
   that generator. Replay invents nothing.
2. Every instance whose cache entry is provably unshared reproduces its stored
   record byte for byte.
3. Every instance that shares an entry with the other retriever arm returns one
   of the two recorded responses.

It **reports** the overall reproduction rate, which is about 97 percent rather
than 100. The cache is keyed on the prompt, not on the configuration that issued
it, so two configurations that build an identical prompt address one entry and
the run's last writer holds it. That happens two ways: two retriever arms can
rank a case's passages into the same order, and the agentic decompose prompt is
built from the query alone, so every instance carrying that query shares it. The
second is also the only way a cache miss can occur — a chain that starts from
another instance's decomposition asks a question this configuration never asked.
`verify` classifies those and fails on any it cannot account for.

This is Appendix A.1's determinism caveat, made checkable. No reported number
depends on it: scoring ran off `runs/results.jsonl`, which recorded each arm's
own response.

## The HTML casebook

```bash
python scripts/build_demo_data.py       # writes demo/index.html (~500 KB)
python scripts/build_demo_data.py --json   # also demo/demo_data.json
```

One self-contained file: 24 cases, 144 stored records, no server and no
repository needed. It reads the stored records rather than replaying, so it
shows the response each reported number was computed from. `demo/template.html`
is the page; `__DEMO_DATA__` is where the payload is substituted in.

Open a specific case with a fragment: `index.html#conflict-nq-4643`.

## Troubleshooting

**`chain forked at the shared decompose entry`** — expected on a handful of
agentic instances, explained above. The case display prints the answer that
configuration recorded, so nothing is lost. Use `--generator 2` if you want a
clean run of that particular case on stage.

**`unknown case`** — `python scripts/demo.py list` prints the curated set. Any
`case_id` in `data/cases/benchmark.jsonl` works, not only the curated ones.

**A `verify` failure** is a real finding, not a flake: the sample is drawn with a
fixed seed (`--seed`), so it is the same 200 instances every time.

## Tests

`tests/test_demo.py` and the replay section of `tests/test_providers.py` cover
this. The curated case ids are checked against the benchmark, so a renamed case
fails the suite rather than the demonstration.
