#!/usr/bin/env python3
"""Run the benchmark across the P1 Table 5.5 configuration matrix.

    python scripts/run_experiment.py --smoke 5      # 5 cases per config, real calls
    python scripts/run_experiment.py                # the full matrix
    python scripts/run_experiment.py --configs naive'|'bm25'|'standard_1

Resumable by construction: results append to one JSONL and the completed keys
are rebuilt from that file on start, so an interrupted run continues rather than
restarting (P1 5.8). The response cache sits underneath, so even a re-run of a
pair that was recorded but lost costs no new generator call.

Deliberately kept thin. Matrix expansion and resume bookkeeping live in
`ragrobust.runner` where they are unit-tested without a GPU; this file only
wires providers, pipelines, and corpora to them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ragrobust.cache import ResponseCache  # noqa: E402
from ragrobust.corpus import SandboxedCorpus, build_distractor_pool  # noqa: E402
from ragrobust.pipelines.agentic import AgenticPipeline  # noqa: E402
from ragrobust.pipelines.base import RetrievalPolicy  # noqa: E402
from ragrobust.pipelines.naive import NaivePipeline  # noqa: E402
from ragrobust.pipelines.reasoning import ReasoningPipeline  # noqa: E402
from ragrobust.providers.cached import CachedProvider  # noqa: E402
from ragrobust.providers.factory import build_provider, load_config  # noqa: E402
from ragrobust.runner import (  # noqa: E402
    ResultStore,
    expand_matrix,
    pending_work,
    run_plan,
    smoke_subset,
)
from ragrobust.schema import TestCase  # noqa: E402

DATA = ROOT / "data"
RUNS = ROOT / "runs"
BENCHMARK = DATA / "cases" / "benchmark.jsonl"
CACHE_PATH = DATA / "cache" / "generators.sqlite"
RESULTS = RUNS / "results.jsonl"


def load_benchmark(path: Path) -> list[TestCase]:
    with open(path, encoding="utf-8") as fh:
        return [TestCase.model_validate_json(line) for line in fh if line.strip()]


def make_pipeline(config, provider, policy, models_cfg):
    """Construct the pipeline for one matrix cell.

    The reasoning form travels from the generator's own spec, never from the
    pipeline class: `reasoning_1` elicits reasoning by thinking mode and
    `reasoning_2` by a chain-of-thought prompt (P1 5.6.2), and the agentic class
    inherits whichever form its generator carries (P1 5.6.3).
    """
    spec = models_cfg["generators"][config.generator_key]
    controls = models_cfg.get("controls") or {}
    seed = controls.get("seed")
    thinking = bool(spec.get("thinking", False))
    cot = bool(spec.get("cot_prompt", False))
    # One value for every class. Passing the per-class defaults through would
    # give the naive arm half the reasoning arm's output budget and confound the
    # comparison the study exists to make (P1 5.6.5).
    max_tokens = int(controls.get("max_output_tokens", 4096))
    temperature = float(controls.get("temperature", 0.0))

    if config.pipeline_class == "naive":
        return NaivePipeline(provider, policy, max_tokens=max_tokens,
                             temperature=temperature, seed=seed)
    if config.pipeline_class == "reasoning":
        return ReasoningPipeline(provider, policy, thinking=thinking,
                                 cot_prompt=cot, max_tokens=max_tokens,
                                 temperature=temperature, seed=seed)
    return AgenticPipeline(
        provider, policy, thinking=thinking, cot_prompt=cot,
        max_steps=int(controls.get("max_agentic_steps", 5)),
        max_tokens=max_tokens, temperature=temperature, seed=seed,
    )


async def main_async(args: argparse.Namespace) -> int:
    models_cfg = load_config(args.models_config)
    controls = models_cfg.get("controls") or {}
    all_cases = load_benchmark(Path(args.benchmark))
    cases = all_cases
    configs = expand_matrix(models_cfg)
    if args.configs:
        wanted = set(args.configs.split(","))
        configs = [c for c in configs if c.config_id in wanted]
        if not configs:
            print(f"no configuration matches {sorted(wanted)}", file=sys.stderr)
            return 2
    if args.smoke:
        cases = smoke_subset(cases, args.smoke)

    store = ResultStore(RESULTS if not args.out else Path(args.out))
    plan = run_plan(cases, configs, store)
    print(json.dumps(plan, indent=2))
    if args.plan_only:
        store.close()
        return 0
    if plan["pending"] == 0:
        print("nothing to do")
        store.close()
        return 0

    # Loaded only once the run is known to have work: --plan-only should not
    # pay for a model download to print six numbers.
    encoder = None
    if any(c.retriever == "dpr" for c in configs):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        emb_model = yaml.safe_load(Path(args.dataset_config).read_text())["embedding_model"]
        print(f"loading dense encoder {emb_model} for the DPR arm...")
        encoder = SentenceTransformer(emb_model)

    cache = ResponseCache(CACHE_PATH)
    providers: dict[str, CachedProvider] = {}
    for key in {c.generator_key for c in configs}:
        providers[key] = CachedProvider(
            build_provider(key, models_cfg["generators"][key], models_cfg), cache
        )
        await providers[key].open()

    # Drawn from the WHOLE benchmark, never from the subset being run. P1 5.6.5
    # requires "a controlled distractor pool", and a pool assembled from
    # whichever cases this invocation happens to select is not controlled: it
    # made a --smoke run construct a different sandbox from the full run, and
    # the same case under the same configuration and seed answered "1934" in one
    # and "INSUFFICIENT EVIDENCE" in the other. Smoke results have to predict
    # full-run results or they are not a test of anything.
    #
    # build_distractor_pool re-checks lexically per case, so an unflagged
    # answer-bearing passage cannot slip in and defeat that case's perturbation.
    all_passages = [p for c in all_cases for p in c.retrieved_passages]

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    failed = 0
    started = time.time()
    total = plan["pending"]
    lock = asyncio.Lock()

    async def run_one(item) -> None:
        nonlocal done, failed
        async with sem:
            config = item.config
            policy = RetrievalPolicy(
                retriever=config.retriever,
                top_k=int(controls.get("top_k", 5)),
                max_context_tokens=int(controls.get("max_context_tokens", 8192)),
                encoder=encoder,
            )
            pool = build_distractor_pool(all_passages, item.case.answer, max_size=20)
            corpus = SandboxedCorpus(item.case, distractor_pool=pool)
            if config.retriever == "dpr" and encoder is not None:
                corpus.index_dense(encoder)
            pipeline = make_pipeline(config, providers[config.generator_key], policy, models_cfg)
            t0 = time.time()
            try:
                result = await pipeline.run(corpus)
                payload = result.as_record()
            except Exception as exc:  # noqa: BLE001 - recorded, never fatal to the run
                payload = {"final_answer": None, "raw_text": "", "error": f"{type(exc).__name__}: {exc}"}
            payload["latency_s"] = round(time.time() - t0, 3)
            async with lock:
                store.record(item, payload)
                done += 1
                if payload.get("error"):
                    failed += 1
                if done % args.report_every == 0 or done == total:
                    rate = done / max(1e-9, time.time() - started)
                    eta = (total - done) / rate / 3600 if rate else float("inf")
                    print(f"  {done}/{total} done, {failed} failed, "
                          f"{rate:.2f}/s, eta {eta:.1f}h", flush=True)

    tasks = [asyncio.create_task(run_one(item)) for item in pending_work(cases, configs, store)]
    await asyncio.gather(*tasks)

    for p in providers.values():
        await p.close()
    store.close()
    elapsed = (time.time() - started) / 3600
    print(f"\ncompleted {done} instances in {elapsed:.2f}h, {failed} failed")
    print(f"results -> {RESULTS if not args.out else args.out}")
    return 1 if failed and args.strict else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default=str(BENCHMARK))
    p.add_argument("--models-config", default="configs/models.yaml")
    p.add_argument("--dataset-config", default="configs/dataset.yaml")
    p.add_argument("--out", default=None, help="results JSONL (default runs/results.jsonl)")
    p.add_argument("--configs", default=None,
                   help="comma-separated config_ids to run, e.g. 'naive|bm25|standard_1'")
    p.add_argument("--smoke", type=int, default=0,
                   help="run only N cases per dimension; use before the full matrix")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--report-every", type=int, default=50)
    p.add_argument("--plan-only", action="store_true", help="print the plan and exit")
    p.add_argument("--strict", action="store_true", help="exit non-zero if any instance failed")
    args = p.parse_args()

    if not args.plan_only and not os.environ.get("MODAL_VLLM_URL"):
        print("MODAL_VLLM_URL is unset; the evaluated generators are served on Modal.",
              file=sys.stderr)
        return 2
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
