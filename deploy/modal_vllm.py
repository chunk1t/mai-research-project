"""Modal app serving the evaluated generators behind vLLM.

Why self-hosted at all: P1 Section 5.8 budgets roughly seventy thousand
generator calls against a fixed budget. Hosted per-token pricing cannot absorb
that, and Google's free tier is capped near 1,500 requests a day, so the four
evaluated generators run on Modal GPU credits instead (HANDOFF Section 2).

One vLLM server process serves one model, so each model gets its own Modal
function and its own endpoint. The model identifiers are read from
`configs/models.yaml` at deploy time rather than hardcoded, keeping the project
rule that swapping a model is a configuration edit.

Deploy:
    modal deploy deploy/modal_vllm.py

The printed URL goes into `.env` as MODAL_VLLM_URL, and the shared secret as
MODAL_VLLM_KEY. `providers/openai_compat.py` then talks to it unchanged: vLLM
exposes the OpenAI chat-completions shape, which is exactly what that client
already speaks.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal
import yaml

# --------------------------------------------------------------------------
# Serving parameters. These are the knobs worth turning if the throughput spike
# says the run does not fit.
# --------------------------------------------------------------------------

# A10G (24 GB). L40S was chosen first, on an estimate that the largest case
# needed ~18,800 tokens; counting with the actual Qwen tokenizer put the true
# maximum at 10,574, so a 16,384 window suffices and A10G still holds it.
# (L40S also requires a payment method on the Modal account, which credits alone
# do not satisfy.) If GLM-4-9B cannot fit its KV cache at this window -- it is
# ~18.8 GB of weights against A10G's 24 GB -- the escalation is a payment method
# and RAGROBUST_GPU=L40S, which is a one-word change.
GPU_TYPE = os.environ.get("RAGROBUST_GPU", "A10G")

# vLLM is pinned so a re-run months later serves the same engine. P1 Section
# 5.6.5 requires recording the version of each model; the serving stack is part
# of that record.
VLLM_VERSION = "0.8.5"

# Sized by COUNTING, and re-measured after every dataset change. Four previous
# values each singled out exactly the r=0.90 noise cases -- the rightmost point
# of the Noise Degradation Curve, where NDC-AUC, NDC-50 and NDC-Cliff all live:
#
#   8,192  rejected every r=0.90 case outright
#   12,288 overflowed 479 of 2,831 once the output budget rose to 4,096
#   20,480 fitted the old benchmark, but the rebuilt one carries 700-token
#          same-article distractors, which pushed r=0.90 prompts from a 10,574
#          maximum to 17,568 and put 14% of them back over the line
#   32,768 clears everything but leaves GLM-4-9B (~18.8 GB of weights on a 24 GB
#          A10G) roughly one concurrent sequence, which is slower than the
#          overflow it prevents
#
# Measured directly, tokenizing 150 r=0.90 cases with AutoTokenizer("Qwen/Qwen3-8B"):
#   median 9,732   p90 12,507   p99 16,148   max 17,568
#   17,568 + ~350 template + 6,144 completion = 24,062, inside 24,576.
MAX_MODEL_LEN = 24576

# How long a server may sit idle before Modal releases the GPU. Long enough to
# span a batch boundary, short enough not to burn credits between runs.
SCALEDOWN_WINDOW_S = 300

# Weight download can be slow on a cold volume; vLLM then compiles CUDA graphs.
STARTUP_TIMEOUT_S = 20 * 60

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _evaluated_models() -> list[str]:
    """Distinct model identifiers to serve, read from the project config.

    Standard and reasoning arms share base weights by design (HANDOFF decision
    2), so the four generator entries collapse to two servers. Thinking mode is
    a per-request flag, not a separate deployment.
    """
    cfg = yaml.safe_load((_REPO_ROOT / "configs" / "models.yaml").read_text())
    seen: list[str] = []
    for spec in (cfg.get("generators") or {}).values():
        if spec.get("provider") == "modal_vllm" and spec["model"] not in seen:
            seen.append(spec["model"])
    return seen


MODELS = _evaluated_models()

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        "huggingface_hub[hf_transfer]",
        # Pinned below 5.x deliberately. vLLM 0.8.5 uses the transformers 4.x
        # tokenizer API, and transformers 5 removed
        # `all_special_tokens_extended`, which makes the engine fail at
        # tokenizer init with an AttributeError long after the image builds
        # cleanly. Leaving this unpinned lets the resolver pick 5.x.
        "transformers>=4.51.1,<5",
    )
    # hf_transfer makes the first weight pull several times faster, which
    # matters because a cold start otherwise dominates a short spike.
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_V1": "1"})
    # This module is imported in the container as well as locally, and it reads
    # the model roster from the project config at import time. Only the script
    # itself is mounted by default, so the config is baked into the image at the
    # path the container's _REPO_ROOT resolves to ("/"). Copying it keeps
    # configs/models.yaml the single source of truth for the model identifiers
    # rather than duplicating them into an environment variable.
    .add_local_file(_REPO_ROOT / "configs" / "models.yaml", "/configs/models.yaml", copy=True)
)

# Weights persist across deploys, so only the first start pays the download.
hf_cache = modal.Volume.from_name("ragrobust-hf-cache", create_if_missing=True)

app = modal.App("ragrobust-vllm")


def _serve(model_id: str) -> None:
    """Start the vLLM OpenAI-compatible server in this container."""
    cmd = [
        "vllm",
        "serve",
        model_id,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        # vLLM defaults to 0.90, leaving 10% of the card unused. On a 24 GB A10G
        # that is 2.4 GB, which matters enormously for GLM-4-9B: its weights are
        # ~18.8 GB, so the default left 0.59 GiB of KV cache against the 0.62 GiB
        # needed to serve a single 16,384-token sequence, and the engine refused
        # to start:
        #
        #   ValueError: To serve at least one request with the model's max seq
        #   len (16384), 0.62 GiB KV cache is needed, which is larger than the
        #   available KV cache memory (0.59 GiB).
        #
        # 0.95 frees roughly another 1.2 GB, enough for a handful of concurrent
        # long-context sequences. Qwen3-8B has more headroom either way and
        # simply gets more KV cache, hence more concurrency, from the same flag.
        "--gpu-memory-utilization",
        "0.95",
        # Determinism support. P1 Section 5.6.5 holds decoding constant; the
        # per-request seed the provider sends is honoured by the engine.
        "--seed",
        "20260721",
    ]

    # Only the reasoning-trained arm needs a trace parser, and the available
    # choices are version-dependent: vLLM 0.8.5 offers deepseek_r1 and granite
    # but no qwen3 parser. deepseek_r1 parses <think>...</think>, which is the
    # form Qwen3 emits, so it is the correct choice here. GLM-4 is the
    # chain-of-thought-prompted arm and emits no think tags, so giving it a
    # parser would be meaningless.
    #
    # If a future vLLM drops this flag entirely, nothing breaks:
    # providers/_common.py falls back to parsing think tags out of the content,
    # and a test covers that path.
    if "qwen" in model_id.lower():
        cmd += ["--reasoning-parser", "deepseek_r1"]

    # GLM-4 ships its modelling code in the weights repository rather than in
    # transformers, so the engine refuses to load it without this flag:
    #
    #   ValueError: The repository THUDM/glm-4-9b-chat contains custom code
    #   which must be executed to correctly load the model. Please pass the
    #   argument `trust_remote_code=True`.
    #
    # The server exits during startup, so the failure never reaches a request:
    # Modal reports only a cold start that never completes, and the arm looks
    # slow rather than broken. Scoped to the models that need it rather than set
    # globally, because the flag executes code from the weights repository and
    # that should be a deliberate, per-model decision -- THUDM is the official
    # publisher of GLM-4.
    if "glm" in model_id.lower():
        cmd += ["--trust-remote-code"]

    # The API key travels in the environment, never on the command line: argv
    # is echoed into container logs and into `ps`, and a shared secret does not
    # belong in either.
    env = {**os.environ, "VLLM_API_KEY": os.environ["MODAL_VLLM_KEY"]}

    print("launching:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, env=env)


def _endpoint_name(model_id: str) -> str:
    """A stable, greppable function name derived from the model identifier."""
    return "serve_" + model_id.replace("/", "_").replace(".", "_").lower()


# Exactly two servers: standard and reasoning arms share base weights
# (HANDOFF decision 2) and thinking mode is a per-request flag, so the four
# generator entries in configs/models.yaml collapse to two deployments.
#
# These are written out at module scope rather than built in a loop because
# Modal registers a function by importing it by qualname, and a closure has no
# stable one. The alternative, serialized=True, additionally requires the local
# interpreter to match the image's Python version, which couples the deploy to
# whatever happens to be installed on the developer's machine. The model
# IDENTIFIERS still come from the config; only the count is fixed here.
if len(MODELS) != 2:
    raise ValueError(
        f"expected exactly 2 distinct modal_vllm models in configs/models.yaml, "
        f"found {len(MODELS)}: {MODELS}. Add or remove a matching server "
        f"definition in {__file__} if the model roster changes."
    )

MODEL_A, MODEL_B = MODELS

# Horizontal scale. A single A10G saturates at roughly 0.35 requests/second on
# this workload -- raising client concurrency past that only lengthens the vLLM
# queue (p50 latency went 28s -> 84s between concurrency 16 and 32), so the
# lever is more GPUs, not more in-flight requests.
#
# This buys WALL CLOCK, not budget: the GPU-hours are conserved, so N replicas
# finish N times sooner for the same spend. Cost is governed by scope, not by
# this number.
# The workspace GPU tier caps CONCURRENT GPUs at 10, and this file defines TWO
# server functions (one per model), each of which scales independently. The
# default is therefore half the cap, so both models being warm at once still
# fits inside the tier.
#
# A run that drives one generator at a time can safely take the whole budget:
#   RAGROBUST_MAX_CONTAINERS=10 modal deploy deploy/modal_vllm.py
GPU_TIER_LIMIT = int(os.environ.get("RAGROBUST_GPU_TIER_LIMIT", "10"))
MAX_CONTAINERS = int(os.environ.get("RAGROBUST_MAX_CONTAINERS", str(GPU_TIER_LIMIT // 2)))

# Inputs one container accepts before Modal starts another. vLLM batches
# internally, so a container should be kept busy rather than handed one request
# at a time.
#
# LOWERED from 16 to 8 on 25 August. At 16, Modal only starts a second container
# once the first has sixteen requests queued -- so a client running 48 concurrent
# requests across TWO models never justified more than about two containers each,
# and the full run peaked at five of the ten GPUs it is allowed. Measured 0.91
# instances/sec across ~5 containers, i.e. 0.182 each.
#
# At 8, the same client concurrency spreads over roughly twice the containers,
# which buys wall clock without buying compute: GPU-hours are conserved, so the
# run costs about the same and finishes in about half the time. It also lowers
# per-container KV pressure, which matters for GLM-4-9B -- ~18.8 GB of weights on
# a 24 GB A10G -- whose arm produced every one of the six ReadErrors seen at
# concurrency 48.
INPUTS_PER_CONTAINER = int(os.environ.get("RAGROBUST_INPUTS_PER_CONTAINER", "8"))

_FN_KWARGS = dict(
    image=vllm_image,
    gpu=GPU_TYPE,
    max_containers=MAX_CONTAINERS,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("ragrobust-vllm-key"),
    ],
    timeout=60 * 60,
    scaledown_window=SCALEDOWN_WINDOW_S,
)


@app.function(name=_endpoint_name(MODEL_A), **_FN_KWARGS)
@modal.concurrent(max_inputs=INPUTS_PER_CONTAINER)
@modal.web_server(port=8000, startup_timeout=STARTUP_TIMEOUT_S)
def serve_a() -> None:
    _serve(MODEL_A)


@app.function(name=_endpoint_name(MODEL_B), **_FN_KWARGS)
@modal.concurrent(max_inputs=INPUTS_PER_CONTAINER)
@modal.web_server(port=8000, startup_timeout=STARTUP_TIMEOUT_S)
def serve_b() -> None:
    _serve(MODEL_B)


@app.local_entrypoint()
def main() -> None:
    """Print what would be served, without starting a GPU."""
    print(f"GPU:          {GPU_TYPE} x up to {MAX_CONTAINERS} containers")
    print(f"inputs/ctr:   {INPUTS_PER_CONTAINER}")
    print(f"vLLM:         {VLLM_VERSION}")
    print(f"max_model_len {MAX_MODEL_LEN}")
    print("models read from configs/models.yaml:")
    for model in MODELS:
        print(f"  - {model}")
    if not MODELS:
        print("  (none: no generator in configs/models.yaml uses provider modal_vllm)")
