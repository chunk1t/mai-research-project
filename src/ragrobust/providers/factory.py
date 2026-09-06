"""Build providers from `configs/models.yaml`.

The project convention is that swapping a model is a configuration edit and never
a code edit, so every provider in the experiment is constructed here from the
YAML rather than instantiated at its call site.

The factory also enforces the anti-circularity control of P1 Section 5.9 at
construction time. `tests/test_dataset.py` already audits the config file, but a
test only protects the file as committed; enforcing it in the constructor means a
hand-edited config or an override passed at runtime fails before any API call is
issued rather than after a partial run has been paid for.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from ._common import expand_env
from .anthropic import AnthropicProvider
from .base import LLMProvider, RateLimiter
from .google import GoogleProvider
from .openai_compat import OpenAICompatProvider

# Roles that P1 Section 5.9 requires to be drawn from disjoint model families.
DISJOINT_ROLES = ("evaluated", "case_generator", "crs_judge")


class ConfigError(ValueError):
    """Raised when the model configuration is unusable or unsafe."""


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return cfg


def families_by_role(cfg: Mapping[str, Any]) -> dict[str, set[str]]:
    """Collect the model families used by each role group."""
    generators = cfg.get("generators") or {}
    roles: dict[str, set[str]] = {
        "evaluated": {g["family"] for g in generators.values()},
    }
    for role in ("case_generator", "ragas_judge", "crs_judge", "refusal_classifier"):
        spec = cfg.get(role)
        if spec:
            roles[role] = {spec["family"]}
    return roles


def assert_disjoint_families(cfg: Mapping[str, Any]) -> None:
    """Enforce the anti-circularity control of P1 Section 5.9.

    The case generator, the evaluated generators, and the CRS judge must not
    share a family: a judge could otherwise reward the failure modes its sibling
    generator produces, and a generator could exploit artefacts left by a
    same-family case generator.

    The RAGAS judge is checked against the evaluated generators too. It is the
    established baseline the purpose-built metrics are compared against
    (P1 Section 5.4.3), so sharing a family with the systems it assesses would
    undermine that comparison.

    Deliberately not enforced: the RAGAS judge and the case generator may share
    a family, which the shipped config does (both `google`). P1 Section 5.9
    names three roles, and RAGAS is not among them, so rejecting that pairing
    here would be a stricter rule than the frozen design specifies. It is a real
    residual risk — Gemini writes the contradiction passage and a Gemini-family
    judge then scores faithfulness against a context containing it — and belongs
    in the Chapter 8 limitations rather than in a constructor that would refuse
    to build the configuration the study actually ran.
    """
    roles = families_by_role(cfg)
    for i, a in enumerate(DISJOINT_ROLES):
        for b in DISJOINT_ROLES[i + 1 :]:
            overlap = roles.get(a, set()) & roles.get(b, set())
            if overlap:
                raise ConfigError(
                    f"anti-circularity violation (P1 5.9): roles '{a}' and '{b}' "
                    f"share model family {sorted(overlap)}"
                )
    ragas = roles.get("ragas_judge", set())
    overlap = ragas & roles.get("evaluated", set())
    if overlap:
        raise ConfigError(
            f"RAGAS judge shares model family {sorted(overlap)} with an evaluated "
            "generator, which would compromise the baseline comparison (P1 5.4.3)"
        )

    # The refusal classifier decides whether an evaluated generator refused, so
    # a same-family classifier could read its sibling's phrasing as commitment
    # and bias Refusal F1 in one direction (P1 Sections 5.6.4 and 5.9).
    classifier = roles.get("refusal_classifier", set())
    overlap = classifier & roles.get("evaluated", set())
    if overlap:
        raise ConfigError(
            f"refusal classifier shares model family {sorted(overlap)} with an "
            "evaluated generator, which would bias Refusal F1 (P1 5.6.4, 5.9)"
        )


def _limiter(provider_cfg: Mapping[str, Any]) -> RateLimiter:
    return RateLimiter(
        max_concurrency=int(provider_cfg.get("max_concurrency", 4)),
        requests_per_day=provider_cfg.get("requests_per_day"),
    )


def build_provider(
    role: str,
    spec: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    client: Any | None = None,
    **overrides: Any,
) -> LLMProvider:
    """Construct one provider from its role spec.

    `client` is injectable so the suite exercises every provider against a mock
    transport, with no keys and no network.
    """
    env = os.environ if env is None else env
    providers_cfg = cfg.get("providers") or {}
    provider_name = spec["provider"]
    if provider_name not in providers_cfg:
        raise ConfigError(f"{role}: unknown provider '{provider_name}'")
    pcfg = providers_cfg[provider_name]

    kind = pcfg.get("kind")
    api_key = None
    if pcfg.get("api_key_env"):
        api_key = env.get(pcfg["api_key_env"])

    limiter = _limiter(pcfg)
    common = {
        "name": provider_name,
        "model": spec["model"],
        "family": spec["family"],
        "limiter": limiter,
        "client": client,
    }

    if kind == "openai_compat":
        # A generator may override the provider's base_url. Modal serves ONE
        # model per web endpoint, so the four evaluated arms live behind two
        # different URLs; a single provider-level base_url sent every GLM
        # request to the Qwen server, which answered 404 "model does not exist"
        # for half the experimental matrix.
        base_url = expand_env(str(spec.get("base_url") or pcfg["base_url"]), env)
        return OpenAICompatProvider(
            base_url=base_url,
            api_key=api_key,
            # Whether the CHECKPOINT accepts the toggle, which is not the same
            # as whether this arm wants thinking on. Qwen3 defaults to thinking
            # ON, so the standard arm must actively send enable_thinking=false;
            # reading the arm's own `thinking` flag here would omit the field
            # entirely and leave both arms reasoning, collapsing the standard
            # and reasoning classes into the same system.
            thinking_supported=bool(spec.get("supports_thinking", False)),
            cot_prompt=bool(spec.get("cot_prompt", False)),
            **common,
            **overrides,
        )

    if kind == "google":
        if not api_key:
            raise ConfigError(f"{role}: {pcfg.get('api_key_env')} is required but unset")
        return GoogleProvider(
            api_key=api_key,
            base_url=expand_env(str(pcfg.get("base_url", "")), env)
            or "https://generativelanguage.googleapis.com/v1beta",
            **common,
            **overrides,
        )

    if kind == "anthropic":
        if not api_key:
            raise ConfigError(f"{role}: {pcfg.get('api_key_env')} is required but unset")
        return AnthropicProvider(api_key=api_key, **common, **overrides)

    raise ConfigError(f"{role}: unsupported provider kind '{kind}'")


def build_all(
    cfg: Mapping[str, Any],
    *,
    env: Mapping[str, str] | None = None,
    client: Any | None = None,
    roles: tuple[str, ...] | None = None,
    **overrides: Any,
) -> dict[str, LLMProvider]:
    """Construct every configured provider, keyed by role name.

    Evaluated generators keep their config key (`standard_1`, `reasoning_2`, ...)
    so a result row can be traced back to the exact arm that produced it.
    """
    assert_disjoint_families(cfg)

    out: dict[str, LLMProvider] = {}
    wanted = roles

    for name, spec in (cfg.get("generators") or {}).items():
        if wanted is None or name in wanted:
            out[name] = build_provider(name, spec, cfg, env=env, client=client, **overrides)

    for role in ("case_generator", "ragas_judge", "crs_judge", "refusal_classifier"):
        spec = cfg.get(role)
        if spec and (wanted is None or role in wanted):
            out[role] = build_provider(role, spec, cfg, env=env, client=client, **overrides)

    return out
