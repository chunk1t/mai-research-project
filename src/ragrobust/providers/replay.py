"""Offline replay provider: serve a finished run from its own response cache.

Every generator call in the experiment was content-addressed into
`data/cache/generators.sqlite` (`cache.py`), so a finished configuration can be
re-executed end to end with the real pipeline objects and no model behind them.
That is what makes a demonstration of this study possible at all: the evaluated
generators are self-hosted on Modal at 7-9B and cost GPU-minutes to wake, and
P1 Section 5.6.5 fixes the decoding parameters, so a live re-run would be both
expensive and, at temperature 0 across a different batch composition, not
guaranteed to reproduce the recorded answer anyway.

Why this is a separate class rather than `CachedProvider` with its inner
provider omitted.

A miss must raise, not fall through. `CachedProvider` treats the cache as an
optimisation and calls the model when it misses; `LLMProvider.generate` then
converts any exception into a `GenerationResponse` carrying `error`, so a cold
Modal container becomes a several-minute hang and an empty answer. In a replay
context both behaviours are wrong: an empty answer is indistinguishable from a
model that declined to answer, and would be scored as a failure. Here a miss is
`CacheMiss`, raised immediately, and the caller decides.

The inability to reach the network is structural, not promised. This module
imports no HTTP client and holds none: there is no transport to disable and no
flag to forget to set. `tests/test_providers.py` asserts that property against
the module's own import list, so it cannot regress into a "mostly offline"
provider later.

The limiter is a no-op token bucket. Nothing is being rate limited, but the
`LLMProvider` contract carries one, and constructing it here keeps a replay
provider substitutable everywhere a real provider is accepted -- including
`make_pipeline`, which needs no change to run offline.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..cache import ResponseCache, cache_key
from .base import GenerationRequest, GenerationResponse, LLMProvider, RateLimiter


class CacheMiss(LookupError):
    """A replayed request has no stored response.

    Carries the key and the model so a miss is diagnosable from the message
    alone: the usual cause is a request that differs from the recorded one in
    some decoding parameter, and the key covers every one of them, so knowing
    which model was asked narrows it immediately.
    """

    def __init__(self, key: str, model: str, family: str):
        super().__init__(
            f"no cached response for key {key[:16]}... (model={model}, family={family}); "
            "the request differs from anything recorded in this run"
        )
        self.key = key
        self.model = model
        self.family = family


class ReplayProvider(LLMProvider):
    """Serves recorded responses only. Raises `CacheMiss` rather than calling out."""

    def __init__(
        self,
        cache: ResponseCache,
        *,
        name: str,
        model: str,
        family: str,
        limiter: RateLimiter | None = None,
    ):
        super().__init__(
            name=name,
            model=model,
            family=family,
            # Generous, because it gates nothing: every call is a local SQLite
            # read. A tight limiter here would only serialise the demo.
            limiter=limiter or RateLimiter(max_concurrency=64),
        )
        self.cache = cache
        self.n_hits = 0
        self.n_misses = 0

    @classmethod
    def for_generator(
        cls,
        cache: ResponseCache,
        generator_key: str,
        models_cfg: Mapping[str, Any],
    ) -> ReplayProvider:
        """Build the replay arm for one generator in `configs/models.yaml`.

        The model id and family are read from the config rather than named here,
        for the same reason the factory does it: swapping a generator must stay
        a configuration edit. It also matters for correctness -- the cache key
        covers the model id, so a hard-coded name that drifted from the config
        would miss every entry the run wrote.
        """
        spec = (models_cfg.get("generators") or {}).get(generator_key)
        if spec is None:
            raise KeyError(f"unknown generator '{generator_key}' in models config")
        return cls(
            cache,
            name=f"replay:{generator_key}",
            model=spec["model"],
            family=spec["family"],
        )

    def key_for(self, req: GenerationRequest) -> str:
        return cache_key(self.model, self.family, req)

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        # Overrides `generate`, not `_generate`, so the base class's
        # exception-to-error-response conversion never runs. A miss has to stay
        # an exception: as a response carrying `error` it would be recorded as a
        # failed instance and quietly change what a demonstration appears to show.
        key = self.key_for(req)
        hit = self.cache.get(key)
        if hit is None:
            self.n_misses += 1
            raise CacheMiss(key, self.model, self.family)
        self.n_hits += 1
        return hit

    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        # Reached only if a caller deliberately bypasses `generate`. It raises
        # rather than falling back, so there is no path through this class that
        # ends at a network call.
        raise CacheMiss(self.key_for(req), self.model, self.family)

    async def open(self) -> None:
        """No-op. Present so replay is substitutable for a real provider."""

    async def close(self) -> None:
        """No-op. The cache is owned by the caller, which closes it."""
