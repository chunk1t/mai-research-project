"""Cache-backed provider wrapper.

P1 Section 5.8 commits to batched, resumable execution: "the experiment is
batched and resumable so that interrupted runs can be continued without
re-running completed cases." With roughly seventy thousand generator calls, an
un-resumable run is not merely inconvenient, it is unaffordable, so resumability
is treated as a correctness property.

Implemented as a wrapper rather than folded into `LLMProvider` for two reasons.
It keeps `cache.py` importing `providers.base` in one direction only, avoiding an
import cycle. And it keeps the abstract provider contract free of storage
concerns, so a provider can be unit tested with no database present.

Wrapping happens above `generate()`, not `_generate()`, so a cache hit skips the
rate limiter entirely. That matters for the Google free tier: re-running an
analysis over already-generated cases must not consume any of the day's quota.
"""

from __future__ import annotations

from ..cache import ResponseCache, cache_key
from .base import GenerationRequest, GenerationResponse, LLMProvider


class CachedProvider(LLMProvider):
    """Wraps a provider so identical requests are issued at most once."""

    def __init__(self, inner: LLMProvider, cache: ResponseCache):
        super().__init__(
            name=inner.name, model=inner.model, family=inner.family, limiter=inner.limiter
        )
        self.inner = inner
        self.cache = cache

    def key_for(self, req: GenerationRequest) -> str:
        return cache_key(self.inner.model, self.inner.family, req)

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        key = self.key_for(req)
        hit = self.cache.get(key)
        if hit is not None:
            return hit

        resp = await self.inner.generate(req)
        # `ResponseCache.put` refuses failures, so a transient error never
        # becomes a permanent empty answer on the next resume.
        self.cache.put(key, resp)
        return resp

    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        # Never reached through `generate`, but required by the abstract base and
        # meaningful if a caller deliberately bypasses the cache.
        return await self.inner._generate(req)

    async def open(self) -> None:
        opener = getattr(self.inner, "open", None)
        if opener is not None:
            await opener()

    async def close(self) -> None:
        closer = getattr(self.inner, "close", None)
        if closer is not None:
            await closer()
