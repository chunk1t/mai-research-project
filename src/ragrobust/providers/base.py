"""Provider abstraction.

Every model call in the project goes through this interface, which keeps the
model allocation a configuration concern rather than a code concern. Swapping a
generator is a YAML edit.

Two properties matter for this experiment specifically.

Determinism. P1 Section 5.6.5 holds decoding parameters constant, so a request
carries an explicit seed and temperature and the cache key covers both. Re-running
an unchanged configuration must not issue new calls.

Family tagging. Every response records the model family that produced it, so the
anti-circularity control of P1 Section 5.9 can be audited after the fact rather
than merely asserted in the write-up.
"""

from __future__ import annotations

import abc
import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    system: str | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    seed: int | None = None
    # Enables a reasoning-trained model's thinking mode where supported.
    thinking: bool = False
    stop: tuple[str, ...] = ()

    def cache_key_fields(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "system": self.system,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
            "thinking": self.thinking,
            "stop": list(self.stop),
        }


@dataclass
class GenerationResponse:
    text: str
    model: str
    family: str
    # Reasoning trace, kept separate from `text`. P1 Section 5.6.4 requires that
    # only the final answer is scored and the trace never enters the metrics.
    reasoning_trace: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class RateLimiter:
    """Token-bucket limiter with an optional hard daily cap.

    The daily cap exists because Google AI Studio free-tier quotas are enforced
    per day, not merely per minute. Hitting the cap should surface as an explicit
    error rather than a wall of opaque 429s partway through a run.
    """

    def __init__(self, max_concurrency: int, requests_per_day: int | None = None):
        self._sem = asyncio.Semaphore(max_concurrency)
        self._requests_per_day = requests_per_day
        self._day_start = time.time()
        self._count_today = 0

    async def __aenter__(self) -> RateLimiter:
        if self._requests_per_day is not None:
            if time.time() - self._day_start >= 86_400:
                self._day_start = time.time()
                self._count_today = 0
            if self._count_today >= self._requests_per_day:
                raise DailyQuotaExceeded(
                    f"daily cap of {self._requests_per_day} requests reached"
                )
            self._count_today += 1
        await self._sem.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._sem.release()


class DailyQuotaExceeded(RuntimeError):
    """Raised when a provider's daily request cap is exhausted."""


class LLMProvider(abc.ABC):
    """Base class for all model providers."""

    def __init__(self, name: str, model: str, family: str, limiter: RateLimiter):
        self.name = name
        self.model = model
        self.family = family
        self.limiter = limiter

    @abc.abstractmethod
    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        """Issue a single call. Implementations need not handle retries."""

    async def generate(self, req: GenerationRequest) -> GenerationResponse:
        start = time.perf_counter()
        async with self.limiter:
            try:
                resp = await self._generate(req)
            except DailyQuotaExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced on the response
                return GenerationResponse(
                    text="",
                    model=self.model,
                    family=self.family,
                    latency_s=time.perf_counter() - start,
                    error=f"{type(exc).__name__}: {exc}",
                )
        resp.latency_s = time.perf_counter() - start
        return resp
