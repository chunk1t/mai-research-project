"""Anthropic Messages API client.

Carries exactly one role: the Conflict Resolution Score judge (HANDOFF Section
2). P1 Section 5.9 requires the judge to sit in a different model family from
both the evaluated generators and the case generator, so that "a judge might
reward exactly the failure modes its sibling generator produces" is structurally
prevented rather than merely hoped for. `factory.assert_disjoint_families`
enforces that at construction.

Two API constraints are worth stating because they differ from the other two
providers and are easy to trip over on a model swap.

There is no seed parameter. Determinism therefore rests on temperature zero
alone (P1 Section 5.6.5), which is weaker than the seeded determinism available
from vLLM. The judge is a scoring instrument rather than a system under test, so
this is acceptable, but it should be stated in Chapter 6 rather than glossed.

The request surface changed across model generations. Models from Claude 4.6
onward reject `temperature` and reject `budget_tokens`, using adaptive thinking
instead. Both are therefore configurable rather than hardcoded, so that swapping
the judge model stays a `configs/models.yaml` edit as required by the project
conventions.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx

from ._common import (
    PermanentProviderError,
    classify_status,
    with_retries,
)
from .base import GenerationRequest, GenerationResponse, LLMProvider, RateLimiter

ANTHROPIC_VERSION = "2023-06-01"

ThinkingStyle = Literal["none", "budget", "adaptive"]


class AnthropicProvider(LLMProvider):
    """Messages API client."""

    def __init__(
        self,
        name: str,
        model: str,
        family: str,
        limiter: RateLimiter,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 300.0,
        max_attempts: int = 4,
        supports_temperature: bool = True,
        thinking_style: ThinkingStyle = "budget",
        thinking_budget_tokens: int = 2048,
        retry_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(name=name, model=model, family=family, limiter=limiter)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.supports_temperature = supports_temperature
        self.thinking_style = thinking_style
        self.thinking_budget_tokens = thinking_budget_tokens
        self._retry_kwargs = dict(retry_kwargs or {})
        self._client = client
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        }

    def build_payload(self, req: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": "user", "content": req.prompt}],
        }
        if req.system:
            payload["system"] = req.system
        if req.stop:
            payload["stop_sequences"] = list(req.stop)
        if self.supports_temperature:
            payload["temperature"] = req.temperature

        if req.thinking and self.thinking_style != "none":
            if self.thinking_style == "adaptive":
                payload["thinking"] = {"type": "adaptive"}
            else:
                # A budget must leave room for the answer itself, so it is
                # clamped below max_tokens rather than trusted from config.
                budget = min(self.thinking_budget_tokens, max(req.max_tokens - 512, 1024))
                payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                # Thinking requires unconstrained sampling on the models that
                # still accept a temperature at all.
                payload.pop("temperature", None)
        return payload

    def parse_response(self, data: dict[str, Any]) -> GenerationResponse:
        """Convert a Messages body into a GenerationResponse.

        Content arrives as a list of typed blocks. Thinking blocks are routed to
        `reasoning_trace` and text blocks to `text`, which keeps the trace out of
        every metric by construction (P1 Section 5.6.4).
        """
        blocks = data.get("content")
        if blocks is None:
            raise PermanentProviderError("response contained no content blocks")

        answer_chunks = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        trace_chunks = [b.get("thinking", "") for b in blocks if b.get("type") == "thinking"]

        usage = data.get("usage") or {}
        return GenerationResponse(
            text="".join(answer_chunks).strip(),
            model=self.model,
            family=self.family,
            reasoning_trace="".join(trace_chunks).strip() or None,
            prompt_tokens=int(usage.get("input_tokens") or 0),
            completion_tokens=int(usage.get("output_tokens") or 0),
            meta={
                "stop_reason": data.get("stop_reason"),
                "provider": self.name,
                "served_model": data.get("model"),
            },
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise PermanentProviderError("provider used without an http client; call open()")
        resp = await client.post(
            f"{self.base_url}/v1/messages",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout_s,
        )
        err = classify_status(resp.status_code, resp.text)
        if err is not None:
            raise err
        return resp.json()

    async def _generate(self, req: GenerationRequest) -> GenerationResponse:
        payload = self.build_payload(req)
        data = await with_retries(
            lambda: self._post(payload),
            max_attempts=self.max_attempts,
            **self._retry_kwargs,
        )
        return self.parse_response(data)

    async def open(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=True)
            self._owns_client = True

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
