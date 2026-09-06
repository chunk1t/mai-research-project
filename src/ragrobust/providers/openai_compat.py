"""OpenAI-compatible chat completions client, targeting Modal/vLLM.

This carries the entire evaluated-generator workload. P1 Section 5.8 budgets
roughly seventy thousand generator calls, which does not fit hosted API pricing
at the available budget, so all four evaluated generators are self-hosted behind
vLLM's OpenAI-compatible server (HANDOFF Section 2).

Two details matter experimentally.

Thinking mode. P1 Section 5.6.2 distinguishes a reasoning-trained generator from
a chain-of-thought-prompted one. For Qwen3 the reasoning-trained form is selected
at request time by the chat template rather than by a different checkpoint, so
the toggle travels in `chat_template_kwargs`. This is what makes `standard_1` and
`reasoning_1` the same weights with the reasoning variable isolated, per the
locked decision in HANDOFF Section 2.

Determinism. P1 Section 5.6.5 holds decoding parameters constant. Temperature and
seed are sent explicitly on every request rather than relying on server defaults,
because a server-side default change between batches would silently break the
comparability of results collected days apart.
"""

from __future__ import annotations

from typing import Any

import httpx

from ._common import (
    PermanentProviderError,
    classify_status,
    split_reasoning,
    with_retries,
)
from .base import GenerationRequest, GenerationResponse, LLMProvider, RateLimiter


class OpenAICompatProvider(LLMProvider):
    """Chat-completions client for vLLM and any OpenAI-compatible endpoint."""

    def __init__(
        self,
        name: str,
        model: str,
        family: str,
        limiter: RateLimiter,
        *,
        base_url: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 600.0,
        max_attempts: int = 4,
        thinking_supported: bool = True,
        cot_prompt: bool = False,
        extra_body: dict[str, Any] | None = None,
        retry_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(name=name, model=model, family=family, limiter=limiter)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        # A non-reasoning checkpoint rejects the thinking toggle, so the flag is
        # a property of the served model rather than of the request.
        self.thinking_supported = thinking_supported
        # Chain-of-thought prompting is the second reasoning form of P1 5.6.2.
        # It is recorded here so a response can be attributed to the right arm,
        # but the prompt text itself is the pipeline layer's business.
        self.cot_prompt = cot_prompt
        self.extra_body = dict(extra_body or {})
        self._retry_kwargs = dict(retry_kwargs or {})
        self._client = client
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def build_payload(self, req: GenerationRequest) -> dict[str, Any]:
        """Assemble the request body. Split out so tests can assert its shape."""
        messages: list[dict[str, str]] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.stop:
            payload["stop"] = list(req.stop)
        if self.thinking_supported:
            # vLLM forwards this to the model's chat template, which is how
            # Qwen3 selects between its thinking and non-thinking forms.
            payload["chat_template_kwargs"] = {"enable_thinking": req.thinking}
        payload.update(self.extra_body)
        return payload

    def parse_response(self, data: dict[str, Any]) -> GenerationResponse:
        """Convert a chat-completions body into a GenerationResponse.

        The trace is separated here so that `text` is answer-only for every
        downstream consumer (P1 Section 5.6.4).
        """
        choices = data.get("choices") or []
        if not choices:
            raise PermanentProviderError("response contained no choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        # Present when vLLM runs with a reasoning parser; absent otherwise, in
        # which case the trace is still inline in the content as think tags.
        reasoning_content = message.get("reasoning_content")

        answer, trace = split_reasoning(content, reasoning_content)
        usage = data.get("usage") or {}

        return GenerationResponse(
            text=answer,
            model=self.model,
            family=self.family,
            reasoning_trace=trace,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            meta={
                "finish_reason": choices[0].get("finish_reason"),
                "provider": self.name,
                "served_model": data.get("model"),
            },
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise PermanentProviderError("provider used without an http client; call open()")
        resp = await client.post(
            f"{self.base_url}/v1/chat/completions",
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
            # follow_redirects, because Modal answers a request that outruns
            # its synchronous window with 303 See Other pointing at a poll URL.
            # httpx does not follow redirects by default, so without this a cold
            # start surfaces as PermanentProviderError("HTTP 303") and the
            # instance is recorded as failed -- which is exactly what the first
            # smoke run did to all six of its instances.
            self._client = httpx.AsyncClient(follow_redirects=True)
            self._owns_client = True

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None
