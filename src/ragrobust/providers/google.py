"""Google AI Studio (Gemini) client.

Carries two low-volume, distinct-family roles: the synthetic case generator that
writes conflict contradictions, and the RAGAS judge for the secondary analysis
(HANDOFF Section 2). It must never carry an evaluated generator, both because the
free-tier daily cap cannot absorb that volume and because P1 Section 5.9 requires
the case generator to sit in a different family from the models under evaluation.

The daily cap is enforced upstream by `RateLimiter`, which raises
`DailyQuotaExceeded` rather than letting a run degrade into a wall of 429s
partway through case generation.
"""

from __future__ import annotations

from typing import Any

import httpx

from ._common import (
    PermanentProviderError,
    classify_status,
    with_retries,
)
from .base import GenerationRequest, GenerationResponse, LLMProvider, RateLimiter


class GoogleProvider(LLMProvider):
    """Gemini `generateContent` client."""

    def __init__(
        self,
        name: str,
        model: str,
        family: str,
        limiter: RateLimiter,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 180.0,
        max_attempts: int = 4,
        retry_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(name=name, model=model, family=family, limiter=limiter)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self._retry_kwargs = dict(retry_kwargs or {})
        self._client = client
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        # The key travels in a header rather than a query string so it cannot
        # leak into request logs or error messages that echo the URL.
        return {"Content-Type": "application/json", "x-goog-api-key": self.api_key}

    def build_payload(self, req: GenerationRequest) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_tokens,
        }
        if req.stop:
            generation_config["stopSequences"] = list(req.stop)
        if req.seed is not None:
            generation_config["seed"] = req.seed

        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": req.prompt}]}],
            "generationConfig": generation_config,
        }
        if req.system:
            payload["systemInstruction"] = {"parts": [{"text": req.system}]}
        return payload

    def parse_response(self, data: dict[str, Any]) -> GenerationResponse:
        """Convert a generateContent body into a GenerationResponse.

        A blocked prompt returns HTTP 200 with no candidates, so an empty
        candidate list is treated as a permanent failure rather than allowed to
        surface as an empty string. An empty answer that scored as incorrect
        would be indistinguishable from a genuine wrong answer in the metrics.
        """
        feedback = data.get("promptFeedback") or {}
        if feedback.get("blockReason"):
            raise PermanentProviderError(f"prompt blocked: {feedback['blockReason']}")

        candidates = data.get("candidates") or []
        if not candidates:
            raise PermanentProviderError("response contained no candidates")

        candidate = candidates[0]
        parts = ((candidate.get("content") or {}).get("parts")) or []

        # Gemini marks reasoning parts with `thought: true`. Keeping them out of
        # `text` enforces P1 Section 5.6.4 at the provider boundary.
        answer_chunks = [p.get("text", "") for p in parts if not p.get("thought")]
        trace_chunks = [p.get("text", "") for p in parts if p.get("thought")]

        answer = "".join(answer_chunks).strip()
        trace = "".join(trace_chunks).strip() or None

        usage = data.get("usageMetadata") or {}
        return GenerationResponse(
            text=answer,
            model=self.model,
            family=self.family,
            reasoning_trace=trace,
            prompt_tokens=int(usage.get("promptTokenCount") or 0),
            completion_tokens=int(usage.get("candidatesTokenCount") or 0),
            meta={
                "finish_reason": candidate.get("finishReason"),
                "provider": self.name,
            },
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._client
        if client is None:
            raise PermanentProviderError("provider used without an http client; call open()")
        resp = await client.post(
            f"{self.base_url}/models/{self.model}:generateContent",
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
