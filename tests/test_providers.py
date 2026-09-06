"""Provider-layer tests, asserted against values worked out by hand.

Every expected payload and every expected number below was written before the
implementation was consulted, so these tests check the code rather than restate
it. No API keys and no network: each provider is driven through an httpx mock
transport, and the rate limiter and cache are exercised directly.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragrobust.cache import ResponseCache
from ragrobust.providers._common import (
    PermanentProviderError,
    TransientProviderError,
    backoff_delay,
    classify_status,
    expand_env,
    split_reasoning,
    with_retries,
)
from ragrobust.providers.anthropic import AnthropicProvider
from ragrobust.providers.base import DailyQuotaExceeded, GenerationRequest, RateLimiter
from ragrobust.providers.cached import CachedProvider
from ragrobust.providers.factory import (
    ConfigError,
    assert_disjoint_families,
    build_all,
    load_config,
)
from ragrobust.providers import replay as replay_module
from ragrobust.providers.google import GoogleProvider
from ragrobust.providers.openai_compat import OpenAICompatProvider
from ragrobust.providers.replay import CacheMiss, ReplayProvider

FAKE_ENV = {
    "MODAL_VLLM_URL": "https://example-modal.run",
    # One endpoint per model: Modal serves a single model per web endpoint.
    "MODAL_VLLM_URL_QWEN": "https://qwen-modal.run",
    "MODAL_VLLM_URL_GLM": "https://glm-modal.run",
    "MODAL_VLLM_KEY": "modal-key",
    "GOOGLE_API_KEY": "google-key",
    "ANTHROPIC_API_KEY": "anthropic-key",
}


async def _no_sleep(_: float) -> None:
    """Injected in place of asyncio.sleep so retry tests run instantly."""
    return None


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def json_route(body: dict, status: int = 200):
    """A handler that always returns `body`, recording every request it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)

    handler.seen = seen  # type: ignore[attr-defined]
    return handler


def limiter() -> RateLimiter:
    return RateLimiter(max_concurrency=2)


# --------------------------------------------------------------------------
# Reasoning-trace separation (P1 Section 5.6.4)
#
# Only the final answer may be scored. If any of these leak trace text into
# `text`, a verbose reasoning model would be scored on its trace and the
# comparison between pipeline classes would be confounded by verbosity.
# --------------------------------------------------------------------------


def test_server_side_reasoning_field_wins():
    answer, trace = split_reasoning("Paris.", reasoning_content="The capital is Paris.")
    assert answer == "Paris."
    assert trace == "The capital is Paris."


def test_inline_think_block_is_stripped_from_the_answer():
    raw = "<think>The passage says 1889.</think>\nFinal answer: 1889"
    answer, trace = split_reasoning(raw)
    assert answer == "Final answer: 1889"
    assert trace == "The passage says 1889."


def test_unclosed_opening_tag_form_splits_on_the_closing_tag():
    # vLLM chat templates often open the block themselves, so the completion
    # carries only the closing tag.
    raw = "Weighing the two passages.</think>Final answer: insufficient evidence"
    answer, trace = split_reasoning(raw)
    assert answer == "Final answer: insufficient evidence"
    assert trace == "Weighing the two passages."


def test_plain_answer_has_no_trace():
    answer, trace = split_reasoning("  1889  ")
    assert answer == "1889"
    assert trace is None


# --------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------


def test_backoff_is_exponential_then_capped():
    # base 0.5, jitter pinned to 1.0 so the ceiling is returned exactly:
    # attempt 1 -> 0.5 * 2^0 = 0.5, attempt 3 -> 0.5 * 2^2 = 2.0,
    # attempt 10 -> 0.5 * 2^9 = 256.0, clamped to the 30.0 cap.
    one = lambda: 1.0  # noqa: E731
    assert backoff_delay(1, base=0.5, cap=30.0, jitter=one) == 0.5
    assert backoff_delay(3, base=0.5, cap=30.0, jitter=one) == 2.0
    assert backoff_delay(10, base=0.5, cap=30.0, jitter=one) == 30.0
    # Full jitter scales the ceiling: half of attempt 3's 2.0 is 1.0.
    assert backoff_delay(3, base=0.5, cap=30.0, jitter=lambda: 0.5) == 1.0


def test_status_classification_splits_transient_from_permanent():
    assert classify_status(200, "") is None
    assert isinstance(classify_status(429, "slow down"), TransientProviderError)
    assert isinstance(classify_status(503, ""), TransientProviderError)
    # A malformed request fails identically on every attempt, so retrying it
    # only burns quota.
    assert isinstance(classify_status(400, "bad"), PermanentProviderError)
    assert isinstance(classify_status(401, ""), PermanentProviderError)


async def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise TransientProviderError("429", status=429)
        return "ok"

    async def no_sleep(_: float) -> None:
        return None

    got = await with_retries(flaky, max_attempts=4, sleep=no_sleep)
    assert got == "ok"
    assert calls["n"] == 3  # two failures, third attempt succeeds


async def test_permanent_failure_is_never_retried():
    calls = {"n": 0}

    async def broken():
        calls["n"] += 1
        raise PermanentProviderError("400", status=400)

    with pytest.raises(PermanentProviderError):
        await with_retries(broken, max_attempts=4)
    assert calls["n"] == 1


async def test_retries_are_bounded():
    calls = {"n": 0}

    async def always_429():
        calls["n"] += 1
        raise TransientProviderError("429", status=429)

    async def no_sleep(_: float) -> None:
        return None

    with pytest.raises(TransientProviderError):
        await with_retries(always_429, max_attempts=3, sleep=no_sleep)
    assert calls["n"] == 3


def test_expand_env_requires_the_variable_to_be_set():
    assert expand_env("${A}/v1", {"A": "https://x"}) == "https://x/v1"
    # An empty base URL would produce a stream of confusing connection errors
    # partway into a run, so it must fail at construction instead.
    with pytest.raises(KeyError, match="MISSING"):
        expand_env("${MISSING}/v1", {})


# --------------------------------------------------------------------------
# OpenAI-compatible provider (Modal / vLLM) — the evaluated generators
# --------------------------------------------------------------------------

VLLM_BODY = {
    "model": "Qwen/Qwen3-8B",
    "choices": [
        {
            "message": {"content": "Final answer: 1889", "reasoning_content": "It says 1889."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 120, "completion_tokens": 30},
}


def vllm_provider(handler, **kw) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        name="modal_vllm",
        model="Qwen/Qwen3-8B",
        family="alibaba",
        limiter=limiter(),
        base_url="https://example-modal.run",
        api_key="modal-key",
        client=mock_client(handler),
        **kw,
    )


def test_vllm_payload_matches_the_expected_shape():
    p = vllm_provider(json_route(VLLM_BODY))
    req = GenerationRequest(
        prompt="In what year?",
        system="You are precise.",
        max_tokens=256,
        temperature=0.0,
        seed=20260721,
        thinking=True,
        stop=("</answer>",),
    )
    assert p.build_payload(req) == {
        "model": "Qwen/Qwen3-8B",
        "messages": [
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "In what year?"},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "seed": 20260721,
        "stop": ["</answer>"],
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_thinking_toggle_travels_in_chat_template_kwargs():
    # standard_1 and reasoning_1 are the same weights; this flag is the only
    # thing separating them (P1 Section 5.6.2).
    p = vllm_provider(json_route(VLLM_BODY))
    off = p.build_payload(GenerationRequest(prompt="q", thinking=False))
    on = p.build_payload(GenerationRequest(prompt="q", thinking=True))
    assert off["chat_template_kwargs"] == {"enable_thinking": False}
    assert on["chat_template_kwargs"] == {"enable_thinking": True}


def test_non_reasoning_checkpoint_omits_the_toggle_entirely():
    # GLM-4 is prompted for chain of thought instead, so sending the Qwen3
    # template flag to it would be meaningless at best.
    p = vllm_provider(json_route(VLLM_BODY), thinking_supported=False)
    assert "chat_template_kwargs" not in p.build_payload(
        GenerationRequest(prompt="q", thinking=True)
    )


async def test_vllm_response_separates_trace_and_counts_tokens():
    p = vllm_provider(json_route(VLLM_BODY))
    r = await p.generate(GenerationRequest(prompt="q"))
    assert r.ok
    assert r.text == "Final answer: 1889"
    assert r.reasoning_trace == "It says 1889."
    assert r.prompt_tokens == 120
    assert r.completion_tokens == 30
    assert r.family == "alibaba"  # family tagging for the P1 5.9 post-hoc audit
    assert r.meta["finish_reason"] == "stop"


async def test_vllm_handles_inline_think_tags_when_no_parser_is_configured():
    body = {
        "choices": [{"message": {"content": "<think>weigh</think>1889"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4},
    }
    p = vllm_provider(json_route(body))
    r = await p.generate(GenerationRequest(prompt="q"))
    assert r.text == "1889"
    assert r.reasoning_trace == "weigh"


async def test_vllm_sends_the_bearer_token():
    handler = json_route(VLLM_BODY)
    p = vllm_provider(handler)
    await p.generate(GenerationRequest(prompt="q"))
    request = handler.seen[0]
    assert request.headers["Authorization"] == "Bearer modal-key"
    assert str(request.url).endswith("/v1/chat/completions")


async def test_rate_limit_is_retried_and_eventually_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json=VLLM_BODY)

    p = vllm_provider(handler, retry_kwargs={"sleep": _no_sleep})
    r = await p.generate(GenerationRequest(prompt="q"))
    assert r.ok and r.text == "Final answer: 1889"
    assert calls["n"] == 2


async def test_permanent_http_error_surfaces_on_the_response_not_as_an_exception():
    # `generate` must never raise for a call-level failure: the runner records
    # the error and moves on, and `ResponseCache` refuses to store it.
    p = vllm_provider(json_route({"error": "bad request"}, status=400))
    r = await p.generate(GenerationRequest(prompt="q"))
    assert not r.ok
    assert "HTTP 400" in (r.error or "")
    assert r.text == ""


async def test_empty_choices_is_an_error_not_an_empty_answer():
    # An empty string would score as an incorrect answer, indistinguishable
    # from a genuine wrong answer in the metrics.
    p = vllm_provider(json_route({"choices": [], "usage": {}}))
    r = await p.generate(GenerationRequest(prompt="q"))
    assert not r.ok
    assert "no choices" in (r.error or "")


# --------------------------------------------------------------------------
# Google provider — case generator and RAGAS judge
# --------------------------------------------------------------------------

GEMINI_BODY = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {"text": "Deliberating.", "thought": True},
                    {"text": "The tower was completed in 1887."},
                ]
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 88, "candidatesTokenCount": 12},
}


def google_provider(handler, **kw) -> GoogleProvider:
    return GoogleProvider(
        name="google_aistudio",
        model="gemini-3-flash",
        family="google",
        limiter=limiter(),
        api_key="google-key",
        client=mock_client(handler),
        **kw,
    )


def test_google_payload_matches_the_expected_shape():
    p = google_provider(json_route(GEMINI_BODY))
    req = GenerationRequest(
        prompt="Rewrite the passage.",
        system="You build test data.",
        max_tokens=512,
        temperature=0.0,
        seed=7,
        stop=("END",),
    )
    assert p.build_payload(req) == {
        "contents": [{"role": "user", "parts": [{"text": "Rewrite the passage."}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "stopSequences": ["END"],
            "seed": 7,
        },
        "systemInstruction": {"parts": [{"text": "You build test data."}]},
    }


async def test_google_routes_thought_parts_to_the_trace():
    p = google_provider(json_route(GEMINI_BODY))
    r = await p.generate(GenerationRequest(prompt="q"))
    assert r.text == "The tower was completed in 1887."
    assert r.reasoning_trace == "Deliberating."
    assert r.prompt_tokens == 88 and r.completion_tokens == 12


async def test_google_blocked_prompt_is_an_error():
    # A blocked prompt returns HTTP 200 with no candidates. Silently accepting
    # it would inject an empty contradiction into the conflict testbed.
    p = google_provider(json_route({"promptFeedback": {"blockReason": "SAFETY"}}))
    r = await p.generate(GenerationRequest(prompt="q"))
    assert not r.ok and "SAFETY" in (r.error or "")


async def test_google_key_travels_in_a_header_not_the_url():
    handler = json_route(GEMINI_BODY)
    p = google_provider(handler)
    await p.generate(GenerationRequest(prompt="q"))
    request = handler.seen[0]
    assert request.headers["x-goog-api-key"] == "google-key"
    assert "google-key" not in str(request.url)
    assert str(request.url).endswith("/models/gemini-3-flash:generateContent")


async def test_daily_quota_exhaustion_is_raised_not_swallowed():
    # The free tier is capped per day. This must stop the run loudly rather
    # than degrade into a wall of opaque 429s (HANDOFF Section 2).
    p = GoogleProvider(
        name="google_aistudio",
        model="gemini-3-flash",
        family="google",
        limiter=RateLimiter(max_concurrency=2, requests_per_day=1),
        api_key="google-key",
        client=mock_client(json_route(GEMINI_BODY)),
    )
    assert (await p.generate(GenerationRequest(prompt="q"))).ok
    with pytest.raises(DailyQuotaExceeded):
        await p.generate(GenerationRequest(prompt="q2"))


# --------------------------------------------------------------------------
# Anthropic provider — the CRS judge
# --------------------------------------------------------------------------

CLAUDE_BODY = {
    "model": "claude-haiku-4-5",
    "content": [
        {"type": "thinking", "thinking": "Both positions are stated."},
        {"type": "text", "text": "3"},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 640, "output_tokens": 8},
}


def anthropic_provider(handler, **kw) -> AnthropicProvider:
    return AnthropicProvider(
        name="anthropic",
        model="claude-haiku-4-5",
        family="anthropic",
        limiter=limiter(),
        api_key="anthropic-key",
        client=mock_client(handler),
        **kw,
    )


def test_anthropic_payload_matches_the_expected_shape():
    p = anthropic_provider(json_route(CLAUDE_BODY))
    req = GenerationRequest(
        prompt="Score this response.",
        system="You are a strict rubric judge.",
        max_tokens=1024,
        temperature=0.0,
        seed=20260721,
        stop=("</score>",),
    )
    payload = p.build_payload(req)
    assert payload == {
        "model": "claude-haiku-4-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Score this response."}],
        "system": "You are a strict rubric judge.",
        "stop_sequences": ["</score>"],
        "temperature": 0.0,
    }
    # The Messages API has no seed parameter; determinism rests on temperature
    # zero alone, which Chapter 6 must state rather than gloss.
    assert "seed" not in payload


def test_anthropic_omits_temperature_for_models_that_reject_it():
    # Claude 4.6 and later reject sampling parameters. Keeping this a
    # constructor flag means a judge swap stays a models.yaml edit.
    p = anthropic_provider(json_route(CLAUDE_BODY), supports_temperature=False)
    assert "temperature" not in p.build_payload(GenerationRequest(prompt="q"))


def test_anthropic_thinking_budget_is_clamped_below_max_tokens():
    # A budget at or above max_tokens leaves no room for the answer itself.
    p = anthropic_provider(json_route(CLAUDE_BODY), thinking_budget_tokens=8000)
    payload = p.build_payload(GenerationRequest(prompt="q", max_tokens=2048, thinking=True))
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1536}  # 2048 - 512
    assert "temperature" not in payload  # thinking requires default sampling


def test_anthropic_adaptive_thinking_style_for_newer_models():
    p = anthropic_provider(json_route(CLAUDE_BODY), thinking_style="adaptive")
    payload = p.build_payload(GenerationRequest(prompt="q", thinking=True))
    assert payload["thinking"] == {"type": "adaptive"}


async def test_anthropic_separates_thinking_blocks_from_the_score():
    p = anthropic_provider(json_route(CLAUDE_BODY))
    r = await p.generate(GenerationRequest(prompt="q"))
    assert r.text == "3"  # the rubric score, uncontaminated by the trace
    assert r.reasoning_trace == "Both positions are stated."
    assert r.prompt_tokens == 640 and r.completion_tokens == 8


async def test_anthropic_sends_the_required_auth_and_version_headers():
    handler = json_route(CLAUDE_BODY)
    p = anthropic_provider(handler)
    await p.generate(GenerationRequest(prompt="q"))
    request = handler.seen[0]
    assert request.headers["x-api-key"] == "anthropic-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert str(request.url).endswith("/v1/messages")


# --------------------------------------------------------------------------
# Cache wrapper — resumability is a correctness property (P1 Section 5.8)
# --------------------------------------------------------------------------


async def test_identical_request_is_issued_once(tmp_path):
    handler = json_route(VLLM_BODY)
    cache = ResponseCache(tmp_path / "c.sqlite")
    p = CachedProvider(vllm_provider(handler), cache)
    req = GenerationRequest(prompt="q", seed=1)

    first = await p.generate(req)
    second = await p.generate(req)

    assert len(handler.seen) == 1  # the second call never reached the network
    assert first.text == second.text == "Final answer: 1889"
    assert first.cached is False and second.cached is True
    cache.close()


async def test_a_changed_decoding_parameter_is_a_different_key(tmp_path):
    handler = json_route(VLLM_BODY)
    cache = ResponseCache(tmp_path / "c.sqlite")
    p = CachedProvider(vllm_provider(handler), cache)

    await p.generate(GenerationRequest(prompt="q", seed=1))
    await p.generate(GenerationRequest(prompt="q", seed=2))

    # A stale result must never be reused after a configuration edit.
    assert len(handler.seen) == 2


async def test_failures_are_never_cached(tmp_path):
    handler = json_route({"error": "boom"}, status=500)
    cache = ResponseCache(tmp_path / "c.sqlite")
    p = CachedProvider(vllm_provider(handler, retry_kwargs={"sleep": _no_sleep}), cache)
    req = GenerationRequest(prompt="q")

    first = await p.generate(req)
    assert not first.ok
    assert cache.stats()["n_cached"] == 0
    after_first = len(handler.seen)

    # A transient outage must not become a permanent empty answer on resume:
    # the second attempt has to reach the network again rather than read a
    # stored failure back out of the cache.
    second = await p.generate(req)
    assert not second.ok
    assert len(handler.seen) > after_first
    cache.close()


# --------------------------------------------------------------------------
# Factory and the anti-circularity control (P1 Section 5.9)
# --------------------------------------------------------------------------


def test_factory_builds_every_role_from_the_real_config():
    cfg = load_config("configs/models.yaml")
    providers = build_all(cfg, env=FAKE_ENV, client=mock_client(json_route(VLLM_BODY)))

    assert set(providers) >= {
        "standard_1",
        "standard_2",
        "reasoning_1",
        "reasoning_2",
        "case_generator",
        "crs_judge",
    }
    assert providers["standard_1"].family == "alibaba"
    assert providers["standard_2"].family == "zhipu"
    assert providers["case_generator"].family == "google"
    assert providers["crs_judge"].family == "anthropic"
    # Model choice stays in config, never in code.
    assert providers["reasoning_1"].model == cfg["generators"]["reasoning_1"]["model"]


def test_factory_wires_the_thinking_toggle_from_config():
    cfg = load_config("configs/models.yaml")
    providers = build_all(cfg, env=FAKE_ENV, client=mock_client(json_route(VLLM_BODY)))
    # Toggle support is a property of the checkpoint, so BOTH Qwen3 arms must
    # declare it: the standard arm has to actively send enable_thinking=false
    # because Qwen3 reasons by default. Reading the arm's own `thinking` flag
    # here would leave standard_1 silently reasoning and make it identical to
    # reasoning_1, voiding the comparison the study exists to make.
    assert providers["reasoning_1"].thinking_supported is True
    assert providers["standard_1"].thinking_supported is True
    assert providers["standard_2"].thinking_supported is False  # GLM-4, prompt-based
    assert providers["reasoning_2"].cot_prompt is True


def test_missing_endpoint_variable_fails_at_construction():
    cfg = load_config("configs/models.yaml")
    env = dict(FAKE_ENV)
    del env["MODAL_VLLM_URL_QWEN"]
    with pytest.raises(KeyError, match="MODAL_VLLM_URL_QWEN"):
        build_all(cfg, env=env, client=mock_client(json_route(VLLM_BODY)))


def test_shipped_config_satisfies_anti_circularity():
    assert_disjoint_families(load_config("configs/models.yaml")) is None


def test_judge_sharing_a_family_with_the_case_generator_is_rejected():
    # The exact scenario P1 Section 5.9 names: a judge rewarding the failure
    # modes its sibling case generator produced.
    cfg = {
        "generators": {"g1": {"family": "alibaba"}},
        "case_generator": {"family": "anthropic"},
        "crs_judge": {"family": "anthropic"},
    }
    with pytest.raises(ConfigError, match="anti-circularity"):
        assert_disjoint_families(cfg)


def test_judge_sharing_a_family_with_an_evaluated_generator_is_rejected():
    cfg = {
        "generators": {"g1": {"family": "anthropic"}},
        "case_generator": {"family": "google"},
        "crs_judge": {"family": "anthropic"},
    }
    with pytest.raises(ConfigError, match="anti-circularity"):
        assert_disjoint_families(cfg)


def test_ragas_baseline_sharing_a_family_with_an_evaluated_generator_is_rejected():
    # RAGAS is the established baseline the purpose-built metrics are measured
    # against (P1 Section 5.4.3), so it must not assess its own family.
    cfg = {
        "generators": {"g1": {"family": "google"}},
        "case_generator": {"family": "anthropic"},
        "crs_judge": {"family": "openai"},
        "ragas_judge": {"family": "google"},
    }
    with pytest.raises(ConfigError, match="RAGAS judge shares"):
        assert_disjoint_families(cfg)


async def _no_sleep(_: float) -> None:
    return None


# --------------------------------------------------------------------------
# Redirect handling: Modal answers a slow request with 303, not an error
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_providers_follow_redirects():
    """A cold Modal container returns 303 See Other pointing at a poll URL.

    httpx does not follow redirects by default, so a provider that leaves the
    default in place turns every cold-start request into
    PermanentProviderError("HTTP 303") and records the instance as failed. That
    is not hypothetical: it failed all six instances of the first smoke run.

    Asserted on the constructed client rather than by mocking a redirect,
    because the defect is the constructor default, not the redirect logic.
    """
    from ragrobust.providers.anthropic import AnthropicProvider
    from ragrobust.providers.google import GoogleProvider
    from ragrobust.providers.openai_compat import OpenAICompatProvider

    built = [
        OpenAICompatProvider(
            name="modal_vllm", model="Qwen/Qwen3-8B", family="alibaba",
            limiter=RateLimiter(max_concurrency=1),
            base_url="https://example.invalid", api_key="k",
        ),
        GoogleProvider(
            name="google_aistudio", model="gemini-3.6-flash", family="google",
            limiter=RateLimiter(max_concurrency=1), api_key="k",
        ),
        AnthropicProvider(
            name="anthropic", model="claude-haiku-4-5-20251001", family="anthropic",
            limiter=RateLimiter(max_concurrency=1), api_key="k",
        ),
    ]
    for provider in built:
        await provider.open()
        try:
            assert provider._client.follow_redirects is True, (
                f"{provider.name} would fail a cold start with HTTP 303"
            )
        finally:
            await provider.close()


def test_generator_base_url_overrides_the_provider_default():
    """Modal serves one model per endpoint, so each arm needs its own URL.

    A single provider-level base_url sent every GLM request to the Qwen server,
    which answered HTTP 404 "The model THUDM/glm-4-9b-chat does not exist" for
    six of the twelve configurations.
    """
    from ragrobust.providers.factory import build_provider

    cfg = {
        "providers": {"modal_vllm": {"kind": "openai_compat",
                                     "base_url": "${QWEN_URL}",
                                     "api_key_env": "K", "max_concurrency": 2}},
    }
    env = {"QWEN_URL": "https://qwen.example", "GLM_URL": "https://glm.example", "K": "k"}

    qwen = build_provider("standard_1", {"provider": "modal_vllm", "model": "Qwen/Qwen3-8B",
                                         "family": "alibaba"}, cfg, env=env)
    glm = build_provider("standard_2", {"provider": "modal_vllm", "model": "THUDM/glm-4-9b-chat",
                                        "family": "zhipu", "base_url": "${GLM_URL}"}, cfg, env=env)
    assert qwen.base_url == "https://qwen.example"
    assert glm.base_url == "https://glm.example", "GLM arm must not point at the Qwen server"


def test_shipped_config_gives_each_model_its_own_endpoint():
    """Two distinct models must never resolve to one URL."""
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "models.yaml").read_text())
    by_model: dict[str, set[str]] = {}
    for g in cfg["generators"].values():
        by_model.setdefault(g["model"], set()).add(g.get("base_url"))
    urls = {next(iter(v)) for v in by_model.values()}
    assert len(urls) == len(by_model), f"models share an endpoint: {by_model}"


# --------------------------------------------------------------------------
# Offline replay (scripts/demo.py)
#
# The demonstration re-executes finished configurations from the run's own
# response cache. Two properties have to hold or a demonstration proves
# nothing: replay must address exactly the entries the run wrote, and it must
# be unable to reach a model, so an apparently offline run cannot silently be
# fetching answers.
# --------------------------------------------------------------------------


def replay_provider(cache, model: str = "Qwen/Qwen3-8B", family: str = "alibaba"):
    return ReplayProvider(cache, name="replay", model=model, family=family)


async def test_replay_reads_back_exactly_what_the_run_wrote(tmp_path):
    """Replay must agree with the provider that populated the cache.

    Written as a round trip rather than against a literal digest, because the
    property that matters is agreement between the writer and the reader: the
    key covers the model, the family and every decoding parameter, so a replay
    arm that computed it even slightly differently would miss all 32,355
    stored entries while looking correct in isolation.
    """
    handler = json_route(VLLM_BODY)
    cache = ResponseCache(tmp_path / "c.sqlite")
    writer = CachedProvider(vllm_provider(handler), cache)
    req = GenerationRequest(prompt="In what year?", seed=20260721, max_tokens=6144)

    live = await writer.generate(req)
    assert len(handler.seen) == 1

    reader = replay_provider(cache)
    assert reader.key_for(req) == writer.key_for(req)

    replayed = await reader.generate(req)
    assert replayed.text == live.text == "Final answer: 1889"
    assert replayed.reasoning_trace == "It says 1889."
    assert replayed.cached is True
    assert (reader.n_hits, reader.n_misses) == (1, 0)
    # Still one. Nothing about the replay touched the transport.
    assert len(handler.seen) == 1
    cache.close()


async def test_a_miss_raises_instead_of_returning_an_empty_answer(tmp_path):
    """A miss must be loud.

    `LLMProvider.generate` converts an exception into a response carrying
    `error` and an empty `text`. That is right for a live run, where the
    instance is simply retried on resume, and wrong here: an empty answer is
    scored as neither a refusal nor a correct answer, so a demonstration would
    appear to run while quietly reporting failures it invented.
    """
    handler = json_route(VLLM_BODY)
    cache = ResponseCache(tmp_path / "c.sqlite")
    writer = CachedProvider(vllm_provider(handler), cache)
    await writer.generate(GenerationRequest(prompt="q", seed=1, temperature=0.0))

    reader = replay_provider(cache)
    # Same prompt, one decoding parameter changed. The key covers temperature,
    # so this is a different request and must not be served the stored answer.
    with pytest.raises(CacheMiss) as excinfo:
        await reader.generate(GenerationRequest(prompt="q", seed=1, temperature=0.7))

    assert excinfo.value.model == "Qwen/Qwen3-8B"
    assert excinfo.value.family == "alibaba"
    assert (reader.n_hits, reader.n_misses) == (0, 1)
    cache.close()


async def test_bypassing_generate_still_cannot_reach_a_model(tmp_path):
    """`_generate` is the abstract entry point; it must not be a back door."""
    cache = ResponseCache(tmp_path / "c.sqlite")
    reader = replay_provider(cache)
    with pytest.raises(CacheMiss):
        await reader._generate(GenerationRequest(prompt="q"))
    cache.close()


def test_replay_holds_no_transport_at_all():
    """Offline is structural, not a promise.

    Asserted over the module's own imports rather than over an instance,
    because an instance can be inspected only in the states a test happens to
    construct. If a future edit imports a client here, this fails before any
    demonstration is given on stage with a live endpoint in the loop.
    """
    import ast
    from pathlib import Path

    src = Path(replay_module.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    networking = {"httpx", "requests", "aiohttp", "urllib", "urllib3", "http",
                  "socket", "ftplib", "telnetlib", "subprocess", "os"}
    assert imported & networking == set(), f"replay imports {sorted(imported & networking)}"
    # `__future__` and `typing` are the whole absolute-import list; everything
    # else this module needs is a relative import inside the package.
    assert imported == {"__future__", "typing"}, \
        f"unexpected top-level imports: {sorted(imported)}"


def test_replay_arm_takes_its_model_id_from_the_config(tmp_path):
    """A hard-coded model id would miss every entry the run wrote.

    Expected values read from `configs/models.yaml`: `standard_1` is
    Qwen/Qwen3-8B in the alibaba family, `standard_2` is THUDM/glm-4-9b-chat in
    the zhipu family.
    """
    cfg = load_config("configs/models.yaml")
    cache = ResponseCache(tmp_path / "c.sqlite")

    qwen = ReplayProvider.for_generator(cache, "standard_1", cfg)
    glm = ReplayProvider.for_generator(cache, "standard_2", cfg)

    assert (qwen.model, qwen.family) == ("Qwen/Qwen3-8B", "alibaba")
    assert (glm.model, glm.family) == ("THUDM/glm-4-9b-chat", "zhipu")

    # Same request, two arms: different models must give different keys, or the
    # two halves of the matrix would replay each other's answers.
    req = GenerationRequest(prompt="q", seed=20260721)
    assert qwen.key_for(req) != glm.key_for(req)

    with pytest.raises(KeyError):
        ReplayProvider.for_generator(cache, "no_such_generator", cfg)
    cache.close()
