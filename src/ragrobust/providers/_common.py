"""Shared provider plumbing: retries, reasoning-trace splitting, config expansion.

Three concerns are factored out here because all three concrete providers need
them and a divergence between them would be a silent experimental confound.

Retries. A transient 429 or 503 must not become a permanent empty answer. The
cache never stores failures (`cache.py`), so an un-retried blip would simply be
re-issued on the next resume; retrying in place is cheaper and keeps a long run
moving. Only transient statuses are retried. A 400 or 401 is a bug in the request
and retrying it wastes quota.

Reasoning-trace splitting. P1 Section 5.6.4 requires that only the delimited
final answer is scored and that the reasoning trace "is retained for qualitative
analysis but never enters the quantitative metrics". Enforcing that split at the
provider boundary means no downstream consumer can accidentally score a trace:
`GenerationResponse.text` is always answer-only by construction.

Config expansion. `configs/models.yaml` holds `${VAR}` placeholders so that
endpoints and keys stay out of version control.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from typing import Awaitable, Callable, Mapping, TypeVar

T = TypeVar("T")

# Statuses worth retrying: rate limits, and transient server/gateway failures.
# 408 is included because a request timeout is a network condition, not a bad
# request. Everything else (400, 401, 403, 404, 422) indicates a malformed or
# unauthorised call that will fail identically on every attempt.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})

DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE_S = 0.5
DEFAULT_BACKOFF_CAP_S = 30.0

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Reasoning-trace delimiters emitted by reasoning-trained open models served
# through vLLM. When vLLM runs with a reasoning parser the trace arrives in a
# separate `reasoning_content` field instead, which is preferred; these patterns
# are the fallback for a server configured without one.
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


class TransientProviderError(RuntimeError):
    """A provider failure worth retrying. Carries the HTTP status when known."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class PermanentProviderError(RuntimeError):
    """A provider failure that will recur identically. Never retried."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def classify_status(status: int, body: str) -> Exception | None:
    """Map an HTTP status onto the retry policy. None means success."""
    if 200 <= status < 300:
        return None
    detail = body.strip()[:400]
    if status in RETRYABLE_STATUS:
        return TransientProviderError(f"HTTP {status}: {detail}", status=status)
    return PermanentProviderError(f"HTTP {status}: {detail}", status=status)


def backoff_delay(
    attempt: int,
    *,
    base: float = DEFAULT_BACKOFF_BASE_S,
    cap: float = DEFAULT_BACKOFF_CAP_S,
    jitter: Callable[[], float] = random.random,
) -> float:
    """Exponential backoff with full jitter, for attempt numbers starting at 1.

    Full jitter rather than fixed backoff because the run issues many concurrent
    requests against one endpoint. Without jitter every worker retries on the
    same schedule and the endpoint is hit by a synchronised thundering herd,
    which turns one 429 into a sustained stall.
    """
    if attempt < 1:
        raise ValueError("attempt numbering starts at 1")
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    return ceiling * jitter()


async def with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base: float = DEFAULT_BACKOFF_BASE_S,
    cap: float = DEFAULT_BACKOFF_CAP_S,
    jitter: Callable[[], float] = random.random,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run `call`, retrying only transient failures.

    `sleep` and `jitter` are injectable so tests exercise the retry path without
    real delays or nondeterminism.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await call()
        except TransientProviderError as exc:
            last = exc
            if attempt == max_attempts:
                break
            await sleep(backoff_delay(attempt, base=base, cap=cap, jitter=jitter))
    assert last is not None
    raise last


def split_reasoning(text: str, reasoning_content: str | None = None) -> tuple[str, str | None]:
    """Separate the scoreable answer from the reasoning trace.

    Returns `(answer, trace)`. Three shapes are handled, in priority order.

    A server-side reasoning parser already split the trace into its own field,
    in which case the content is answer-only and is taken verbatim.

    The model emitted a full `<think>...</think>` block inline.

    The model emitted only a closing `</think>`, which happens when the chat
    template opens the block itself so the opening tag never appears in the
    completion. Everything before the tag is trace.

    Anything else is treated as answer-only with no trace, which is the correct
    reading for a non-reasoning generator.
    """
    if reasoning_content:
        return text.strip(), reasoning_content.strip()

    match = _THINK_BLOCK.search(text)
    if match:
        trace = match.group(1).strip()
        answer = (text[: match.start()] + text[match.end() :]).strip()
        return answer, trace or None

    close = _THINK_CLOSE.search(text)
    if close:
        trace = text[: close.start()].strip()
        answer = text[close.end() :].strip()
        return answer, trace or None

    return text.strip(), None


def expand_env(value: str, env: Mapping[str, str] | None = None) -> str:
    """Substitute `${VAR}` placeholders from the environment.

    Raises on an unset variable rather than substituting an empty string. A
    silently empty base URL produces a stream of confusing connection errors
    partway into a run; failing at construction is far cheaper to diagnose.
    """
    source = os.environ if env is None else env

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in source or not source[name]:
            raise KeyError(f"environment variable {name} is required but unset")
        return source[name]

    return _ENV_PATTERN.sub(_sub, value)
