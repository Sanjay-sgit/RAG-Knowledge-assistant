"""
Shared Gemini client, a rate limiter, and a sane retry policy.

- The client is created lazily (first use), so importing this package never crashes.
- A shared rate limiter paces EVERY Gemini call (chat + embeddings) under
  GEMINI_RPM requests/minute, so a batch job (the eval dashboard, ingest) can't
  burst past the free tier's per-minute quota in the first place.
- Retries only happen for errors that can succeed on a second try
  (rate limits 429, server errors 5xx, network hiccups), honour Google's own
  `retryDelay` hint when it gives one, and stop after MAX_RETRIES.
  A wrong API key or a wrong model name fails immediately with a clear message.
- If a call hits a DAILY quota limit (can't be waited out - resets at midnight
  Pacific), automatically switches to the next Gemini API key from .env
  (key1, key2, ...) instead of failing every call until the quota resets.
"""

import threading
import time
from collections import deque
from functools import lru_cache

import httpx
from google import genai
from google.genai import errors as genai_errors
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from pro_implementation import config


class MissingAPIKeyError(RuntimeError):
    pass


def _mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 4 else "...****"


# ----------------------------------------------------------------------------
# API key rotation - when one key's free-tier DAILY quota is exhausted (can't be
# waited out; resets at midnight Pacific), automatically move to the next key in
# GEMINI_API_KEY / key1, key2, ... (config.get_api_keys()) instead of failing every
# call until .env is edited by hand. A key that's been marked exhausted
# is skipped for the rest of this process, not just retried once.
# ----------------------------------------------------------------------------


class _KeyRotator:
    def __init__(self, keys: list[str]):
        if not keys:
            raise MissingAPIKeyError(
                "No Gemini API key found. Copy .env.example to .env in the project folder "
                "and set GEMINI_API_KEY=<your key> (get one at https://aistudio.google.com/apikey). "
                "Add a second key as key2=<your other key> for automatic failover when one "
                "key's daily quota runs out."
            )
        self._keys = keys
        self._lock = threading.Lock()
        self._index = 0
        self._exhausted: set[int] = set()

    def current_key(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def rotate(self) -> str:
        """Mark the current key exhausted and switch to the next non-exhausted one."""
        with self._lock:
            dead = self._index
            self._exhausted.add(dead)
            for offset in range(1, len(self._keys) + 1):
                candidate = (self._index + offset) % len(self._keys)
                if candidate not in self._exhausted:
                    self._index = candidate
                    break
            else:
                # every key has hit its daily cap - give the pool a fresh look
                # (better than getting permanently stuck) rather than never retrying.
                self._exhausted.clear()
            new_key = self._keys[self._index]
            if self._index != dead:
                print(f"[llm] API key {_mask(self._keys[dead])} hit its daily quota; switching to {_mask(new_key)}.")
            else:
                print(f"[llm] API key {_mask(new_key)} hit its daily quota and no other key is configured/available.")
            return new_key


@lru_cache(maxsize=1)
def _rotator() -> _KeyRotator:
    return _KeyRotator(config.get_api_keys())


_clients: dict[str, genai.Client] = {}
_clients_lock = threading.Lock()


def _client_for_key(api_key: str) -> genai.Client:
    with _clients_lock:
        if api_key not in _clients:
            _clients[api_key] = genai.Client(api_key=api_key)
        return _clients[api_key]


def get_client() -> genai.Client:
    """The Gemini client for the currently-active API key (see _KeyRotator)."""
    return _client_for_key(_rotator().current_key())


GENERATION_DEFAULTS = {"automatic_function_calling": {"disable": True}}


def generation_config(**overrides) -> dict:
    """
    Base config for generate_content calls: disables the SDK's automatic-function-calling
    path (we never use tools/function calling) which otherwise logs a one-time
    "Direct use of AFC ... is not recommended" warning on every fresh process.
    """
    return {**GENERATION_DEFAULTS, **overrides}


# ----------------------------------------------------------------------------
# Text extraction without the noisy "non-text parts... thought_signature" warning
# ----------------------------------------------------------------------------


def extract_text(response) -> str:
    """
    Same result as `response.text`, but without google-genai's console warning.
    Gemini 3 "thinking" models attach a thought_signature part alongside the
    real answer; we just skip non-text / thought parts ourselves.
    """
    candidates = getattr(response, "candidates", None)
    if not candidates or not candidates[0].content or not candidates[0].content.parts:
        return ""
    return "".join(
        part.text
        for part in candidates[0].content.parts
        if isinstance(getattr(part, "text", None), str) and not getattr(part, "thought", False)
    )


# ----------------------------------------------------------------------------
# Rate limiter - ONE bucket PER MODEL, since each Gemini model has its own
# independent per-minute (and per-day) quota. Throttling all models through a
# single shared bucket would waste headroom on models that aren't near their limit.
# ----------------------------------------------------------------------------


class _RateLimiter:
    """Sliding-window limiter: blocks until under `max_per_minute` calls in the last 60s."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max(1, max_per_minute)
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = 60 - (now - self._timestamps[0]) + 0.1
            time.sleep(max(sleep_for, 0.1))


_limiters: dict[str, _RateLimiter] = {}
_limiters_lock = threading.Lock()


def _limiter_for(model: str) -> _RateLimiter:
    with _limiters_lock:
        if model not in _limiters:
            _limiters[model] = _RateLimiter(config.GEMINI_RPM)
        return _limiters[model]


# ----------------------------------------------------------------------------
# Retry policy
# ----------------------------------------------------------------------------


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429  # rate limit / quota
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


def _server_retry_delay(exc: BaseException) -> float | None:
    """Read Google's own suggested wait (e.g. 'retry in 34s') from a 429/5xx error."""
    body = getattr(exc, "details", None) or {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    for detail in (error or {}).get("details", []) or []:
        delay = detail.get("retryDelay")
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1]) + 1  # small buffer
            except ValueError:
                pass
    return None


def _is_daily_quota_exhausted(exc: BaseException) -> bool:
    """
    True for a 429 that's a DAILY-quota exhaustion (quotaId contains "PerDay"),
    as opposed to a per-minute rate limit. A daily quota can't be waited out (it
    resets at midnight Pacific), so this is the signal to switch API keys instead
    of sleeping and retrying the same key.
    """
    body = getattr(exc, "details", None) or {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    for detail in (error or {}).get("details", []) or []:
        if str(detail.get("@type", "")).endswith("QuotaFailure"):
            for violation in detail.get("violations", []) or []:
                if "PerDay" in str(violation.get("quotaId", "")):
                    return True
    return "PerDay" in str(getattr(exc, "message", "") or "") and "quota" in str(exc).lower()


def _wait_time(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if exc is not None and _is_daily_quota_exhausted(exc):
        _rotator().rotate()
        return 0  # new key = a fresh quota bucket, no need to wait before retrying
    hinted = _server_retry_delay(exc) if exc else None
    fallback = wait_exponential(multiplier=2, min=2, max=30)(retry_state)
    return max(hinted, fallback) if hinted is not None else fallback


def with_retry(model: str):
    """
    Decorator factory: throttle calls to `model`'s own per-minute budget and retry
    transient errors (honouring the server's suggested wait), then give up.

    Usage:  @with_retry(config.CHAT_MODEL)
    Keying the limiter by model name means e.g. CHAT_MODEL and UTILITY_MODEL never
    throttle each other, since they draw from separate Gemini quotas.
    """

    def before(retry_state) -> None:  # tenacity "before" hook: runs before every attempt
        _limiter_for(model).acquire()

    def decorator(func):
        return retry(
            retry=retry_if_exception(_is_transient),
            stop=stop_after_attempt(config.MAX_RETRIES),
            wait=_wait_time,
            before=before,
            reraise=True,
        )(func)

    return decorator
