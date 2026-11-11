"""
Minimal Groq client for the LLM-as-a-judge (e.g. openai/gpt-oss-120b).

Uses Groq's OpenAI-compatible REST API through httpx (already a dependency), so no
extra package is needed locally or on the server.

- Strict structured output: the reply is guaranteed to match the pydantic schema.
- Low reasoning effort + a token cap, because gpt-oss "thinks" before answering and
  those tokens count against Groq's per-minute / per-day token limits.
- Errors are raised as GroqError, which tells llm.with_retry whether retrying can help
  (rate limit / server hiccup: yes; daily limit / bad key / bad request: no).
"""

import json
import re
import threading

import httpx
from pydantic import BaseModel

from pro_implementation import config


class GroqError(RuntimeError):
    """A failed Groq call. `transient` tells the retry policy whether to try again."""

    provider = "groq"

    def __init__(self, message: str, status: int | None = None, transient: bool = False, retry_after: float | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.transient = transient
        self.retry_after = retry_after


_client: httpx.Client | None = None
_client_lock = threading.Lock()


def _http() -> httpx.Client:
    global _client
    with _client_lock:
        if _client is None:
            key = config.get_judge_key()
            if not key:
                raise GroqError(
                    "No Groq API key found for the judge. Set GROQ_API_KEY (or judge_key) in .env, "
                    "or use a gemini-... JUDGE_MODEL. Get a key at https://console.groq.com/keys",
                    transient=False,
                )
            _client = httpx.Client(
                base_url=config.GROQ_BASE_URL,
                headers={"Authorization": f"Bearer {key}"},
                timeout=config.GROQ_TIMEOUT_S,
            )
        return _client


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema adapted to strict mode: every field required, no extra fields."""
    schema = model.model_json_schema()

    def tighten(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for value in node:
                tighten(value)

    tighten(schema)
    return schema


def _parse_wait(text: str) -> float | None:
    """Groq's 'Please try again in 7m12.5s' / '2.5s' / '450ms' -> seconds."""
    match = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?\s*(?:([\d.]+)ms)?", text or "")
    if not match or not any(match.groups()):
        return None
    h, m, s, ms = match.groups()
    return int(h or 0) * 3600 + int(m or 0) * 60 + float(s or 0) + float(ms or 0) / 1000


def _error_from_response(response: httpx.Response) -> GroqError:
    try:
        error = response.json().get("error", {})
    except ValueError:
        error = {}
    message = error.get("message") or response.text[:300] or f"HTTP {response.status_code}"
    code = error.get("code") or ""
    status = response.status_code
    label = f"Groq {status}{f' {code}' if code else ''}: {message}"

    if status == 429:
        header = response.headers.get("retry-after")
        try:
            wait = float(header) if header else _parse_wait(message)
        except ValueError:
            wait = _parse_wait(message)
        daily = "per day" in message.lower() or "(tpd)" in message.lower() or "(rpd)" in message.lower()
        if daily or (wait is not None and wait > 120):
            # Daily limit: can't be waited out in a reasonable time. Fail fast so the
            # question is skipped instead of blocking the run for hours.
            return GroqError(f"{label} (daily limit - try again later)", status, transient=False)
        return GroqError(label, status, transient=True, retry_after=wait)
    if status >= 500:
        return GroqError(label, status, transient=True)
    if status == 400 and code == "json_validate_failed":
        # Groq occasionally generates JSON that fails its own schema check; a retry usually works.
        return GroqError(label, status, transient=True)
    if status == 401:
        return GroqError(f"{label} - check GROQ_API_KEY / judge_key", status, transient=False)
    return GroqError(label, status, transient=False)


def chat_json(model: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
    """One chat completion whose reply is parsed into `schema`. Raises GroqError on failure."""
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "strict": True, "schema": strict_schema(schema)},
        },
        "max_completion_tokens": config.GROQ_MAX_TOKENS,
    }
    if config.GROQ_REASONING_EFFORT and "gpt-oss" in model:
        body["reasoning_effort"] = config.GROQ_REASONING_EFFORT

    response = _http().post("/chat/completions", json=body)
    if response.status_code != 200:
        raise _error_from_response(response)

    choice = (response.json().get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if choice.get("finish_reason") == "length":
        raise GroqError(
            f"Groq reply was cut off at GROQ_MAX_TOKENS={config.GROQ_MAX_TOKENS}; raise it in .env",
            transient=False,
        )
    try:
        return schema.model_validate(json.loads(content))
    except ValueError as exc:
        raise GroqError(f"Groq reply was not valid {schema.__name__} JSON: {exc}", transient=True) from exc
