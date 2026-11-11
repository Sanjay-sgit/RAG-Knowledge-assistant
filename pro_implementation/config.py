"""
Central configuration for the Insurellm RAG pipeline.

Every value can be overridden with an environment variable (or a line in .env),
so the same code runs on your laptop and on a server without edits.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root first, then fall back to any .env found in a parent
# directory (useful when this project lives inside a larger workspace).
load_dotenv(PROJECT_ROOT / ".env", override=True)
load_dotenv(override=False)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --- API key(s) ----------------------------------------------------------------
# GEMINI_API_KEY is the standard name; GOOGLE_API_KEY and "key1" are accepted as aliases.
def get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("key1")


def get_api_keys() -> list[str]:
    """
    All configured Gemini API keys, in priority order, for automatic failover when
    one key's free-tier DAILY quota is exhausted (a per-minute limit just waits and
    retries on the same key; a daily limit can't be waited out, so llm.py switches
    to the next key here instead).

    Recognises several naming styles so an existing .env keeps working:
        1st key: GEMINI_API_KEY / GOOGLE_API_KEY / key1
        2nd key: GEMINI_API_KEY_2 / key2
        3rd-9th: GEMINI_API_KEY_3.. / key3.. (extend the same way if you add more)
    Duplicate keys and unset slots are skipped.
    """
    keys: list[str] = []
    primary = get_api_key()
    if primary and primary.strip():
        keys.append(primary.strip())
    for n in range(2, 10):
        extra = os.getenv(f"GEMINI_API_KEY_{n}") or os.getenv(f"key{n}")
        if extra and extra.strip() and extra.strip() not in keys:
            keys.append(extra.strip())
    return keys


# --- Models --------------------------------------------------------------------
# Three roles, deliberately on DIFFERENT models so they draw from separate free-tier
# quotas (each model has its own per-minute AND per-day request budget). Exhausting
# one model's daily quota only degrades that role, not the whole app, and any role can
# be pointed at a different model in .env without touching code.
#
#   CHAT_MODEL     - the final answer the user reads. Needs the most capability.
#   UTILITY_MODEL  - reranking (and query rewriting unless REWRITE_MODEL is set).
#   REWRITE_MODEL  - query rewriting. Tiny prompt, plain-text reply, so a small model with a
#                    high request quota (e.g. a Gemma model) fits well. Moving it off
#                    UTILITY_MODEL halves that model's calls per question. Defaults to UTILITY_MODEL.
#   JUDGE_MODEL    - LLM-as-a-judge scoring in evaluation/eval.py. Defaults to CHAT_MODEL.
#                    Can be a Gemini model or a Groq model (see JUDGE_PROVIDER below).
#   EMBEDDING_MODEL - vector search embeddings.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "gemini-3.5-flash-lite")
REWRITE_MODEL = os.getenv("REWRITE_MODEL") or UTILITY_MODEL
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

# --- Judge provider ------------------------------------------------------------
# A judge from a different model family than the one that wrote the answers avoids
# "grading its own homework", and a separate provider means judge calls never eat
# into the Gemini quota the pipeline needs.
#   gemini - JUDGE_MODEL is called through the Gemini API (same keys as the pipeline)
#   groq   - JUDGE_MODEL (e.g. openai/gpt-oss-120b) is called through Groq's API
# Auto-detected when not set: a "gemini-..." model means gemini, anything else groq.
JUDGE_PROVIDER = (
    os.getenv("JUDGE_PROVIDER") or ("gemini" if JUDGE_MODEL.startswith("gemini") else "groq")
).strip().lower()


def get_judge_key() -> str | None:
    """Groq API key for the judge. GROQ_API_KEY is the standard name; judge_key / JUDGE_API_KEY also work."""
    key = os.getenv("GROQ_API_KEY") or os.getenv("judge_key") or os.getenv("JUDGE_API_KEY")
    return key.strip() if key else None


# Groq free tier for gpt-oss-120b: ~30 requests/min, 1K requests/day, 8K tokens/min,
# 200K tokens/day (check your own numbers at console.groq.com -> Settings -> Limits).
# One judge call is ~1.3-2K tokens, so ~5 calls/min keeps under the 8K tokens/min.
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_RPM = _int("GROQ_RPM", 5)
GROQ_MAX_TOKENS = _int("GROQ_MAX_TOKENS", 1024)  # cap on reasoning + reply tokens per judge call
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")  # low / medium / high (gpt-oss models)
GROQ_TIMEOUT_S = _int("GROQ_TIMEOUT_S", 60)

# --- Storage -----------------------------------------------------------------
KNOWLEDGE_BASE_PATH = Path(os.getenv("KNOWLEDGE_BASE_PATH", PROJECT_ROOT / "knowledge-base"))
DB_PATH = str(Path(os.getenv("CHROMA_DB_PATH", PROJECT_ROOT / "preprocessed_db")))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "docs")

# --- Retrieval ---------------------------------------------------------------
RETRIEVAL_K = _int("RETRIEVAL_K", 20)  # chunks fetched per query from Chroma
FINAL_K = _int("FINAL_K", 10)  # chunks kept after reranking and sent to the LLM
USE_QUERY_REWRITE = _bool("USE_QUERY_REWRITE", True)
USE_RERANK = _bool("USE_RERANK", True)
HISTORY_TURNS = _int("HISTORY_TURNS", 6)  # how many past messages are sent to the LLM

# --- Resilience --------------------------------------------------------------
MAX_RETRIES = _int("MAX_RETRIES", 5)
# Google's free tier allows 15 requests/minute per model; default keeps a safety margin.
GEMINI_RPM = _int("GEMINI_RPM", 10)
# How many parallel LLM calls ingest.py's chunking step makes at once (via multiprocessing).
# Each worker draws from the SAME per-minute quota, so keep this at 1 on the free tier.
INGEST_WORKERS = _int("INGEST_WORKERS", 1)
# Per-HTTP-call timeout for Gemini. The SDK default is "no timeout", so one stalled
# connection could freeze the (single) web worker forever.
GEMINI_HTTP_TIMEOUT_S = _int("GEMINI_HTTP_TIMEOUT_S", 30)

# --- Public demo limits (app.py) -----------------------------------------------
# Every question costs 4 Gemini calls (rewrite, embed, rerank, answer). These caps keep
# a public link from burning the whole free-tier quota.
MAX_QUESTION_CHARS = _int("MAX_QUESTION_CHARS", 500)  # longer questions are rejected
MAX_HISTORY_MESSAGE_CHARS = _int("MAX_HISTORY_MESSAGE_CHARS", 1500)  # earlier turns are trimmed
DAILY_QUESTION_LIMIT = _int("DAILY_QUESTION_LIMIT", 150)  # whole site, resets at midnight UTC; 0 = no limit
QUEUE_MAX_SIZE = _int("QUEUE_MAX_SIZE", 20)  # visitors waiting in line; extra ones are told to retry

# --- Evaluation ----------------------------------------------------------------
# 0 = run all 150 test questions. Set e.g. 10 to try the eval dashboard quickly
# without waiting through the full suite (150 questions x ~4-5 calls each).
EVAL_SAMPLE_SIZE = _int("EVAL_SAMPLE_SIZE", 0)
# Questions evaluated at the same time. Safe for the per-minute quota: every Gemini call
# still waits in its model's shared rate limiter, workers only fill the idle network time.
EVAL_WORKERS = _int("EVAL_WORKERS", 2)
# "Live check" on the /evaluation page: a dropdown of sizes (5, 10, 20 ... 150).
# Each question costs ~5 model calls and counts against DAILY_QUESTION_LIMIT.
EVAL_LIVE_SAMPLE = _int("EVAL_LIVE_SAMPLE", 5)  # size selected by default
# Largest size offered on the PUBLIC page (app.py). A live check holds the server's single
# processing slot - the chat waits until it finishes - and 150 questions would also use
# more than a free model's whole daily quota. The local dashboard (python evaluator.py)
# always offers every size.
EVAL_LIVE_MAX = _int("EVAL_LIVE_MAX", 10)
# Saved results of the latest full run (`python -m evaluation.run`), shown on /evaluation.
EVAL_RESULTS_PATH = Path(os.getenv("EVAL_RESULTS_PATH") or PROJECT_ROOT / "evaluation" / "results.json")
# Sample runs (--sample / EVAL_SAMPLE_SIZE) save here instead, so they never replace the
# full-run results the public page shows. Local only (git-ignored).
EVAL_SAMPLE_RESULTS_PATH = Path(
    os.getenv("EVAL_SAMPLE_RESULTS_PATH") or PROJECT_ROOT / "evaluation" / "results_sample.json"
)
# Every scored question, reused by later runs with identical settings (see evaluation/eval.py).
EVAL_CACHE_PATH = Path(os.getenv("EVAL_CACHE_PATH") or PROJECT_ROOT / "evaluation" / "cache.jsonl")
# Groq's tokens-per-minute limit for the judge model, used only to suggest a safe GROQ_RPM.
GROQ_TPM = _int("GROQ_TPM", 8000)
