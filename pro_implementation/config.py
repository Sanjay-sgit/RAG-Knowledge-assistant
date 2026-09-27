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
#   UTILITY_MODEL  - query rewriting + reranking. Called more often, needs less capability.
#   JUDGE_MODEL    - LLM-as-a-judge scoring in evaluation/eval.py. Defaults to CHAT_MODEL.
#   EMBEDDING_MODEL - vector search embeddings.
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-2.5-flash")
UTILITY_MODEL = os.getenv("UTILITY_MODEL", "gemini-3.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")

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

# --- Evaluation ----------------------------------------------------------------
# 0 = run all 150 test questions. Set e.g. 10 to try the eval dashboard quickly
# without waiting through the full suite (150 questions x ~4-5 calls each).
EVAL_SAMPLE_SIZE = _int("EVAL_SAMPLE_SIZE", 0)
