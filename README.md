# Insurellm Expert Assistant — Advanced RAG

A question-answering assistant over a company knowledge base (76 markdown docs: company, contracts, employees, products), built **without LangChain** on Google Gemini + ChromaDB, with a built-in evaluation suite.

**Pipeline per question:** query rewrite (history-aware) → dual vector search → LLM rerank → grounded answer with sources.

## Project layout

```
app.py                       Gradio chat UI
evaluator.py                 Gradio evaluation dashboard
gradio_compat.py             Gradio 5/6 compatibility helpers
pro_implementation/
    config.py                all settings (env-overridable)
    llm.py                   Gemini client + bounded retry policy
    answer.py                RAG pipeline: rewrite -> retrieve -> rerank -> answer
    ingest.py                build the vector store (LLM chunking + embeddings)
evaluation/
    eval.py                  MRR, nDCG, keyword coverage + LLM-as-a-judge
    test.py, tests.jsonl     150 test questions in 7 categories
tests/smoke_test.py          one-question end-to-end check
knowledge-base/              source documents
preprocessed_db/             Chroma vector store (620 chunks, 3072-dim)
implementation/              LangChain baseline (for comparison only)
notebooks/                   exploration notebooks: fundamentals -> vector DBs -> evaluation
```

## Setup (Windows)

```bat
git clone https://github.com/Sanjay-sgit/RAG-Knowledge-assistant.git
cd RAG-Knowledge-assistant
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set `GEMINI_API_KEY=...` (free key: https://aistudio.google.com/apikey).


macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`, `cp .env.example .env`.

## Run

All commands from the project root, with the venv active:

| What | Command |
|---|---|
| Smoke test (1 question) | `python -m tests.smoke_test "Who founded Insurellm?"` |
| Chat UI | `python app.py` → http://127.0.0.1:7860 |
| Evaluate one test question | `python -m evaluation.eval 0` |
| Evaluation dashboard | `python evaluator.py` |
| Rebuild the vector store (only if knowledge base changes) | `python -m pro_implementation.ingest` |

## Configuration

Everything is set in `.env` (see `.env.example`): models, `RETRIEVAL_K`, `FINAL_K`,
`USE_QUERY_REWRITE`, `USE_RERANK`, `MAX_RETRIES`. Turning off rewrite/rerank cuts Gemini calls
per question from 4 to 2 (useful on the free tier).

## Free-tier quota (429 "RESOURCE_EXHAUSTED")

Google's free tier caps each model at both a per-minute AND a per-day request count (check
your live usage at https://aistudio.google.com/rate-limit). Two things protect you from this:

1. **Three roles, three different models** — `CHAT_MODEL`, `UTILITY_MODEL` (query rewrite +
   rerank) and `EMBEDDING_MODEL` each draw from their own quota (see `.env.example`). If one
   model's daily cap runs out, only that role degrades — rewrite/rerank fall back to the plain
   question / original chunk order instead of crashing the answer.
2. **A per-model rate limiter + retry** (`GEMINI_RPM`, default 10/minute per model) paces every
   call and honours Google's own suggested wait time on a 429 instead of guessing.

A **per-minute** 429 is handled automatically (wait and retry). A **per-day** 429 (the daily
quota is fully used) can't be waited out — it resets at midnight Pacific time. Two ways to
recover from that, and you can use either or both:

1. **Add a second API key.** Set `key2=<another key>` in `.env` (from a second free Google
   account/project — each key has its own independent daily quota). The app detects a daily-quota
   429 automatically and switches to `key2` immediately, no restart needed, and stays on it for
   the rest of the run. `key3`, etc. also work.
2. **Switch the affected role to a different model** in `.env` (e.g. `CHAT_MODEL=gemini-2.0-flash`)
   and restart.

The eval dashboard also **skips a failed test and continues** instead of stopping the whole run
(only relevant once every configured key + model combination is genuinely out of quota), and
tells you how many were skipped.

- Running the **eval dashboard** against all 150 questions takes a while by design (each
  question is 3–5 calls, paced at ~10/minute per model). Set `EVAL_SAMPLE_SIZE=10` in `.env` to
  try it quickly on a subset first.
- Running **ingest** with more than 1 worker (`INGEST_WORKERS`) multiplies quota pressure —
  keep it at 1 on the free tier.

## Model deprecation ("404 NOT_FOUND ... model is no longer available")

Google periodically retires free-tier model names. This is different from a 429 quota
error: it's not transient, so the app doesn't retry it — for `UTILITY_MODEL` (query
rewrite + rerank) it just logs a line and falls back to the plain question / original
retrieval order, same as a quota exhaustion. The fix is a config change, not a code
change: the 404's own error message names the exact replacement model — put that in
`UTILITY_MODEL` (or whichever role hit it) in `.env` and restart.

## Evaluation

- **Retrieval:** MRR, nDCG@10, keyword coverage over 150 questions (direct_fact, temporal, spanning, comparative, numerical, relationship, holistic).
- **Answers:** LLM-as-a-judge scores (1–5) for accuracy, completeness, relevance against reference answers.
