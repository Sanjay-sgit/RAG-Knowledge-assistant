"""
Advanced (manual, no-LangChain) RAG pipeline for Insurellm.

Per question:
    1. rewrite_query        -> Gemini rewrites the question into a short KB search query (uses history)
    2. embed both queries   -> one Gemini embedding call for [original, rewritten]
    3. Chroma search        -> top RETRIEVAL_K chunks for each query, merged + de-duplicated
    4. rerank               -> Gemini orders the chunks by relevance, keep FINAL_K
    5. generate             -> Gemini answers using the chunks as context

Public API (used by app.py, evaluation/eval.py and later the FastAPI service):
    answer_question(question, history) -> (answer: str, chunks: list[Result])
    fetch_context(question, history)   -> list[Result]
"""

import json
from functools import lru_cache

from chromadb import PersistentClient
from pydantic import BaseModel, Field

from pro_implementation import config
from pro_implementation.llm import extract_text, generation_config, get_client, with_retry

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question. Be accurate, relevant and complete.
"""

FALLBACK_ANSWER = "Sorry, I couldn't generate an answer for that. Please try rephrasing your question."


class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# ----------------------------------------------------------------------------
# Vector store (lazy)
# ----------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_collection():
    chroma = PersistentClient(path=config.DB_PATH)
    try:
        return chroma.get_collection(config.COLLECTION_NAME)
    except Exception as exc:  # collection missing -> DB was never built
        raise RuntimeError(
            f"Vector store '{config.COLLECTION_NAME}' not found in {config.DB_PATH}. "
            "Build it first with:  python -m pro_implementation.ingest"
        ) from exc


def display_source(source: str) -> str:
    """Turn 'D:/.../week5/knowledge-base/employees/Alex.md' into 'knowledge-base/employees/Alex.md'."""
    normalized = source.replace("\\", "/")
    marker = "knowledge-base/"
    index = normalized.find(marker)
    return normalized[index:] if index >= 0 else normalized.rsplit("/", 1)[-1]


# ----------------------------------------------------------------------------
# Helpers for chat history (works with Gradio 5 and 6 message formats)
# ----------------------------------------------------------------------------


def message_text(content) -> str:
    """Gradio 6 may send content as a list of parts; Gradio 5 / API clients send a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, (list, tuple)):
        return " ".join(message_text(part) for part in content).strip()
    return "" if content is None else str(content)


def format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    lines = []
    for message in history[-config.HISTORY_TURNS :]:
        role = "User" if message.get("role") == "user" else "Assistant"
        text = message_text(message.get("content"))
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------


@with_retry(config.UTILITY_MODEL)
def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Rewrite the user's question into a short, specific Knowledge Base query."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{format_history(history) or "(no previous messages)"}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
If the question refers to something earlier in the conversation (e.g. "he", "that product"), make it explicit.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = get_client().models.generate_content(
        model=config.UTILITY_MODEL, contents=message, config=generation_config()
    )
    return (extract_text(response) or question).strip()


@with_retry(config.EMBEDDING_MODEL)
def embed_queries(queries: list[str]) -> list[list[float]]:
    """Embed several queries in a single API call."""
    response = get_client().models.embed_content(model=config.EMBEDDING_MODEL, contents=queries)
    return [embedding.values for embedding in response.embeddings]


def search(query_embedding: list[float], k: int | None = None) -> list[Result]:
    results = get_collection().query(query_embeddings=[query_embedding], n_results=k or config.RETRIEVAL_K)
    chunks = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        metadata = dict(metadata or {})
        metadata["source"] = display_source(metadata.get("source", "unknown"))
        chunks.append(Result(page_content=document, metadata=metadata))
    return chunks


def fetch_context_unranked(question: str) -> list[Result]:
    """Plain vector search for one question (kept for notebooks / experiments)."""
    return search(embed_queries([question])[0])


def merge_chunks(chunks: list[Result], extra: list[Result]) -> list[Result]:
    merged = chunks[:]
    existing = {chunk.page_content for chunk in chunks}
    for chunk in extra:
        if chunk.page_content not in existing:
            merged.append(chunk)
            existing.add(chunk.page_content)
    return merged


def _parse_order(response) -> list[int]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, RankOrder):
        return parsed.order
    try:
        return RankOrder.model_validate(json.loads(extract_text(response) or "")).order
    except Exception:
        return []


@with_retry(config.UTILITY_MODEL)
def rerank(question: str, chunks: list[Result]) -> list[Result]:
    """Ask the LLM to order chunks by relevance. Never crashes on a bad reply."""
    if len(chunks) <= 1:
        return chunks

    prompt = f"""
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.

The user has asked the following question:

{question}

Order all the chunks of text by relevance to the question, from most relevant to least relevant.
Include all the chunk ids you are provided with, reranked.

Here are the chunks:

"""
    for index, chunk in enumerate(chunks, start=1):
        prompt += f"# CHUNK ID: {index}:\n\n{chunk.page_content}\n\n"
    prompt += "Reply only with the list of ranked chunk ids, nothing else."

    response = get_client().models.generate_content(
        model=config.UTILITY_MODEL,
        contents=prompt,
        config=generation_config(response_mime_type="application/json", response_schema=RankOrder),
    )

    # Keep only valid, unique ids; anything the model forgot keeps its original position after.
    seen: set[int] = set()
    ranked: list[Result] = []
    for chunk_id in _parse_order(response):
        if isinstance(chunk_id, int) and 1 <= chunk_id <= len(chunks) and chunk_id not in seen:
            seen.add(chunk_id)
            ranked.append(chunks[chunk_id - 1])
    ranked.extend(chunk for index, chunk in enumerate(chunks, start=1) if index not in seen)
    return ranked


def fetch_context(question: str, history: list[dict] | None = None) -> list[Result]:
    """
    Retrieve the FINAL_K most relevant chunks for a question.

    Query rewrite and reranking are quality boosts, not requirements - if UTILITY_MODEL
    is out of quota (or errors for any other reason) even after retries, fall back to the
    plain question / original retrieval order instead of failing the whole answer.
    """
    queries = [question]
    if config.USE_QUERY_REWRITE:
        try:
            rewritten = rewrite_query(question, history)
            if rewritten and rewritten.lower() != question.lower():
                queries.append(rewritten)
        except Exception as exc:
            print(f"[answer] Query rewrite failed ({exc}); continuing with the original question.")

    embeddings = embed_queries(queries)
    chunks: list[Result] = []
    for embedding in embeddings:
        chunks = merge_chunks(chunks, search(embedding))

    if config.USE_RERANK:
        try:
            chunks = rerank(question, chunks)
        except Exception as exc:
            print(f"[answer] Rerank failed ({exc}); using retrieval order instead.")
    return chunks[: config.FINAL_K]


def make_rag_prompt(question: str, history: list[dict] | None, chunks: list[Result]) -> str:
    context = "\n\n".join(
        f"Extract from {chunk.metadata.get('source', 'unknown')}:\n{chunk.page_content}" for chunk in chunks
    )
    prompt = SYSTEM_PROMPT.format(context=context)
    conversation = format_history(history)
    if conversation:
        prompt += f"\n\nConversation history:\n{conversation}"
    prompt += f"\n\nQuestion:\n{question}"
    return prompt


@with_retry(config.CHAT_MODEL)
def generate_answer(prompt: str) -> str:
    response = get_client().models.generate_content(
        model=config.CHAT_MODEL, contents=prompt, config=generation_config()
    )
    return (extract_text(response)).strip() or FALLBACK_ANSWER


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Result]]:
    """Answer a question using RAG and return (answer, retrieved chunks)."""
    question = (question or "").strip()
    if not question:
        return "Please ask a question about Insurellm.", []
    history = history or []
    chunks = fetch_context(question, history)
    answer = generate_answer(make_rag_prompt(question, history, chunks))
    return answer, chunks
