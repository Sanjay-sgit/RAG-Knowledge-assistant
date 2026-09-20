"""
Build the vector store for the advanced RAG pipeline.

    knowledge-base/*.md  ->  Gemini splits each doc into chunks (headline + summary + original text)
                         ->  Gemini embeddings  ->  Chroma (preprocessed_db, collection "docs")

Run from the project root (takes a while on the free tier, ~80 LLM calls + embedding batches):
    python -m pro_implementation.ingest

You only need to run this again if the knowledge base changes.
"""

import os
import time
from multiprocessing import Pool

from chromadb import PersistentClient
from pydantic import BaseModel, Field
from tqdm import tqdm

from pro_implementation import config
from pro_implementation.answer import Result
from pro_implementation.llm import extract_text, generation_config, get_client, with_retry

AVERAGE_CHUNK_SIZE = 500
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", 50))
EMBED_SLEEP_SECONDS = float(os.getenv("EMBED_SLEEP_SECONDS", 30))  # free-tier friendly pause


class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, typically a few words, that is most likely to be surfaced in a query",
    )
    summary: str = Field(
        description="A few sentences summarizing the content of this chunk to answer common questions"
    )
    original_text: str = Field(
        description="The original text of this chunk from the provided document, exactly as is, not changed in any way"
    )

    def as_result(self, document: dict) -> Result:
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata=metadata,
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


def fetch_documents() -> list[dict]:
    """A homemade version of the LangChain DirectoryLoader."""
    documents = []
    for folder in sorted(config.KNOWLEDGE_BASE_PATH.iterdir()):
        if not folder.is_dir():
            continue
        for file in sorted(folder.rglob("*.md")):
            documents.append(
                {
                    "type": folder.name,
                    # store a portable, relative path (not D:/your/disk/...)
                    "source": file.relative_to(config.KNOWLEDGE_BASE_PATH.parent).as_posix(),
                    "text": file.read_text(encoding="utf-8"),
                }
            )
    print(f"Loaded {len(documents)} documents")
    return documents


def make_prompt(document: dict) -> str:
    how_many = (len(document["text"]) // AVERAGE_CHUNK_SIZE) + 1
    return f"""
You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned across the chunks - don't leave anything out.
This document should probably be split into at least {how_many} chunks, but you can have more or less as appropriate, ensuring that there are individual chunks to answer specific questions.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.
"""


@with_retry(config.CHAT_MODEL)
def process_document(document: dict) -> list[Result]:
    response = get_client().models.generate_content(
        model=config.CHAT_MODEL,
        contents=make_prompt(document),
        config=generation_config(response_mime_type="application/json", response_schema=Chunks),
    )
    parsed = response.parsed
    if not isinstance(parsed, Chunks):
        parsed = Chunks.model_validate_json(extract_text(response) or "")
    return [chunk.as_result(document) for chunk in parsed.chunks]


def create_chunks(documents: list[dict]) -> list[Result]:
    """Create chunks using a number of workers in parallel."""
    chunks: list[Result] = []
    workers = config.INGEST_WORKERS
    if workers <= 1:
        for document in tqdm(documents, desc="Chunking"):
            chunks.extend(process_document(document))
        return chunks
    with Pool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents), desc="Chunking"):
            chunks.extend(result)
    return chunks


@with_retry(config.EMBEDDING_MODEL)
def embed_batch(batch: list[str]) -> list[list[float]]:
    response = get_client().models.embed_content(model=config.EMBEDDING_MODEL, contents=batch)
    return [embedding.values for embedding in response.embeddings]


def create_embeddings(chunks: list[Result]) -> None:
    texts = [chunk.page_content for chunk in chunks]
    vectors: list[list[float]] = []
    for start in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Embedding"):
        vectors.extend(embed_batch(texts[start : start + EMBED_BATCH_SIZE]))
        if start + EMBED_BATCH_SIZE < len(texts):
            time.sleep(EMBED_SLEEP_SECONDS)

    # Only replace the old collection once all embeddings succeeded.
    chroma = PersistentClient(path=config.DB_PATH)
    if config.COLLECTION_NAME in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(config.COLLECTION_NAME)
    collection = chroma.get_or_create_collection(config.COLLECTION_NAME)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=vectors,
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    print(f"Vectorstore created with {collection.count()} chunks")


def main() -> None:
    get_client()  # fail fast if the API key is missing
    documents = fetch_documents()
    chunks = create_chunks(documents)
    print(f"Total chunks created: {len(chunks)}")
    create_embeddings(chunks)
    print("Ingestion complete")


if __name__ == "__main__":
    main()
