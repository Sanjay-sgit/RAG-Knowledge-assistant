"""
Quick end-to-end check that the RAG pipeline works (uses ~4 Gemini calls).

Run from the project root:
    python -m tests.smoke_test
    python -m tests.smoke_test "Who founded Insurellm?"
"""

import sys

from pro_implementation.answer import answer_question


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What is Insurellm?"
    answer, chunks = answer_question(question)

    print(f"\n===== QUESTION =====\n{question}")
    print(f"\n===== ANSWER =====\n{answer}")
    print(f"\n===== RETRIEVED CHUNKS ({len(chunks)}) =====")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}  |  {chunk.metadata.get('source')}")
        print("-" * 60)
        print(chunk.page_content[:300])


if __name__ == "__main__":
    main()
