import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


def load_tests(limit: int | None = None) -> list[TestQuestion]:
    """Load test questions from JSONL file. `limit` keeps only the first N (0/None = all)."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))
    if limit:
        tests = tests[:limit]
    return tests


def sample_tests(n: int | None = None) -> list[TestQuestion]:
    """
    A sample that covers every category: takes the 1st question of each category, then
    the 2nd of each, and so on (round-robin, in file order) until n questions.
    e.g. n=21 -> 3 from each of the 7 categories. 0/None = all questions.

    Deterministic on purpose: the same n always gives the same questions, so runs are
    comparable and the evaluation cache can reuse them. (load_tests(n) instead takes the
    first n lines - the file starts with 70 direct_fact questions, so small samples
    would only test the easiest category.)
    """
    tests = load_tests()
    if not n or n >= len(tests):
        return tests
    by_category: dict[str, list[TestQuestion]] = {}
    for test in tests:
        by_category.setdefault(test.category, []).append(test)
    picked: list[TestQuestion] = []
    depth = 0
    while len(picked) < n:
        for questions in by_category.values():
            if depth < len(questions) and len(picked) < n:
                picked.append(questions[depth])
        depth += 1
    return picked
