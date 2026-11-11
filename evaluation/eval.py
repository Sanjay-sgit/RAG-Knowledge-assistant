"""
RAG evaluation: retrieval metrics (MRR, nDCG, keyword coverage) + LLM-as-a-judge answer scores.

Run one test from the project root:
    python -m evaluation.eval 0

Full run (all questions, saved for the /evaluation page):
    python -m evaluation.run
"""

import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from evaluation.test import TestQuestion, load_tests
from pro_implementation import config
from pro_implementation.answer import answer_question, fetch_context
from pro_implementation.groq_llm import chat_json
from pro_implementation.llm import extract_text, generation_config, get_client, with_retry

MODEL = config.JUDGE_MODEL


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.page_content.lower():
            return 1.0 / rank
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.page_content.lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top documents to retrieve (default 10)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    # Retrieve documents using shared answer module
    retrieved_docs = fetch_context(test.question)
    return retrieval_metrics(test, retrieved_docs, k)


def retrieval_metrics(test: TestQuestion, retrieved_docs: list, k: int = 10) -> RetrievalEval:
    """MRR, nDCG and keyword coverage for chunks that were already retrieved (no API calls)."""
    # Calculate MRR (average across all keywords)
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # Calculate nDCG (average across all keywords)
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Calculate keyword coverage
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


def judge(judge_messages: list[dict]) -> AnswerEval:
    """
    Score an answer with the LLM judge. JUDGE_PROVIDER picks the API:
    gemini (same keys as the pipeline) or groq (e.g. openai/gpt-oss-120b - a different
    model family, so the answers aren't graded by the model that wrote them).
    """
    if config.JUDGE_PROVIDER == "groq":
        return _judge_groq(judge_messages)
    return _judge_gemini(judge_messages)


@with_retry(config.JUDGE_MODEL, rpm=config.GROQ_RPM)
def _judge_groq(judge_messages: list[dict]) -> AnswerEval:
    """Groq judge with strict structured output."""
    return chat_json(
        MODEL,
        system=judge_messages[0]["content"],
        user=judge_messages[1]["content"],
        schema=AnswerEval,
    )


@with_retry(config.JUDGE_MODEL)
def _judge_gemini(judge_messages: list[dict]) -> AnswerEval:
    """Gemini judge with structured output (replaces the old litellm call)."""
    system = judge_messages[0]["content"]
    user = judge_messages[1]["content"]
    response = get_client().models.generate_content(
        model=MODEL,
        contents=user,
        config=generation_config(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=AnswerEval,
        ),
    )
    if isinstance(response.parsed, AnswerEval):
        return response.parsed
    return AnswerEval.model_validate_json(extract_text(response) or "")


def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    # Get RAG response using shared answer module
    generated_answer, retrieved_docs = answer_question(test.question)
    answer_eval = judge(judge_messages_for(test, generated_answer))
    return answer_eval, generated_answer, retrieved_docs


def judge_messages_for(test: TestQuestion, generated_answer: str) -> list[dict]:
    """The LLM-as-a-judge prompt comparing a generated answer with the reference answer."""
    return [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]


# ----------------------------------------------------------------------------
# One-pass evaluation: both retrieval and answer scores from ONE pipeline run.
#
# The old flow ran retrieval (rewrite, embed, rerank) once for the retrieval metrics
# and then the whole pipeline AGAIN for the answer metrics. answer_question() already
# returns the chunks it used (identical to fetch_context(question)), so scoring them
# directly gives the same retrieval metrics for ~5 calls per question instead of ~8.
# ----------------------------------------------------------------------------


class TestResult(BaseModel):
    """All scores for one test question."""

    question: str
    category: str
    mrr: float
    ndcg: float
    keyword_coverage: float
    accuracy: float
    completeness: float
    relevance: float
    answer: str
    feedback: str


def evaluate_one(test: TestQuestion, k: int = 10) -> TestResult:
    """Run the RAG pipeline once and compute retrieval + answer scores for one question."""
    generated_answer, retrieved_docs = answer_question(test.question)
    retrieval = retrieval_metrics(test, retrieved_docs, k)
    answer_eval = judge(judge_messages_for(test, generated_answer))
    return TestResult(
        question=test.question,
        category=test.category,
        mrr=retrieval.mrr,
        ndcg=retrieval.ndcg,
        keyword_coverage=retrieval.keyword_coverage,
        accuracy=answer_eval.accuracy,
        completeness=answer_eval.completeness,
        relevance=answer_eval.relevance,
        answer=generated_answer,
        feedback=answer_eval.feedback,
    )


def evaluate_many(tests: list[TestQuestion], workers: int | None = None):
    """
    Evaluate tests with `workers` questions in flight at once (default EVAL_WORKERS).
    Yields (test, result, done, total) as each finishes; result is None for a test that
    errored out (quota, overloaded model, ...) so one bad test doesn't stop the run.
    Per-minute quota stays safe: every call still waits in its model's shared limiter.
    """
    workers = max(1, workers or config.EVAL_WORKERS)
    total = len(tests)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_one, test): test for test in tests}
        for done, future in enumerate(as_completed(futures), start=1):
            test = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"[eval] Skipping test ({test.question[:60]!r}): {exc}")
                result = None
            yield test, result, done, total


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[TestResult], skipped: int = 0, duration_s: float = 0.0) -> dict:
    """Averages overall and per category, plus the settings the run used."""
    by_category = defaultdict(list)
    for r in results:
        by_category[r.category].append(r)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "num_tests": len(results),
        "skipped": skipped,
        "duration_s": round(duration_s, 1),
        "models": {
            "chat": config.CHAT_MODEL,
            "rewrite": config.REWRITE_MODEL,
            "rerank": config.UTILITY_MODEL,
            "embedding": config.EMBEDDING_MODEL,
            "judge": config.JUDGE_MODEL,
            "judge_provider": config.JUDGE_PROVIDER,
        },
        "settings": {
            "retrieval_k": config.RETRIEVAL_K,
            "final_k": config.FINAL_K,
            "query_rewrite": config.USE_QUERY_REWRITE,
            "rerank": config.USE_RERANK,
        },
        "retrieval": {
            "mrr": _avg([r.mrr for r in results]),
            "ndcg": _avg([r.ndcg for r in results]),
            "keyword_coverage": _avg([r.keyword_coverage for r in results]),
        },
        "answer": {
            "accuracy": _avg([r.accuracy for r in results]),
            "completeness": _avg([r.completeness for r in results]),
            "relevance": _avg([r.relevance for r in results]),
        },
        "by_category": [
            {
                "category": category,
                "count": len(rows),
                "mrr": _avg([r.mrr for r in rows]),
                "ndcg": _avg([r.ndcg for r in rows]),
                "keyword_coverage": _avg([r.keyword_coverage for r in rows]),
                "accuracy": _avg([r.accuracy for r in rows]),
                "completeness": _avg([r.completeness for r in rows]),
                "relevance": _avg([r.relevance for r in rows]),
            }
            for category, rows in sorted(by_category.items())
        ],
        "results": [r.model_dump() for r in results],
    }


def save_results(summary: dict, path: Path | None = None) -> Path:
    """Write the summary as JSON (via a temp file, so a crash never leaves half a file)."""
    path = Path(path or config.EVAL_RESULTS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def load_results(path: Path | None = None) -> dict | None:
    """The saved summary of the latest full run, or None if there isn't a valid one."""
    path = Path(path or config.EVAL_RESULTS_PATH)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def random_tests(n: int) -> list[TestQuestion]:
    """n random test questions (for the quick live check)."""
    tests = load_tests()
    return random.sample(tests, min(max(n, 1), len(tests)))


def evaluate_all_retrieval():
    """
    Evaluate retrieval tests (all 150, or config.EVAL_SAMPLE_SIZE of them if set).
    Yields (test, result, progress); result is None for a test that errored out
    (e.g. a model ran out of quota) so one bad test doesn't stop the whole run.
    """
    tests = load_tests(config.EVAL_SAMPLE_SIZE or None)
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result = evaluate_retrieval(test)
        except Exception as exc:
            print(f"[eval] Skipping retrieval test {index} ({test.question[:60]!r}): {exc}")
            result = None
        progress = (index + 1) / total_tests
        yield test, result, progress


def evaluate_all_answers():
    """
    Evaluate answers to tests, one at a time (all 150, or config.EVAL_SAMPLE_SIZE if set).
    Yields (test, result, progress); result is None for a test that errored out, so one
    bad test doesn't stop the whole run.
    """
    tests = load_tests(config.EVAL_SAMPLE_SIZE or None)
    total_tests = len(tests)
    for index, test in enumerate(tests):
        try:
            result = evaluate_answer(test)[0]
        except Exception as exc:
            print(f"[eval] Skipping answer test {index} ({test.question[:60]!r}): {exc}")
            result = None
        progress = (index + 1) / total_tests
        yield test, result, progress


def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test (CLI helper)."""
    # Load tests
    tests = load_tests()

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    # Get the test
    test = tests[test_number]

    # Print test info
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # Retrieval Evaluation
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    # Answer Evaluation
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


def main():
    """CLI to evaluate a specific test by row number."""
    if len(sys.argv) != 2:
        print("Usage (from the project root): python -m evaluation.eval <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
