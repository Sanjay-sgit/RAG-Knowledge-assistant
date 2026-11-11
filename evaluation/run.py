"""
Full evaluation run (local). Scores every test question once - retrieval metrics and
LLM-as-a-judge answer scores from the same pipeline run - and saves the summary to
evaluation/results.json, which the public /evaluation page displays.

From the project root:
    python -m evaluation.run                # all 150 questions
    python -m evaluation.run --sample 20    # first 20 only (quick check)
    python -m evaluation.run --workers 3    # questions in flight at once (default EVAL_WORKERS)

Cost: ~5 Gemini calls per question (rewrite, embed, rerank, answer, judge).
Use a key the live site doesn't use, so the demo keeps its quota.
"""

import argparse
import sys
import time

from evaluation.eval import evaluate_many, save_results, summarize
from evaluation.test import load_tests
from pro_implementation import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline and save the results.")
    parser.add_argument("--sample", type=int, default=0, help="only the first N questions (0 = all)")
    parser.add_argument("--workers", type=int, default=config.EVAL_WORKERS, help="questions in flight at once")
    parser.add_argument("--out", default=str(config.EVAL_RESULTS_PATH), help="where to save the JSON summary")
    args = parser.parse_args()

    tests = load_tests(args.sample or None)
    print(
        f"Evaluating {len(tests)} questions with {args.workers} worker(s)\n"
        f"  chat={config.CHAT_MODEL}  rewrite={config.REWRITE_MODEL}  rerank={config.UTILITY_MODEL}\n"
        f"  judge={config.JUDGE_MODEL} ({config.JUDGE_PROVIDER})  embedding={config.EMBEDDING_MODEL}  GEMINI_RPM={config.GEMINI_RPM}\n"
    )

    started = time.monotonic()
    results, skipped = [], 0
    for _test, result, done, total in evaluate_many(tests, args.workers):
        if result is None:
            skipped += 1
        else:
            results.append(result)
        elapsed = time.monotonic() - started
        eta = elapsed / done * (total - done)
        print(f"  [{done:>3}/{total}] ok={len(results)} skipped={skipped}  elapsed {elapsed / 60:.1f} min  ETA {eta / 60:.1f} min")

    if not results:
        print("\nEvery question failed - check your API key, model names and quota. Nothing saved.")
        sys.exit(1)

    summary = summarize(results, skipped=skipped, duration_s=time.monotonic() - started)
    path = save_results(summary, args.out)

    r, a = summary["retrieval"], summary["answer"]
    print(
        f"\nDone in {summary['duration_s'] / 60:.1f} min - {len(results)} scored, {skipped} skipped\n"
        f"  Retrieval: MRR {r['mrr']:.4f}   nDCG {r['ndcg']:.4f}   keyword coverage {r['keyword_coverage']:.1f}%\n"
        f"  Answers:   accuracy {a['accuracy']:.2f}/5   completeness {a['completeness']:.2f}/5   relevance {a['relevance']:.2f}/5\n"
        f"Saved to {path}"
    )


if __name__ == "__main__":
    main()
