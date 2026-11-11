"""
Evaluation run (local). Scores test questions once - retrieval metrics and
LLM-as-a-judge answer scores from the same pipeline run.

From the project root:
    python -m evaluation.run                # all 150 questions -> evaluation/results.json (shown on /evaluation)
    python -m evaluation.run --sample 21    # 3 per category    -> evaluation/results_sample.json
    python -m evaluation.run --fresh        # ignore the cache, score everything again
    python -m evaluation.run --workers 3    # questions in flight at once (default EVAL_WORKERS)

Cache: every scored question is saved to evaluation/cache.jsonl as soon as it finishes.
A later run with the SAME settings reuses it, so a run stopped by a daily limit just
resumes, and a sample's questions aren't scored twice. If only the judge changed, saved
answers are re-judged without calling Gemini.

Cost: ~4-5 model calls per new question (rewrite, embed, [rerank], answer, judge).
Use a key the live site doesn't use, so the demo keeps its quota.
"""

import argparse
import sys
import time

from evaluation.eval import EvalCache, evaluate_many, save_results, summarize
from evaluation.test import sample_tests
from pro_implementation import config
from pro_implementation.groq_llm import token_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline and save the results.")
    parser.add_argument("--sample", type=int, default=0, help="N questions spread over all categories (0 = all)")
    parser.add_argument("--workers", type=int, default=config.EVAL_WORKERS, help="questions in flight at once")
    parser.add_argument("--fresh", action="store_true", help="ignore the cache and score every question again")
    parser.add_argument("--out", default=None, help="where to save the JSON summary")
    args = parser.parse_args()

    tests = sample_tests(args.sample or None)
    is_sample = len(tests) < len(sample_tests())
    out = args.out or (config.EVAL_SAMPLE_RESULTS_PATH if is_sample else config.EVAL_RESULTS_PATH)
    cache = None if args.fresh else EvalCache()

    print(
        f"Evaluating {len(tests)} questions{' (sample over all categories)' if is_sample else ''} "
        f"with {args.workers} worker(s)\n"
        f"  chat={config.CHAT_MODEL}  rewrite={config.REWRITE_MODEL if config.USE_QUERY_REWRITE else 'off'}  "
        f"rerank={config.UTILITY_MODEL if config.USE_RERANK else 'off'}\n"
        f"  judge={config.JUDGE_MODEL} ({config.JUDGE_PROVIDER})  embedding={config.EMBEDDING_MODEL}\n"
        f"  GEMINI_RPM={config.GEMINI_RPM}"
        + (f"  GROQ_RPM={config.GROQ_RPM}" if config.JUDGE_PROVIDER == "groq" else "")
    )
    if cache is not None:
        plan = cache.plan(tests)
        print(
            f"  cache: {plan['cached']} already scored, {plan['rejudge']} to re-judge only, "
            f"{plan['new']} to run  ({cache.path.name}; --fresh to ignore)"
        )
    print()

    started = time.monotonic()
    results, skipped = [], 0
    for _test, result, done, total in evaluate_many(tests, args.workers, cache):
        if result is None:
            skipped += 1
        else:
            results.append(result)
        elapsed = time.monotonic() - started
        eta = elapsed / done * (total - done)
        print(f"  [{done:>3}/{total}] ok={len(results)} skipped={skipped}  elapsed {elapsed / 60:.1f} min  ETA {eta / 60:.1f} min")

    if not results:
        print("\nEvery question failed - check your API keys, model names and quota. Nothing saved.")
        sys.exit(1)

    summary = summarize(results, skipped=skipped, duration_s=time.monotonic() - started, sample=is_sample)
    path = save_results(summary, out)

    r, a = summary["retrieval"], summary["answer"]
    reused = summary["from_cache"] + summary["rejudged"]
    print(
        f"\nDone in {summary['duration_s'] / 60:.1f} min - {len(results)} scored"
        f" ({summary['from_cache']} from cache, {summary['rejudged']} re-judged), {skipped} skipped\n"
        f"  Retrieval: MRR {r['mrr']:.4f}   nDCG {r['ndcg']:.4f}   keyword coverage {r['keyword_coverage']:.1f}%\n"
        f"  Answers:   accuracy {a['accuracy']:.2f}/5   completeness {a['completeness']:.2f}/5   relevance {a['relevance']:.2f}/5\n"
        f"Saved to {path}"
    )
    if skipped:
        print(f"{skipped} question(s) failed. Run the same command again later: only those will be retried.")
    if reused and not skipped:
        print("Tip: --fresh re-scores everything from scratch.")

    tokens = token_stats()
    if config.JUDGE_PROVIDER == "groq" and tokens["calls"]:
        avg = tokens["avg_tokens_per_call"]
        safe_rpm = max(1, int(config.GROQ_TPM * 0.9 // max(avg, 1)))
        print(
            f"\nGroq judge: {tokens['calls']} calls, ~{avg} tokens/call "
            f"(prompt {tokens['prompt_tokens'] // tokens['calls']}, reply+reasoning {tokens['completion_tokens'] // tokens['calls']}).\n"
            f"  Groq's {config.GROQ_TPM} tokens/min allows ~{safe_rpm} calls/min (with 10% headroom); "
            f"GROQ_RPM is {config.GROQ_RPM}."
            + (f" You can raise it to {min(safe_rpm, 30)}." if safe_rpm > config.GROQ_RPM else "")
            + (f" Lower it to {safe_rpm}." if safe_rpm < config.GROQ_RPM else "")
        )


if __name__ == "__main__":
    main()
