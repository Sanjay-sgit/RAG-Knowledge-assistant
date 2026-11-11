"""
RAG Evaluation page (Gradio).

Two ways to use it:
- Public, inside app.py at /evaluation: shows the saved results of the latest full run
  (evaluation/results.json) plus a small "live check" on a few random questions.
- Locally, standalone:   python evaluator.py
  Same page plus a "Run full evaluation" button that re-scores the test set and saves
  the results. (Command-line alternative: python -m evaluation.run)
"""

import html
import time
from collections.abc import Callable

import gradio as gr
import pandas as pd

from evaluation.eval import EvalCache, evaluate_many, load_results, random_tests, save_results, summarize
from evaluation.test import sample_tests
from gradio_compat import blocks_kwargs, launch_kwargs
from pro_implementation import config

# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0

THRESHOLDS = {
    "mrr": (MRR_GREEN, MRR_AMBER),
    "ndcg": (NDCG_GREEN, NDCG_AMBER),
    "coverage": (COVERAGE_GREEN, COVERAGE_AMBER),
    "accuracy": (ANSWER_GREEN, ANSWER_AMBER),
    "completeness": (ANSWER_GREEN, ANSWER_AMBER),
    "relevance": (ANSWER_GREEN, ANSWER_AMBER),
}

ABOUT = """
# 📊 RAG Evaluation
How well does the assistant find the right information and answer correctly?
It is scored against a fixed test set of **150 questions** with known answers, in 7 categories
(direct facts, temporal, spanning several documents, comparative, numerical, relationships, holistic).

- **Retrieval** - do the retrieved chunks contain the expected keywords?
  **MRR** (how high the first hit ranks), **nDCG** (ranking quality), **keyword coverage** (share of keywords found).
- **Answers** - an LLM judge compares each answer with the reference answer and scores
  **accuracy**, **completeness** and **relevance** from 1 to 5.
"""


def get_color(value: float, metric_type: str) -> str:
    """Green / orange / red based on the metric's thresholds."""
    if metric_type not in THRESHOLDS:
        return "black"
    green, amber = THRESHOLDS[metric_type]
    if value >= green:
        return "green"
    if value >= amber:
        return "orange"
    return "red"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def message_html(text: str, color: str = "#999") -> str:
    return f"<div style='padding: 20px; text-align: center; color: {color};'>{html.escape(text)}</div>"


# ----------------------------------------------------------------------------
# Turning a summary (see evaluation.eval.summarize) into component values
# ----------------------------------------------------------------------------


def retrieval_html(summary: dict | None) -> str:
    if not summary:
        return message_html("No saved results yet.")
    r = summary["retrieval"]
    return (
        format_metric_html("Mean Reciprocal Rank (MRR)", r["mrr"], "mrr")
        + format_metric_html("Normalized DCG (nDCG)", r["ndcg"], "ndcg")
        + format_metric_html("Keyword Coverage", r["keyword_coverage"], "coverage", is_percentage=True)
    )


def answer_html(summary: dict | None) -> str:
    if not summary:
        return message_html("No saved results yet.")
    a = summary["answer"]
    return (
        format_metric_html("Accuracy", a["accuracy"], "accuracy", score_format=True)
        + format_metric_html("Completeness", a["completeness"], "completeness", score_format=True)
        + format_metric_html("Relevance", a["relevance"], "relevance", score_format=True)
    )


def run_info_md(summary: dict | None) -> str:
    if not summary:
        return (
            "### Latest full run\n"
            "*No saved results yet. Run `python -m evaluation.run` locally and commit "
            "`evaluation/results.json`.*"
        )
    m = summary["models"]
    provider = {"groq": "Groq", "gemini": "Gemini"}.get(m.get("judge_provider", "gemini"), m.get("judge_provider"))
    skipped = f", {summary['skipped']} skipped" if summary.get("skipped") else ""
    return (
        f"### {'Sample run' if summary.get('sample') else 'Latest full run'}: {summary['num_tests']} questions{skipped}\n"
        f"Run on **{summary['created_at'][:10]}** (took {summary.get('duration_s', 0) / 60:.0f} min) · "
        f"answer `{m['chat']}` · rewrite `{m['rewrite']}` · rerank `{m['rerank']}` · "
        f"embeddings `{m['embedding']}` · judge `{m['judge']}` ({provider})"
    )


def category_frames(summary: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = (summary or {}).get("by_category", [])
    mrr = pd.DataFrame([{"Category": c["category"], "Average MRR": c["mrr"]} for c in rows])
    accuracy = pd.DataFrame([{"Category": c["category"], "Average Accuracy": c["accuracy"]} for c in rows])
    return mrr, accuracy


def questions_frame(results: list[dict], with_answers: bool = False) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "Category": r["category"],
            "Question": r["question"],
            "MRR": round(r["mrr"], 2),
            "nDCG": round(r["ndcg"], 2),
            "Coverage %": round(r["keyword_coverage"], 1),
            "Accuracy": r["accuracy"],
            "Completeness": r["completeness"],
            "Relevance": r["relevance"],
        }
        if with_answers:
            row["Answer"] = r["answer"]
            row["Judge feedback"] = r["feedback"]
        rows.append(row)
    return pd.DataFrame(rows)


def saved_values(summary: dict | None) -> tuple:
    """Values for: info, retrieval cards, answer cards, MRR chart, accuracy chart, question table."""
    mrr_df, accuracy_df = category_frames(summary)
    return (
        run_info_md(summary),
        retrieval_html(summary),
        answer_html(summary),
        mrr_df,
        accuracy_df,
        questions_frame((summary or {}).get("results", [])),
    )


# ----------------------------------------------------------------------------
# Live check sizes and time estimate
# ----------------------------------------------------------------------------

LIVE_SIZES = (5, 10, 20, 50, 80, 100, 120, 150)


def live_size_choices(live_max: int | None) -> list[int]:
    """Dropdown sizes, capped at live_max (None = all)."""
    sizes = [n for n in LIVE_SIZES if live_max is None or n <= live_max]
    return sizes or [max(1, live_max or 1)]


def calls_per_question() -> dict[str, int]:
    """Model calls one evaluated question makes, per model (judge on Groq is keyed 'judge:...')."""
    calls: dict[str, int] = {}

    def add(model: str) -> None:
        calls[model] = calls.get(model, 0) + 1

    if config.USE_QUERY_REWRITE:
        add(config.REWRITE_MODEL)
    add(config.EMBEDDING_MODEL)
    if config.USE_RERANK:
        add(config.UTILITY_MODEL)
    add(config.CHAT_MODEL)
    judge_key = f"judge:{config.JUDGE_MODEL}" if config.JUDGE_PROVIDER == "groq" else config.JUDGE_MODEL
    add(judge_key)
    return calls


def questions_per_minute() -> float:
    """
    Rough throughput from the rate limits: the model with the least headroom
    (calls per question vs. its per-minute limit) sets the pace.
    """
    calls = calls_per_question()

    def rpm(model: str) -> int:
        return config.GROQ_RPM if model.startswith("judge:") else config.GEMINI_RPM

    by_limits = min(rpm(m) / c for m, c in calls.items())
    by_latency = 60 / 10 * max(1, config.EVAL_WORKERS)  # ~10 s per question per worker
    return max(0.5, min(by_limits, by_latency))


def live_estimate_md(size, counts_toward_limit: bool) -> str:
    try:
        n = int(size)
    except (TypeError, ValueError):
        return ""
    minutes = n / questions_per_minute()
    when = "under a minute" if minutes < 1 else f"about {minutes:.0f} min"
    limit = f" · uses {n} of today's {config.DAILY_QUESTION_LIMIT} demo questions" if counts_toward_limit and config.DAILY_QUESTION_LIMIT > 0 else ""
    return f"<small>⏱️ {when} · ~{n * sum(calls_per_question().values())} model calls{limit}</small>"


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------


def build_evaluation_page(
    spend: Callable[[int], bool] | None = None,
    allow_full_run: bool = False,
    live_max: int | None = None,
    concurrency_id: str | None = None,
) -> None:
    """
    Add the evaluation page's components to the current gr.Blocks (or route) context.

    spend(n)        - called before a live check of n questions; return False to refuse
                      (app.py passes its daily question budget). None = no limit.
    allow_full_run  - show the "Run full evaluation" button (local use only: a full run
                      costs ~750 Gemini calls).
    live_max        - largest live-check size offered in the dropdown (app.py passes
                      EVAL_LIVE_MAX so visitors can't start a 150-question run; a live
                      check holds the server's single processing slot, blocking the chat).
                      None = every size (local standalone dashboard).
    concurrency_id  - share a concurrency slot with other events (app.py shares one slot
                      with the chat, so only one pipeline runs on the server at a time).
    """
    sizes = live_size_choices(live_max)
    default_size = config.EVAL_LIVE_SAMPLE if config.EVAL_LIVE_SAMPLE in sizes else sizes[0]
    summary = load_results()
    info0, retrieval0, answer0, mrr0, accuracy0, table0 = saved_values(summary)

    gr.Markdown(ABOUT)

    # ---- Saved results of the latest full run ---------------------------------
    saved_info = gr.Markdown(info0)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### 🔍 Retrieval")
            saved_retrieval = gr.HTML(retrieval0)
        with gr.Column(scale=1):
            gr.Markdown("#### 💬 Answers")
            saved_answer = gr.HTML(answer0)
    with gr.Row():
        mrr_chart = gr.BarPlot(
            value=mrr0, x="Category", y="Average MRR", title="Average MRR by Category", y_lim=[0, 1], height=350
        )
        accuracy_chart = gr.BarPlot(
            value=accuracy0,
            x="Category",
            y="Average Accuracy",
            title="Average Accuracy by Category",
            y_lim=[1, 5],
            height=350,
        )
    with gr.Accordion("Scores for every question", open=False):
        saved_table = gr.Dataframe(value=table0, interactive=False, wrap=True)

    saved_outputs = [saved_info, saved_retrieval, saved_answer, mrr_chart, accuracy_chart, saved_table]

    # ---- Live check -----------------------------------------------------------
    gr.Markdown(
        "### 🧪 Live check\n"
        "Score random test questions **right now** with the live pipeline, to see the numbers above are real."
    )
    with gr.Row(equal_height=True):
        live_size = gr.Dropdown(
            choices=sizes,
            value=default_size,
            label="Number of questions",
            scale=1,
            min_width=160,
        )
        live_estimate = gr.Markdown(live_estimate_md(default_size, spend is not None))
    live_button = gr.Button("Run live check", variant="primary")
    live_metrics = gr.HTML(message_html("Results of the live check appear here."))
    live_table = gr.Dataframe(interactive=False, wrap=True, visible=False)

    def run_live_check(size, progress=gr.Progress()):
        try:
            n = int(size)
        except (TypeError, ValueError):
            n = default_size
        if n not in sizes:  # never trust the browser: enforce the cap on the server too
            return message_html(f"Please pick one of: {', '.join(map(str, sizes))}.", "#c0392b"), gr.update(visible=False)
        if spend is not None and not spend(n):
            return (
                message_html("The demo has reached its daily question limit. Please come back tomorrow!", "#c0392b"),
                gr.update(visible=False),
            )
        results, skipped = [], 0
        try:
            for _test, result, done, total in evaluate_many(random_tests(n)):
                progress(done / total, desc=f"Scored {done}/{total} questions")
                if result is None:
                    skipped += 1
                else:
                    results.append(result)
        except Exception as exc:  # never show a stack trace to visitors
            print(f"[evaluator] live check failed: {exc!r}")
        if not results:
            return (
                message_html("The live check couldn't run right now (model busy or quota used up). Try again later.", "#c0392b"),
                gr.update(visible=False),
            )
        live = summarize(results, skipped=skipped)
        note = f" ({skipped} skipped - model busy)" if skipped else ""
        cards = (
            "<div style='display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 0 16px;'>"
            f"<div>{retrieval_html(live)}</div><div>{answer_html(live)}</div></div>"
            f"<div style='text-align:center; color:#155724;'>✓ {len(results)} questions scored{note}</div>"
        )
        return cards, gr.update(value=questions_frame(live["results"], with_answers=True), visible=True)

    live_size.change(
        lambda n: live_estimate_md(n, spend is not None), inputs=live_size, outputs=live_estimate, queue=False
    )
    live_button.click(
        run_live_check,
        inputs=live_size,
        outputs=[live_metrics, live_table],
        concurrency_limit=1,
        concurrency_id=concurrency_id,
    )

    # ---- Full run (local only) ------------------------------------------------
    if allow_full_run:
        size = config.EVAL_SAMPLE_SIZE
        target = config.EVAL_SAMPLE_RESULTS_PATH if size else config.EVAL_RESULTS_PATH
        gr.Markdown(
            f"### ▶️ Run evaluation (local only)\n"
            f"Scores **{f'{size} questions spread over all categories' if size else 'all 150 questions'}** "
            f"(EVAL_SAMPLE_SIZE) and saves them to `{target.name}`. Questions already scored with the "
            f"same settings are reused from the cache. Command-line alternative: `python -m evaluation.run`."
        )
        full_button = gr.Button("Run evaluation", variant="secondary")

        def run_full(progress=gr.Progress()):
            results, skipped = [], 0
            tests = sample_tests(size or None)
            started = time.monotonic()
            for _test, result, done, total in evaluate_many(tests, cache=EvalCache()):
                progress(done / total, desc=f"Scored {done}/{total} questions")
                if result is None:
                    skipped += 1
                else:
                    results.append(result)
            if not results:
                values = list(saved_values(load_results()))
                values[0] = "### Latest full run\n**Every question failed - check your API key, model names and quota.**"
                return tuple(values)
            new_summary = summarize(
                results, skipped=skipped, duration_s=time.monotonic() - started, sample=bool(size)
            )
            save_results(new_summary, target)
            return saved_values(new_summary)

        full_button.click(run_full, outputs=saved_outputs, concurrency_limit=1)


def main():
    """Standalone local dashboard (with the full-run button)."""
    with gr.Blocks(title="RAG Evaluation Dashboard", **blocks_kwargs()) as app:
        build_evaluation_page(allow_full_run=True)
    app.queue(default_concurrency_limit=1)
    app.launch(**launch_kwargs())


if __name__ == "__main__":
    main()
