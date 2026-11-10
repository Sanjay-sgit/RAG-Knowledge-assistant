"""
Insurellm Expert Assistant - Gradio chat UI for the advanced RAG pipeline.

Run from the project root:
    python app.py

The same file runs on a hosting platform (e.g. Render): when the platform sets PORT,
gradio_compat.launch_kwargs() makes the app listen on 0.0.0.0:$PORT.

Public-demo protection (all tunable via env vars, see pro_implementation/config.py):
- one question is answered at a time; at most QUEUE_MAX_SIZE visitors wait in line
- questions longer than MAX_QUESTION_CHARS are rejected before any Gemini call
- earlier chat messages are trimmed to MAX_HISTORY_MESSAGE_CHARS
- at most DAILY_QUESTION_LIMIT questions per day for the whole site
- visitors see short, friendly errors; full details go to the server log
"""

import datetime
import logging
import threading
import time

import gradio as gr
from google.genai import errors as genai_errors

from gradio_compat import blocks_kwargs, chatbot_kwargs, launch_kwargs
from pro_implementation import config
from pro_implementation.answer import answer_question, message_text
from pro_implementation.llm import MissingAPIKeyError

log = logging.getLogger("app")


# ----------------------------------------------------------------------------
# Daily question budget (whole site). In-memory: it also resets when the server
# restarts, which is fine for a single-instance demo.
# ----------------------------------------------------------------------------


class DailyBudget:
    def __init__(self, limit: int):
        self.limit = limit
        self._lock = threading.Lock()
        self._day = None
        self._used = 0

    def try_spend(self) -> bool:
        """Count one question. False if today's limit is already reached."""
        if self.limit <= 0:  # 0 = unlimited
            return True
        with self._lock:
            today = datetime.datetime.now(datetime.timezone.utc).date()
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True


budget = DailyBudget(config.DAILY_QUESTION_LIMIT)


# ----------------------------------------------------------------------------
# Chat callbacks
# ----------------------------------------------------------------------------


def format_context(context) -> str:
    if not context:
        return "*No context retrieved.*"
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata.get('source', 'unknown')}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def trim_history(messages: list[dict]) -> list[dict]:
    """Cap the length of earlier messages (the browser sends the whole chat back each time)."""
    limit = config.MAX_HISTORY_MESSAGE_CHARS
    return [
        {"role": m.get("role", "user"), "content": message_text(m.get("content", ""))[:limit]}
        for m in messages
    ]


def get_answer(question: str, prior: list[dict]) -> tuple[str, list]:
    """Apply the demo limits, then run the RAG pipeline. Never raises."""
    if not question:
        return "Please ask a question about Insurellm.", []
    if len(question) > config.MAX_QUESTION_CHARS:
        return (
            f"⚠️ Please keep your question under {config.MAX_QUESTION_CHARS} characters "
            f"(yours has {len(question)}).",
            [],
        )
    if not budget.try_spend():
        log.warning("daily question limit (%s) reached", config.DAILY_QUESTION_LIMIT)
        return (
            "⚠️ This demo has reached its daily question limit (it runs on a free Gemini quota). "
            "Please come back tomorrow!",
            [],
        )

    started = time.monotonic()
    try:
        answer, context = answer_question(question, trim_history(prior))
        log.info("answered in %.1fs", time.monotonic() - started)  # no question text in logs
        return answer, context
    except MissingAPIKeyError:
        log.exception("no Gemini API key configured")
        return "⚠️ The server has no Gemini API key configured yet.", []
    except genai_errors.ClientError as e:
        log.exception("Gemini client error")
        if getattr(e, "code", None) == 429:
            return (
                "⚠️ The demo's free Gemini quota is used up right now. Please try again later.",
                [],
            )
        return "⚠️ Sorry, the language model rejected this request. Please try again.", []
    except Exception:
        log.exception("unexpected error while answering")
        return "⚠️ Sorry, something went wrong while answering. Please try again.", []


def chat(history):
    question = message_text(history[-1]["content"]).strip()
    answer, context = get_answer(question, history[:-1])
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Insurellm Expert Assistant", **blocks_kwargs()) as ui:
        gr.Markdown(
            "# 🏢 Insurellm Expert Assistant\n"
            "Ask me anything about Insurellm, a fictional insurance-tech company "
            "(employees, products, contracts).\n\n"
            "<small>Demo app: questions are sent to Google's Gemini API, so please don't "
            "enter personal information.</small>"
        )

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="💬 Conversation", height=600, **chatbot_kwargs())
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                    max_length=config.MAX_QUESTION_CHARS,
                )

            with gr.Column(scale=1):
                context_markdown = gr.Markdown(
                    label="📚 Retrieved Context",
                    value="*Retrieved context will appear here*",
                    container=True,
                    height=600,
                )

        message.submit(
            put_message_in_chatbot, inputs=[message, chatbot], outputs=[message, chatbot]
        ).then(chat, inputs=chatbot, outputs=[chatbot, context_markdown])

    # One answer at a time (protects the free per-minute quota); a bounded waiting line.
    ui.queue(max_size=config.QUEUE_MAX_SIZE, default_concurrency_limit=1)
    return ui


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    build_ui().launch(**launch_kwargs())


if __name__ == "__main__":
    main()
