"""
Insurellm Expert Assistant - Gradio chat UI for the advanced RAG pipeline.

Run from the project root:
    python app.py
"""

import gradio as gr

from gradio_compat import blocks_kwargs, chatbot_kwargs, launch_kwargs
from pro_implementation.answer import answer_question, message_text
from pro_implementation.llm import MissingAPIKeyError


def format_context(context) -> str:
    if not context:
        return "*No context retrieved.*"
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += f"<span style='color: #ff7800;'>Source: {doc.metadata.get('source', 'unknown')}</span>\n\n"
        result += doc.page_content + "\n\n"
    return result


def chat(history):
    last_message = message_text(history[-1]["content"])
    prior = history[:-1]
    try:
        answer, context = answer_question(last_message, prior)
    except MissingAPIKeyError as e:
        answer, context = f"⚠️ {e}", []
    except Exception as e:  # show the error in the chat instead of hanging
        answer, context = f"⚠️ Something went wrong: {type(e).__name__}: {e}", []
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Insurellm Expert Assistant", **blocks_kwargs()) as ui:
        gr.Markdown("# 🏢 Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(label="💬 Conversation", height=600, **chatbot_kwargs())
                message = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
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
        
    return ui


def main():
    build_ui().launch(**launch_kwargs())


if __name__ == "__main__":
    main()
