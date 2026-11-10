"""
Small helpers so the Gradio UIs run on both Gradio 5.x and 6.x.

Gradio 6 removed Chatbot(type=..., show_copy_button=...) and moved `theme` from
gr.Blocks(...) to .launch(...). These helpers pick the right arguments.
"""

import os

import gradio as gr

GRADIO_MAJOR = int(gr.__version__.split(".")[0])
THEME = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])


def chatbot_kwargs() -> dict:
    if GRADIO_MAJOR >= 6:
        return {"buttons": ["copy"]}
    return {"type": "messages", "show_copy_button": True}


def blocks_kwargs() -> dict:
    return {} if GRADIO_MAJOR >= 6 else {"theme": THEME}


def launch_kwargs() -> dict:
    """
    Locally: host/port come from Gradio's own env vars (GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT), default 127.0.0.1:7860, and a browser tab opens.

    On a host such as Render, the platform sets PORT and routes public traffic to it.
    The app must then listen on 0.0.0.0:$PORT (not localhost) and must not try to
    open a browser. OPEN_BROWSER=false also disables the browser tab explicitly.
    """
    port = os.getenv("PORT")
    if port:  # running on a hosting platform
        kwargs = {"server_name": "0.0.0.0", "server_port": int(port), "inbrowser": False}
    else:
        kwargs = {"inbrowser": os.getenv("OPEN_BROWSER", "true").lower() in {"1", "true", "yes"}}
    if GRADIO_MAJOR >= 6:
        kwargs["theme"] = THEME
    return kwargs
