import gradio as gr
from core import get_answer
import os

def get_answer_app(video_link, question):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before starting the application.")
    return get_answer(api_key, video_link, question)

iface = gr.Interface(
    fn=get_answer_app,
    inputs=["text", "text"],
    outputs=["text", "text"],
)

iface.queue().launch()
