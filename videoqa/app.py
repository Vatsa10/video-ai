"""Gradio web UI: drop a video, process it, then chat with it.

  python -m videoqa.app    (opens http://127.0.0.1:7860)
"""
import hashlib
from pathlib import Path

import gradio as gr

from .ask import ask
from .ingest import ingest


def _video_id(path: str) -> str:
    # content hash -> same video reuses its index, different videos don't collide
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]
    return f"v_{h}"


def process(video_path, interval):
    if not video_path:
        return None, "Upload a video first."
    vid = _video_id(video_path)
    n = ingest(video_path, vid, float(interval))
    return vid, f"Ready — {n} keyframes indexed. Ask away below."


def chat(message, history, video_id):
    if not video_id:
        return "Process a video first (top panel)."
    return ask(video_id, message)


with gr.Blocks(title="videoqa") as demo:
    gr.Markdown("# videoqa — ask questions about any video")
    vid_state = gr.State(None)

    with gr.Row():
        video = gr.Video(label="Video")
        with gr.Column():
            interval = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="Seconds between frames")
            go = gr.Button("Process", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[vid_state],
        type="messages",
        title="Chat",
    )

    go.click(process, [video, interval], [vid_state, status])

if __name__ == "__main__":
    demo.launch()
