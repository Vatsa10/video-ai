"""Gradio web UI: drop a video, process it, chat with it. Ephemeral — data is wiped
when the user leaves (browser/tab close) and by a TTL sweep for anything orphaned.

  python -m videoqa.app    (opens http://127.0.0.1:7860)
"""
import hashlib
import threading
from pathlib import Path

import gradio as gr

from .ask import ask
from .cleanup import sweep, track, wipe, wipe_all
from .ingest import ingest

TTL_SECONDS = 1800  # data older than this is wiped (idle/orphaned sessions)

wipe_all()  # clean slate on startup — no leftovers from a previous run


def _sweeper():
    sweep(TTL_SECONDS)
    threading.Timer(300, _sweeper).start()  # re-check every 5 min, even with no traffic


_sweeper()


def _video_id(path: str) -> str:
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]
    return f"v_{h}"


def process(video_path, interval, prev_id):
    sweep(TTL_SECONDS)  # opportunistic TTL cleanup of orphaned sessions
    if prev_id:
        wipe(prev_id)  # user loaded a new video — drop the old one's data
    if not video_path:
        return None, "Upload a video first."
    vid = _video_id(video_path)
    n = ingest(video_path, vid, float(interval))
    track(vid)
    return vid, f"Ready — {n} keyframes indexed. Ask away below."


def chat(message, history, video_id):
    if not video_id:
        return "Process a video first (top panel)."
    return ask(video_id, message)


with gr.Blocks(title="videoqa") as demo:
    gr.Markdown("# videoqa — ask questions about any video\n_Data is erased when you leave._")
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

    go.click(process, [video, interval, vid_state], [vid_state, status])
    demo.unload(lambda: sweep(TTL_SECONDS))  # fires on tab/session close

if __name__ == "__main__":
    demo.launch()
