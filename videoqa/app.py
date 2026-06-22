"""Gradio web UI: drop a video, process it, chat with it. Ephemeral — data is wiped
when the user leaves (browser/tab close) and by a TTL sweep for anything orphaned.

  python -m videoqa.app    (opens http://127.0.0.1:7860)
"""
import hashlib
import threading
import time
from pathlib import Path

import gradio as gr

from .ask import ask
from .cleanup import sweep, track, wipe
from .ingest import ingest
from .store import load_understanding
from .understand import details_md, summary_md

TTL_SECONDS = 1800  # data older than this is wiped (idle/orphaned sessions)


def _maintenance():
    """Warm the embedding models, then run the periodic TTL sweep. All off the launch path.

    Daemon thread so module import returns instantly and the port binds fast. Warming CLIP
    + bge here means the model download/load happens during boot, not on the user's first
    Analyze. No startup Chroma call (`_seen` empty -> sweep() is a no-op until a video runs).
    """
    try:
        from .embed import warm

        warm()  # pay the model load now, while the user is still uploading
    except Exception as e:
        print(f"[warm] {type(e).__name__}: {e}")
    while True:
        time.sleep(300)
        try:
            sweep(TTL_SECONDS)
        except Exception as e:
            print(f"[sweeper] {type(e).__name__}: {e}")


threading.Thread(target=_maintenance, daemon=True).start()


def _video_id(path: str) -> str:
    h = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]
    return f"v_{h}"


def process(video_path, interval, prev_id):
    sweep(TTL_SECONDS)  # opportunistic TTL cleanup of orphaned sessions
    if prev_id:
        wipe(prev_id)  # user loaded a new video — drop the old one's data
    if not video_path:
        return None, "Upload a video first.", summary_md({}), "", gr.update()
    vid = _video_id(video_path)
    n = ingest(video_path, vid, float(interval))
    track(vid)
    u = load_understanding(vid)
    # open the chat accordion now that there's something to ask about
    return vid, f"Analyzed {n} keyframes.", summary_md(u), details_md(u), gr.update(open=True)


def chat(message, history, video_id):
    if not video_id:
        return "Analyze a video first (top panel)."
    return ask(video_id, message)


with gr.Blocks(title="videoqa") as demo:
    gr.Markdown(
        "# 🎬 videoqa — understand any video\n"
        "Upload a video and it tells you **what happens in it** — synthesized from "
        "frame-by-frame vision analysis and the spoken audio. _Data is erased when you leave._"
    )
    vid_state = gr.State(None)

    with gr.Row():
        video = gr.Video(label="Video")
        with gr.Column():
            interval = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="Seconds between frames")
            go = gr.Button("Analyze", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

    # Hero: the narrative of what happened
    summary_view = gr.Markdown(summary_md({}))

    # Supporting structure
    details_view = gr.Markdown("")

    # Asking is secondary — collapsed by default, auto-opens after Analyze
    with gr.Accordion("💬 Ask about specific moments", open=False) as chat_acc:
        gr.ChatInterface(fn=chat, additional_inputs=[vid_state])

    go.click(
        process,
        [video, interval, vid_state],
        [vid_state, status, summary_view, details_view, chat_acc],
    )
    demo.unload(lambda: sweep(TTL_SECONDS))  # fires on tab/session close

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
