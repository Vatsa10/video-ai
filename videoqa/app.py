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
from .cleanup import sweep, track, wipe, wipe_all
from .ingest import ingest
from .store import load_understanding
from .understand import to_markdown

TTL_SECONDS = 1800  # data older than this is wiped (idle/orphaned sessions)


def _maintenance():
    """Startup wipe + periodic TTL sweep, all OFF the import/launch path.

    Runs in a daemon thread so module import returns instantly and the server binds the
    port fast (HF health check passes). Chroma calls here also stop spawning event loops
    on the main thread (the source of the harmless asyncio __del__ noise at startup).
    """
    try:
        wipe_all()
    except Exception as e:
        print(f"[startup] wipe_all skipped: {type(e).__name__}: {e}")
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
        return None, "Upload a video first.", ""
    vid = _video_id(video_path)
    n = ingest(video_path, vid, float(interval))
    track(vid)
    understanding = to_markdown(load_understanding(vid))
    return vid, f"Ready — {n} keyframes analyzed. Ask away below.", understanding


def chat(message, history, video_id):
    if not video_id:
        return "Process a video first (top panel)."
    return ask(video_id, message)


with gr.Blocks(title="videoqa") as demo:
    gr.Markdown(
        "# videoqa — video understanding pipeline\n"
        "Upload a video → it's analyzed frame-by-frame and synthesized into a "
        "structured understanding. Then explore it or ask questions. "
        "_Data is erased when you leave._"
    )
    vid_state = gr.State(None)

    with gr.Row():
        video = gr.Video(label="Video")
        with gr.Column():
            interval = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="Seconds between frames")
            go = gr.Button("Analyze", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

    understanding_md = gr.Markdown("_Analyze a video to see its understanding._")

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[vid_state],
        title="Ask",
    )

    go.click(process, [video, interval, vid_state], [vid_state, status, understanding_md])
    demo.unload(lambda: sweep(TTL_SECONDS))  # fires on tab/session close

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
