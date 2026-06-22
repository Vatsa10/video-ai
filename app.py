"""HF Spaces entrypoint. Real app lives in videoqa/app.py.

HF runs `python app.py`, so launch must happen under __main__ and BLOCK — otherwise the
process exits right after binding the port and the Space hangs on "Starting".
"""
from videoqa.app import demo

if __name__ == "__main__":
    # .queue() required for ChatInterface; ssr_mode=False — Gradio 6 SSR wedges HF Spaces.
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
