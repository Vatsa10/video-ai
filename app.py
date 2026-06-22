"""HF Spaces entrypoint. Real app lives in videoqa/app.py."""
from videoqa.app import demo

# ssr_mode=False: Gradio 6 SSR (Node proxy) can wedge HF Spaces at "Starting".
demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
