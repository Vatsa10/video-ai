"""Entry point — the live FastAPI + WebSocket UI.

  python app.py            # or: python -m videoqa.live
HF Spaces uses the Dockerfile (sdk: docker) to run uvicorn on port 7860.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("videoqa.live.main:app", host="0.0.0.0", port=7860, ws="wsproto")
