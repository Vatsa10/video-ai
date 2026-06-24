"""python -m videoqa.live -> live UI server on http://127.0.0.1:7860"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("videoqa.live.main:app", host="0.0.0.0", port=7860, ws="wsproto")
