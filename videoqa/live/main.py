"""Live UI server (FastAPI + WebSocket). Fork of edge-mission-control rewired to our CLIP
encoder, cloud captioner, and Edge store, plus an understanding panel and video upload.

Run: python -m videoqa.live   (uvicorn videoqa.live.main:app on :7860)
"""
import asyncio
import json
import logging
import threading
import uuid

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .captioner import Captioner
from .constants import STATIC_DIR, UPLOAD_DIR
from .detector import ObjectDetector
from .encoder import get_encoder
from .session import DemoSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class Hub:
    def __init__(self):
        self.sockets: set[WebSocket] = set()
        self.loop = None

    async def send_all(self, event: dict):
        dead = []
        for ws in self.sockets:
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.discard(ws)

    def emit(self, event: dict):
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.send_all(event), self.loop)


hub = Hub()
encoder = get_encoder()
detector = ObjectDetector()
captioner = Captioner()
session: DemoSession | None = None
demo_lock = asyncio.Lock()
demo_running = False
current_video = None  # path of the most recently uploaded video


@app.on_event("startup")
async def startup():
    hub.loop = asyncio.get_running_loop()

    def _warm():
        # Off the startup path so the port binds instantly (HF health check passes);
        # models download/load in the background, ready before the first Analyze.
        try:
            logger.info("Warming models ...")
            encoder.warm()
            detector.warm()
            logger.info("Models ready.")
        except Exception as e:
            print(f"[warm] {type(e).__name__}: {e}")

    threading.Thread(target=_warm, daemon=True).start()


@app.post("/upload")
async def upload(file: UploadFile):
    global current_video
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}.mp4"
    dest.write_bytes(await file.read())
    current_video = dest
    return JSONResponse({"ok": True})


async def run_demo(mode: str):
    global demo_running, session
    async with demo_lock:
        if demo_running or current_video is None:
            return
        demo_running = True
    try:
        if session is not None:
            session.shutdown()
        session = DemoSession(encoder, detector, captioner, hub.emit)
        await session.start(current_video, boot_delay=0.55)
        await asyncio.sleep(2.0)
        await session.warm_query()
        await asyncio.to_thread(session.pipeline.thread.join)  # play to the end
        await session.emit_understanding()  # narrative summary + timeline
    finally:
        demo_running = False


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    hub.sockets.add(ws)
    await ws.send_text(json.dumps({"type": "ready", "running": demo_running}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start" and not demo_running:
                asyncio.create_task(run_demo(msg.get("mode", "interactive")))
            elif cmd == "query" and session is not None:
                text = (msg.get("text") or "").strip()[:120]
                cls = (msg.get("cls") or "").strip()[:60] or None
                if text:
                    asyncio.create_task(session.run_query(text, cls=cls))
            elif cmd == "label" and session is not None:
                text = (msg.get("text") or "").strip()[:60]
                if text:
                    asyncio.create_task(session.teach(text))
    except WebSocketDisconnect:
        hub.sockets.discard(ws)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/video")
async def video():
    if current_video is None:
        return JSONResponse({"error": "no video uploaded"}, status_code=404)
    return FileResponse(current_video, media_type="video/mp4")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
