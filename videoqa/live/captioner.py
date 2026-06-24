"""Async object captioning for the live UI — our cloud LLM instead of Florence-2.

LIFO worker thread: the most recently discovered object is captioned first, so fresh
discoveries become text-searchable while the trail of older ones drains. Same interface the
demo's session/registry expect (submit / start / stop / warm / on_caption callback).
"""
import base64
import io
import logging
import queue
import threading

from ..caption import _MODEL, _PROMPT, _client

logger = logging.getLogger(__name__)


class Captioner:
    def __init__(self):
        self.jobs: queue.LifoQueue = queue.LifoQueue()
        self.on_caption = None  # set by session: callback(obj_id, caption)
        self.is_running = False
        self.thread = None

    def load(self):
        pass  # cloud — nothing to load

    def warm(self):
        pass

    def caption(self, image) -> str:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=80)
        url = "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()
        resp = _client.chat.completions.create(
            model=_MODEL,
            max_tokens=60,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()

    def submit(self, obj_id: str, crop):
        self.jobs.put((obj_id, crop))

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        self.jobs.put(None)

    def pending(self) -> int:
        return self.jobs.qsize()

    def _worker(self):
        while self.is_running:
            job = self.jobs.get()
            if job is None:
                continue
            obj_id, crop = job
            try:
                text = self.caption(crop)
            except Exception:
                logger.exception("Caption failed for %s", obj_id)
                continue
            if self.on_caption is not None:
                try:
                    self.on_caption(obj_id, text)
                except Exception:
                    logger.exception("Caption callback failed for %s", obj_id)
