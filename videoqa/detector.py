"""Open-vocabulary object detection + tracking (YOLOE-11L-seg + BoT-SORT).

Ported from edge-mission-control/app/detector.py, trimmed for our offline ingest.
Heavy imports (torch, ultralytics) are lazy — only loaded when object mode is enabled,
so the default pipeline deploys without ultralytics installed.
"""
import os
import threading

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # no pip calls at runtime

DETECTOR_WEIGHTS = "yoloe-11l-seg.pt"
DETECT_CONF = 0.32
DETECT_IMGSZ = 640
DETECT_MAX_DET = 32
MIN_BOX_AREA = 0.004  # drop boxes < 0.4% of frame
# Generic indoor/object vocabulary; extend for your domain.
DETECTOR_VOCAB = [
    "person", "chair", "sofa", "table", "lamp", "television", "laptop", "phone",
    "cup", "bottle", "bowl", "plate", "book", "bag", "backpack", "box", "remote",
    "keyboard", "mouse", "monitor", "plant", "potted plant", "clock", "picture frame",
    "shoes", "glasses", "headphones", "camera", "bed", "pillow", "door", "window",
    "refrigerator", "microwave", "oven", "sink", "toilet", "bathtub", "towel", "car",
]


class Detection:
    __slots__ = ("track_id", "cls", "conf", "box")

    def __init__(self, track_id: int, cls: str, conf: float, box: tuple):
        self.track_id = track_id
        self.cls = cls
        self.conf = conf
        self.box = box  # (x1, y1, x2, y2) normalized to [0, 1]


class ObjectDetector:
    def __init__(self):
        self.model = None
        self.device = None
        self.vocab = list(DETECTOR_VOCAB)
        self._lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return
        import torch
        from ultralytics import YOLO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(DETECTOR_WEIGHTS)  # downloads on first run
        self._apply_vocab(self.vocab)

    def _apply_vocab(self, vocab):
        import torch

        with torch.inference_mode():
            self.model.set_classes(vocab, self.model.get_text_pe(vocab))
        self.vocab = vocab

    def track(self, frame_bgr) -> list[Detection]:
        """Detect + track one frame -> list of normalized Detections."""
        h, w = frame_bgr.shape[:2]
        result = self.model.track(
            frame_bgr,
            device=self.device,
            conf=DETECT_CONF,
            imgsz=DETECT_IMGSZ,
            max_det=DETECT_MAX_DET,
            persist=True,
            verbose=False,
        )[0]

        out = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for tid, c, cf, xyxy in zip(boxes.id, boxes.cls, boxes.conf, boxes.xyxy):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                box = (x1 / w, y1 / h, x2 / w, y2 / h)
                if (box[2] - box[0]) * (box[3] - box[1]) < MIN_BOX_AREA:
                    continue
                out.append(Detection(int(tid), result.names[int(c)], float(cf), box))
        return out
