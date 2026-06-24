"""Config for the live UI subsystem (forked from edge-mission-control, rewired to our
CLIP encoder + cloud captioner + Edge store). 512-d CLIP, ephemeral storage under storage/live.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent  # videoqa/live (holds static/)
STATIC_DIR = PROJECT_ROOT / "static"

DATA_DIR = Path("storage/live")
SHARD_DIR = DATA_DIR / "shard"
THUMBS_DIR = DATA_DIR / "thumbs"
UPLOAD_DIR = DATA_DIR / "uploads"

# CLIP (clip-ViT-B-32) — our embed.py models
VECTOR_DIMENSION = 512
VECTOR_NAME = "vision"
SPARSE_VECTOR_NAME = "caption"

INGEST_FPS = 3.0  # detection + frame-embed ticks per second of video time

# --- detector (YOLOE-11L open-vocab + BoT-SORT) ---
DETECTOR_WEIGHTS = "yoloe-11l-seg.pt"
DETECTOR_VOCAB = [
    "person", "sofa", "armchair", "dining chair", "bar stool", "bench", "ottoman",
    "coffee table", "dining table", "desk", "nightstand", "dresser", "cabinet",
    "bookshelf", "bed", "pillow", "blanket", "rug", "floor lamp", "table lamp",
    "framed picture", "mirror", "houseplant", "vase", "television", "laptop",
    "phone", "keyboard", "monitor", "cup", "bottle", "bowl", "tray", "book", "bag",
    "backpack", "box", "remote", "glasses", "headphones", "camera", "shoes", "clock",
    "wine bottle", "wine glass", "refrigerator", "oven", "microwave", "sink", "stove",
    "bathtub", "toilet", "towel", "window", "door", "car",
]
DETECT_CONF = 0.32
DETECT_IMGSZ = 640
DETECT_MAX_DET = 32
MAX_VOCAB = 80

CONFIRM_SIGHTINGS = 3
REID_THRESHOLD = 0.90
MIN_BOX_AREA = 0.004
MAX_CROP_EMBEDS_PER_TICK = 4
CROP_PAD = 0.22

THUMB_WIDTH = 320
CROP_THUMB_WIDTH = 192
JPEG_QUALITY = 72

OBJECT_SEARCH_LIMIT = 4
MOMENT_SEARCH_LIMIT = 3
RRF_K = 60
MMR_LAMBDA = 0.9
MMR_MAX_CANDIDATES = 100
WEAK_OBJECT_SCORE = 0.10
