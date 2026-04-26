import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config import settings
from ..core.edit_plan import generate_edit_plan
from ..core.pipeline import run_pipeline
from ..models.edit_plan import EditPlan, EditPlanRequest
from ..models.features import VideoFeatures
from ..services import cache, ffmpeg, storage

router = APIRouter()

# in-memory job table for async analyze
JOBS: dict[str, dict] = {}


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/analyze", response_model=VideoFeatures)
async def analyze(
    file: UploadFile = File(...),
    faces: bool = Query(default=settings.ENABLE_FACES),
    objects: bool = Query(default=settings.ENABLE_OBJECTS),
    embeddings: bool = Query(default=settings.ENABLE_EMBEDDINGS),
    asr: bool = Query(default=settings.ENABLE_ASR),
):
    try:
        _, path = storage.save_upload(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        vf = run_pipeline(
            str(path),
            enable_faces=faces,
            enable_objects=objects,
            enable_embeddings=embeddings,
            enable_asr=asr,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")
    return vf


@router.post("/analyze/async")
async def analyze_async(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    faces: bool = Query(default=settings.ENABLE_FACES),
    objects: bool = Query(default=settings.ENABLE_OBJECTS),
    embeddings: bool = Query(default=settings.ENABLE_EMBEDDINGS),
    asr: bool = Query(default=settings.ENABLE_ASR),
):
    try:
        _, path = storage.save_upload(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "video_id": None, "error": None}

    def _work():
        JOBS[job_id]["status"] = "running"
        try:
            vf = run_pipeline(str(path), enable_faces=faces, enable_objects=objects,
                              enable_embeddings=embeddings, enable_asr=asr)
            JOBS[job_id].update(status="done", video_id=vf.video_id)
        except Exception as e:
            JOBS[job_id].update(status="failed", error=str(e))

    bg.add_task(_work)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    return j


@router.get("/features/{video_id}", response_model=VideoFeatures)
def get_features(video_id: str):
    vf = cache.load_features(video_id)
    if not vf:
        raise HTTPException(404, "video_id not found")
    return vf


@router.post("/edit-plan", response_model=EditPlan)
def edit_plan(req: EditPlanRequest):
    vf = cache.load_features(req.video_id)
    if not vf:
        raise HTTPException(404, "video_id not found; run /analyze first")
    return generate_edit_plan(vf, req.segments, mode=req.mode)


@router.post("/render")
def render(req: EditPlanRequest):
    vf = cache.load_features(req.video_id)
    if not vf:
        raise HTTPException(404, "video_id not found")
    plan = generate_edit_plan(vf, req.segments, mode=req.mode)

    src = cache.source_video_for(req.video_id)
    if not src:
        raise HTTPException(500, "normalized video not in cache")

    out_dir = settings.OUTPUT_DIR / req.video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{req.mode}.mp4"
    try:
        ffmpeg.cut_and_concat(str(src), plan.final_segments, str(dst))
    except Exception as e:
        raise HTTPException(500, f"render failed: {e}")
    return FileResponse(str(dst), media_type="video/mp4", filename=dst.name)


@router.get("/")
def root():
    return JSONResponse({
        "service": "video-ai backend",
        "endpoints": [
            "POST /analyze",
            "POST /analyze/async",
            "GET  /jobs/{job_id}",
            "GET  /features/{video_id}",
            "POST /edit-plan",
            "POST /render",
            "GET  /health",
        ],
    })
