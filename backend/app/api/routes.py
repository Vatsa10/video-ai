import uuid
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config import settings
from ..core.edit_plan import generate_edit_plan
from ..core.pipeline import run_pipeline
from ..models.edit_plan import EditPlan, EditPlanRequest
from ..models.features import VideoFeatures
from ..services import cache, ffmpeg, storage

router = APIRouter()

JOBS: dict[str, dict] = {}

SceneCardMode = Literal["light", "full", "none"]


def _shape_scene_card(vf: VideoFeatures, mode: SceneCardMode) -> VideoFeatures:
    if mode == "light":
        return vf  # already attached light variant by pipeline
    if mode == "none":
        for seg in vf.timeline:
            seg.scene_card = None
        return vf
    # mode == "full" — rebuild full variant per segment
    from analysis.scene_card import build_full
    for i, seg in enumerate(vf.timeline):
        # adapt pydantic seg → dataclass-like view for build_full
        class _S:  # minimal shim
            pass
        shim = _S()
        shim.t0, shim.t1 = seg.t0, seg.t1
        shim.features = seg
        shim.scores = type("S", (), {"highlight": seg.highlight, "energy": seg.energy})()
        shim.tags = seg.tags
        shim.decisions = seg.decisions
        shim.transcript = seg.transcript
        try:
            seg.scene_card = build_full(shim, vf.video_id, i)
        except Exception:
            pass
    return vf


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/analyze", response_model=VideoFeatures)
async def analyze(
    file: UploadFile = File(...),
    faces: bool = Query(default=settings.ENABLE_FACES),
    objects: bool = Query(default=settings.ENABLE_OBJECTS),
    embeddings: bool = Query(default=True),
    clip_zeroshot: bool = Query(default=True),
    camera_motion: bool = Query(default=True),
    ocr: bool = Query(default=True),
    quality: bool = Query(default=True),
    dedup: bool = Query(default=True),
    asr: bool = Query(default=settings.ENABLE_ASR),
    pose: bool = Query(default=True),
    saliency: bool = Query(default=True),
    depth: bool = Query(default=True),
    captions: bool = Query(default=True),
    action: bool = Query(default=True),
    tracking: bool = Query(default=True),
    narrative: bool = Query(default=True),
    narrative_polish: bool = Query(default=False),
    include_scene_card: SceneCardMode = Query(default="light"),
):
    try:
        _, path = storage.save_upload(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        vf = run_pipeline(
            str(path),
            enable_faces=faces, enable_objects=objects,
            enable_embeddings=embeddings, enable_asr=asr,
            enable_clip_zeroshot=clip_zeroshot, enable_camera_motion=camera_motion,
            enable_ocr=ocr, enable_quality=quality, enable_dedup=dedup,
            enable_pose=pose, enable_saliency=saliency, enable_depth=depth,
            enable_captions=captions, enable_action=action, enable_tracking=tracking,
            enable_narrative=narrative, narrative_polish=narrative_polish,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"pipeline failed: {e}")
    return _shape_scene_card(vf, include_scene_card)


@router.post("/analyze/async")
async def analyze_async(
    bg: BackgroundTasks,
    file: UploadFile = File(...),
    faces: bool = Query(default=settings.ENABLE_FACES),
    objects: bool = Query(default=settings.ENABLE_OBJECTS),
    embeddings: bool = Query(default=True),
    clip_zeroshot: bool = Query(default=True),
    camera_motion: bool = Query(default=True),
    ocr: bool = Query(default=True),
    quality: bool = Query(default=True),
    dedup: bool = Query(default=True),
    asr: bool = Query(default=settings.ENABLE_ASR),
    pose: bool = Query(default=True),
    saliency: bool = Query(default=True),
    depth: bool = Query(default=True),
    captions: bool = Query(default=True),
    action: bool = Query(default=True),
    tracking: bool = Query(default=True),
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
            vf = run_pipeline(
                str(path),
                enable_faces=faces, enable_objects=objects,
                enable_embeddings=embeddings, enable_asr=asr,
                enable_clip_zeroshot=clip_zeroshot, enable_camera_motion=camera_motion,
                enable_ocr=ocr, enable_quality=quality, enable_dedup=dedup,
                enable_pose=pose, enable_saliency=saliency, enable_depth=depth,
                enable_captions=captions, enable_action=action, enable_tracking=tracking,
            )
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
def get_features(
    video_id: str,
    include_scene_card: SceneCardMode = Query(default="light"),
):
    vf = cache.load_features(video_id)
    if not vf:
        raise HTTPException(404, "video_id not found")
    return _shape_scene_card(vf, include_scene_card)


@router.get("/narrative/{video_id}")
def get_narrative(
    video_id: str,
    style: Literal["paragraph", "bullets", "scenes", "summary", "all"] = Query(default="paragraph"),
):
    vf = cache.load_features(video_id)
    if not vf:
        raise HTTPException(404, "video_id not found")
    if style == "paragraph":
        return {"video_id": video_id, "paragraph": vf.narrative}
    if style == "bullets":
        return {"video_id": video_id, "bullets": vf.narrative_bullets}
    if style == "scenes":
        return {"video_id": video_id, "scenes": [s.model_dump() for s in vf.narrative_scenes]}
    if style == "summary":
        return {"video_id": video_id, "summary": vf.narrative_summary or vf.narrative[:500]}
    return {
        "video_id": video_id,
        "paragraph": vf.narrative,
        "summary": vf.narrative_summary,
        "bullets": vf.narrative_bullets,
        "scenes": [s.model_dump() for s in vf.narrative_scenes],
    }


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
            "POST /analyze[?include_scene_card=light|full|none]",
            "POST /analyze/async",
            "GET  /jobs/{job_id}",
            "GET  /features/{video_id}[?include_scene_card=...]",
            "POST /edit-plan",
            "POST /render",
            "GET  /health",
        ],
    })
