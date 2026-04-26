"""End-to-end pipeline. No hardcoded thresholds — quality, OCR gating, camera motion
all consume per-video percentile stats from `features.adaptive`.

Two-stage execution:
  Stage A (parallel): visual flow + audio + faces + objects(+detections)
                     + embeddings + clip_zeroshot + ASR
  Stage B (post-stats serial-in-parallel): camera_motion, shot_type, quality,
                     OCR (gated), pose, saliency, depth, captions, action,
                     tracking. All driven by VideoStats from Stage A outputs.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .preprocess import (
    cache_dir, extract_audio, normalize, probe, video_id_for,
)
from .segmentation import segments_for
from .features.adaptive import compute as compute_stats
from .features.visual import visual_per_segment
from .features.audio import audio_per_segment
from .features.faces import faces_per_segment
from .features.objects import objects_per_segment
from .features.embeddings import embeddings_per_segment
from .features.transcript import transcribe, assign_words_to_segments
from .features.camera_motion import camera_motion_per_segment
from .features.shot_type import shot_type_per_segment
from .features.quality import quality_per_segment
from .features.clip_zeroshot import clip_zeroshot_per_segment
from .features.fusion import fallback_scene_category
from .features.ocr import ocr_per_segment
from .features.pose import pose_per_segment
from .features.saliency import saliency_per_segment
from .features.depth import depth_per_segment
from .features.captions import captions_per_segment
from .features.action import action_per_segment
from .features.tracking import tracking_per_segment
from .scoring import attach_fusion_tags, score_segment, select_highlights, tag_segment
from .decision import decide_global, decide_segment
from .dedup import dedup_highlights
from .scene_card import attach_scene_cards
from .narrative import compose as compose_narrative
from .schema import NarrativeScene, Segment, SegmentFeatures, SegmentScores, VideoFeatures
from .store import write_parquet


def run(src: str, storage_root: str = "storage", do_normalize: bool = True,
        # Tier 1
        enable_faces: bool = True, enable_objects: bool = True,
        enable_embeddings: bool = True, enable_asr: bool = False,
        enable_clip_zeroshot: bool = True, enable_camera_motion: bool = True,
        enable_ocr: bool = True, enable_quality: bool = True,
        enable_dedup: bool = True,
        # Tier 2
        enable_pose: bool = True, enable_saliency: bool = True,
        enable_depth: bool = True, enable_captions: bool = True,
        enable_action: bool = True, enable_tracking: bool = True,
        # Narrative
        enable_narrative: bool = True, narrative_polish: bool = False,
        write_store: bool = True) -> VideoFeatures:
    src = str(Path(src).resolve())
    vid = video_id_for(src)
    cdir = cache_dir(storage_root, vid)

    work_video = src
    if do_normalize:
        work_video = normalize(src, str(cdir / "normalized.mp4"))
    wav = extract_audio(work_video, str(cdir / "audio.wav"))

    duration, fps, width, height = probe(work_video)
    segs = segments_for(work_video, duration)

    if enable_dedup and not enable_embeddings:
        enable_embeddings = True
    if enable_tracking and not enable_objects:
        enable_objects = True

    # ---- Stage A — primary feature extraction
    with ThreadPoolExecutor(max_workers=8) as ex:
        f_visual = ex.submit(visual_per_segment, work_video, segs)
        f_audio = ex.submit(audio_per_segment, wav, segs)
        f_faces = ex.submit(faces_per_segment, work_video, segs) if enable_faces else None
        f_objects = ex.submit(objects_per_segment, work_video, segs, 1.0,
                              "yolov8n.pt", 0.35, enable_tracking) if enable_objects else None
        f_embed = ex.submit(embeddings_per_segment, work_video, segs) if enable_embeddings else None
        f_clipzs = ex.submit(clip_zeroshot_per_segment, work_video, segs) if enable_clip_zeroshot else None
        f_pose = ex.submit(pose_per_segment, work_video, segs) if enable_pose else None
        f_sal = ex.submit(saliency_per_segment, work_video, segs) if enable_saliency else None
        f_asr = ex.submit(transcribe, wav) if enable_asr else None

        visual = f_visual.result()
        audio = f_audio.result()
        faces = f_faces.result() if f_faces else [{"faces": 0, "face_size": 0.0}] * len(segs)
        objects = f_objects.result() if f_objects else [{"objects": [], "object_counts": {}, "detections": []}] * len(segs)
        embeddings = f_embed.result() if f_embed else [None] * len(segs)
        clip_zs = f_clipzs.result() if f_clipzs else [{"scene_category": None, "clip_tags": [], "clip_scores": {}}] * len(segs)
        pose = f_pose.result() if f_pose else [{"pose_present": False, "pose_action_hint": None, "keypoints_summary": {}}] * len(segs)
        sal = f_sal.result() if f_sal else [{"salient_center": None, "salient_bbox": None, "salient_area_ratio": 0.0}] * len(segs)
        words = f_asr.result() if f_asr else []

    # ---- Compute per-video adaptive stats from visual + audio
    stats = compute_stats(visual, audio)

    # ---- Stage B — adaptive classifiers + GPU-heavy semantic models (parallel)
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_depth = ex.submit(depth_per_segment, work_video, segs) if enable_depth else None
        f_cap = ex.submit(captions_per_segment, work_video, segs) if enable_captions else None
        f_action = ex.submit(action_per_segment, work_video, segs, clip_zs, pose) if enable_action else None
        # tracking depends on objects detections
        if enable_tracking:
            det_per_seg = [o.get("detections", []) for o in objects]
            f_track = ex.submit(tracking_per_segment, det_per_seg, segs, fps)
        else:
            f_track = None

        depth = f_depth.result() if f_depth else [{"depth_fg_ratio": 0.0, "depth_subject_distance": None}] * len(segs)
        captions = f_cap.result() if f_cap else [{"caption": ""}] * len(segs)
        action = f_action.result() if f_action else [{"action_top1": None, "action_top5": []}] * len(segs)
        track = f_track.result() if f_track else [{"track_ids": [], "dominant_track_id": None, "track_persistence": 0.0}] * len(segs)

    # adaptive serial classifiers (cheap)
    cam_motions = (camera_motion_per_segment(visual, stats) if enable_camera_motion
                   else [{"camera_motion": "unknown", "camera_motion_conf": 0.0}] * len(segs))
    shots = shot_type_per_segment(faces, visual)
    qual = (quality_per_segment(visual, stats) if enable_quality
            else [{"low_quality": False, "quality_tags": []}] * len(segs))

    # OCR uses adaptive gate
    if enable_ocr:
        ocr_data = ocr_per_segment(work_video, segs, visual, shots, stats,
                                   scene_cuts=[True] * len(segs))
    else:
        ocr_data = [{"ocr_text": "", "ocr_boxes": [], "has_text_overlay": False}] * len(segs)

    # ---- Build timeline
    timeline = []
    for i, ((t0, t1), v, a, fc, ob, em, cz, cm, st, q, oc, ps, sa, dp, cp, ac, tr) in enumerate(
            zip(segs, visual, audio, faces, objects, embeddings, clip_zs,
                cam_motions, shots, qual, ocr_data, pose, sal, depth, captions, action, track)):
        feats = SegmentFeatures(
            motion=v["motion"], stability=v["stability"],
            brightness=v["brightness"], contrast=v["contrast"],
            edge_density=v.get("edge_density", 0.0), blur_score=v.get("blur_score", 0.0),
            flow_fx_mean=v.get("flow_fx_mean", 0.0), flow_fy_mean=v.get("flow_fy_mean", 0.0),
            flow_divergence=v.get("flow_divergence", 0.0), flow_dir_var=v.get("flow_dir_var", 0.0),
            audio_energy=a["audio_energy"], onset_strength=a["onset_strength"],
            spectral_flux=a["spectral_flux"], speech=a["speech"],
            speech_ratio=a["speech_ratio"], music_prob=a["music_prob"],
            faces=fc["faces"], face_size=fc["face_size"],
            objects=ob["objects"], object_counts=ob["object_counts"],
            embedding=em, scene_cut=True,
            scene_category=cz["scene_category"], clip_tags=cz["clip_tags"],
            clip_scores=cz["clip_scores"],
            camera_motion=cm["camera_motion"], camera_motion_conf=cm["camera_motion_conf"],
            shot_type=st["shot_type"],
            ocr_text=oc["ocr_text"], has_text_overlay=oc["has_text_overlay"],
            low_quality=q["low_quality"],
            pose_present=ps["pose_present"], pose_action_hint=ps["pose_action_hint"],
            keypoints_summary=ps.get("keypoints_summary", {}),
            salient_center=sa["salient_center"], salient_bbox=sa["salient_bbox"],
            salient_area_ratio=sa["salient_area_ratio"],
            depth_fg_ratio=dp["depth_fg_ratio"], depth_subject_distance=dp["depth_subject_distance"],
            caption=cp["caption"],
            action_top1=ac["action_top1"], action_top5=ac["action_top5"],
            track_ids=tr["track_ids"], dominant_track_id=tr["dominant_track_id"],
            track_persistence=tr["track_persistence"],
        )
        feats.scene_category = fallback_scene_category(
            feats.scene_category, feats.objects, feats.object_counts,
            feats.brightness, feats.faces,
        )

        seg = Segment(t0=t0, t1=t1, features=feats, scores=SegmentScores())
        attach_fusion_tags(seg)
        score_segment(seg, stats, i)
        tag_segment(seg)
        decide_segment(seg)
        timeline.append(seg)

    if words:
        transcripts = assign_words_to_segments(words, timeline)
        for seg, txt in zip(timeline, transcripts):
            seg.transcript = txt

    highlights = (dedup_highlights(timeline) if enable_dedup
                  else select_highlights(timeline))

    vf = VideoFeatures(
        video_id=vid, source_path=src, duration=duration, fps=fps,
        width=width, height=height,
        timeline=timeline, highlights=highlights, words=words,
    )
    decide_global(vf)
    attach_scene_cards(vf)

    if enable_narrative:
        narr = compose_narrative(timeline, polish=narrative_polish)
        vf.narrative = narr["paragraph"]
        vf.narrative_summary = narr.get("summary")
        vf.narrative_bullets = narr.get("bullets", [])
        vf.narrative_scenes = [NarrativeScene(t0=s["t0"], t1=s["t1"], text=s["text"])
                               for s in narr.get("scenes", [])]

    if write_store:
        try:
            write_parquet(vf, str(cdir / "features.parquet"))
        except Exception as e:
            import sys
            print(f"[warn] parquet write failed: {e}", file=sys.stderr)

    return vf
