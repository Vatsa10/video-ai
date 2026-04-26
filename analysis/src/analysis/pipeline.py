from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .preprocess import (
    cache_dir,
    extract_audio,
    normalize,
    probe,
    video_id_for,
)
from .segmentation import segments_for
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
from .scoring import attach_fusion_tags, score_segment, select_highlights, tag_segment
from .decision import decide_global, decide_segment
from .dedup import dedup_highlights
from .scene_card import attach_scene_cards
from .schema import Segment, SegmentFeatures, SegmentScores, VideoFeatures
from .store import write_parquet


def run(src: str, storage_root: str = "storage", do_normalize: bool = True,
        enable_faces: bool = True, enable_objects: bool = True,
        enable_embeddings: bool = True, enable_asr: bool = False,
        enable_clip_zeroshot: bool = True, enable_camera_motion: bool = True,
        enable_ocr: bool = True, enable_quality: bool = True,
        enable_dedup: bool = True,
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

    # Embeddings forced on if dedup enabled
    if enable_dedup and not enable_embeddings:
        enable_embeddings = True

    with ThreadPoolExecutor(max_workers=6) as ex:
        f_visual = ex.submit(visual_per_segment, work_video, segs)
        f_audio = ex.submit(audio_per_segment, wav, segs)
        f_faces = ex.submit(faces_per_segment, work_video, segs) if enable_faces else None
        f_objects = ex.submit(objects_per_segment, work_video, segs) if enable_objects else None
        f_embed = ex.submit(embeddings_per_segment, work_video, segs) if enable_embeddings else None
        f_clipzs = ex.submit(clip_zeroshot_per_segment, work_video, segs) if enable_clip_zeroshot else None
        f_asr = ex.submit(transcribe, wav) if enable_asr else None

        visual = f_visual.result()
        audio = f_audio.result()
        faces = f_faces.result() if f_faces else [{"faces": 0, "face_size": 0.0}] * len(segs)
        objects = f_objects.result() if f_objects else [{"objects": [], "object_counts": {}}] * len(segs)
        embeddings = f_embed.result() if f_embed else [None] * len(segs)
        clip_zs = f_clipzs.result() if f_clipzs else [{"scene_category": None, "clip_tags": [], "clip_scores": {}}] * len(segs)
        words = f_asr.result() if f_asr else []

    # Camera motion (CPU, consumes flow stats already in `visual`)
    cam_motions = camera_motion_per_segment(visual) if enable_camera_motion else \
        [{"camera_motion": "unknown", "camera_motion_conf": 0.0}] * len(segs)

    # Shot type (post-merge: needs faces + visual)
    shots = shot_type_per_segment(faces, visual)

    # Quality filter
    qual = quality_per_segment(visual) if enable_quality else \
        [{"low_quality": False, "quality_tags": []}] * len(segs)

    # OCR — gated, runs in main thread (fast post-process; reader load is heavy but only triggers if any segment passes gate)
    if enable_ocr:
        ocr = ocr_per_segment(work_video, segs, visual, shots,
                              scene_cuts=[True] * len(segs))
    else:
        ocr = [{"ocr_text": "", "ocr_boxes": [], "has_text_overlay": False}] * len(segs)

    timeline = []
    for i, ((t0, t1), v, a, fc, ob, em, cz, cm, st, q, oc) in enumerate(
            zip(segs, visual, audio, faces, objects, embeddings, clip_zs,
                cam_motions, shots, qual, ocr)):
        feats = SegmentFeatures(
            motion=v["motion"],
            stability=v["stability"],
            brightness=v["brightness"],
            contrast=v["contrast"],
            edge_density=v.get("edge_density", 0.0),
            blur_score=v.get("blur_score", 0.0),
            flow_fx_mean=v.get("flow_fx_mean", 0.0),
            flow_fy_mean=v.get("flow_fy_mean", 0.0),
            flow_divergence=v.get("flow_divergence", 0.0),
            flow_dir_var=v.get("flow_dir_var", 0.0),
            audio_energy=a["audio_energy"],
            onset_strength=a["onset_strength"],
            spectral_flux=a["spectral_flux"],
            speech=a["speech"],
            speech_ratio=a["speech_ratio"],
            music_prob=a["music_prob"],
            faces=fc["faces"],
            face_size=fc["face_size"],
            objects=ob["objects"],
            object_counts=ob["object_counts"],
            embedding=em,
            scene_cut=True,
            scene_category=cz["scene_category"],
            clip_tags=cz["clip_tags"],
            clip_scores=cz["clip_scores"],
            camera_motion=cm["camera_motion"],
            camera_motion_conf=cm["camera_motion_conf"],
            shot_type=st["shot_type"],
            ocr_text=oc["ocr_text"],
            has_text_overlay=oc["has_text_overlay"],
            low_quality=q["low_quality"],
        )
        # scene category fallback
        feats.scene_category = fallback_scene_category(
            feats.scene_category, feats.objects, feats.object_counts,
            feats.brightness, feats.faces,
        )

        seg = Segment(t0=t0, t1=t1, features=feats, scores=SegmentScores())
        attach_fusion_tags(seg)
        score_segment(seg)
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

    if write_store:
        try:
            write_parquet(vf, str(cdir / "features.parquet"))
        except Exception as e:
            import sys
            print(f"[warn] parquet write failed: {e}", file=sys.stderr)

    return vf
