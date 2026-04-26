from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

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
from .scoring import score_segment, select_highlights, tag_segment
from .decision import decide_global, decide_segment
from .schema import Segment, SegmentFeatures, SegmentScores, VideoFeatures
from .store import write_parquet


def run(src: str, storage_root: str = "storage", do_normalize: bool = True,
        enable_faces: bool = True, enable_objects: bool = True,
        enable_embeddings: bool = False, enable_asr: bool = False,
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

    # Parallel feature extraction. CPU-bound items are fine in threads thanks to GIL release in cv2/librosa.
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_visual = ex.submit(visual_per_segment, work_video, segs)
        f_audio = ex.submit(audio_per_segment, wav, segs)
        f_faces = ex.submit(faces_per_segment, work_video, segs) if enable_faces else None
        f_objects = ex.submit(objects_per_segment, work_video, segs) if enable_objects else None
        f_embed = ex.submit(embeddings_per_segment, work_video, segs) if enable_embeddings else None
        f_asr = ex.submit(transcribe, wav) if enable_asr else None

        visual = f_visual.result()
        audio = f_audio.result()
        faces = f_faces.result() if f_faces else [{"faces": 0, "face_size": 0.0}] * len(segs)
        objects = f_objects.result() if f_objects else [{"objects": [], "object_counts": {}}] * len(segs)
        embeddings = f_embed.result() if f_embed else [None] * len(segs)
        words = f_asr.result() if f_asr else []

    timeline = []
    for (t0, t1), v, a, fc, ob, em in zip(segs, visual, audio, faces, objects, embeddings):
        feats = SegmentFeatures(
            motion=v["motion"],
            stability=v["stability"],
            brightness=v["brightness"],
            contrast=v["contrast"],
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
        )
        seg = Segment(t0=t0, t1=t1, features=feats, scores=SegmentScores())
        score_segment(seg)
        tag_segment(seg)
        decide_segment(seg)
        timeline.append(seg)

    if words:
        transcripts = assign_words_to_segments(words, timeline)
        for seg, txt in zip(timeline, transcripts):
            seg.transcript = txt

    highlights = select_highlights(timeline)

    vf = VideoFeatures(
        video_id=vid, source_path=src, duration=duration, fps=fps,
        width=width, height=height,
        timeline=timeline, highlights=highlights, words=words,
    )
    decide_global(vf)

    if write_store:
        try:
            write_parquet(vf, str(cdir / "features.parquet"))
        except Exception as e:
            import sys
            print(f"[warn] parquet write failed: {e}", file=sys.stderr)

    return vf
