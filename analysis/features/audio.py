from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import webrtcvad

try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    _HAS_LIBROSA = False


def _frame_generator(pcm: bytes, sr: int, frame_ms: int = 30):
    n = int(sr * frame_ms / 1000) * 2
    for i in range(0, len(pcm) - n, n):
        yield pcm[i:i + n]


def audio_per_segment(wav_path: str, segments: List[Tuple[float, float]]) -> List[Dict]:
    audio, sr = sf.read(wav_path, dtype="int16", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype("int16")
    pcm = audio.tobytes()
    vad = webrtcvad.Vad(2)

    # 100Hz RMS envelope
    hop = max(1, sr // 100)
    rms = np.array([
        float(np.sqrt(np.mean(audio[i:i + hop].astype(np.float32) ** 2)))
        for i in range(0, len(audio), hop)
    ])
    if rms.size:
        rms = rms / (rms.max() + 1e-6)

    # librosa onset + spectral flux + music/speech proxy
    onset_env = None
    flux = None
    sc = None  # spectral centroid
    zcr = None
    if _HAS_LIBROSA:
        y = audio.astype(np.float32) / 32768.0
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        S = np.abs(librosa.stft(y, hop_length=hop))
        flux = np.sqrt(np.maximum(np.diff(S, axis=1), 0).sum(axis=0))
        if onset_env.size:
            onset_env = onset_env / (onset_env.max() + 1e-6)
        if flux.size:
            flux = flux / (flux.max() + 1e-6)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

    out = []
    for t0, t1 in segments:
        i0, i1 = int(t0 * 100), max(int(t0 * 100) + 1, int(t1 * 100))
        seg_rms = rms[i0:i1] if rms.size else np.array([0.0])
        energy = float(seg_rms.mean()) if seg_rms.size else 0.0

        onset = float(onset_env[i0:i1].mean()) if onset_env is not None and onset_env.size else 0.0
        sflux = float(flux[i0:i1].mean()) if flux is not None and flux.size else 0.0

        # speech VAD
        b0 = (int(t0 * sr) * 2) & ~1
        b1 = (int(t1 * sr) * 2) & ~1
        chunk = pcm[b0:b1]
        spk = tot = 0
        for f in _frame_generator(chunk, sr, 30):
            tot += 1
            try:
                if vad.is_speech(f, sr):
                    spk += 1
            except Exception:
                pass
        ratio = spk / tot if tot else 0.0

        # music vs speech heuristic: speech = high zcr variance + mid centroid
        # music = sustained centroid + low zcr variance. Output music_prob in [0,1].
        music_prob = 0.0
        if sc is not None and zcr is not None and sc.size and zcr.size:
            seg_sc = sc[i0:i1]
            seg_zcr = zcr[i0:i1]
            if seg_sc.size and seg_zcr.size:
                # low zcr std + high mean centroid → music-like
                zcr_std = float(seg_zcr.std())
                centroid_mean = float(seg_sc.mean()) / (sr / 2)
                music_prob = float(np.clip(centroid_mean - zcr_std * 2.0, 0.0, 1.0))
                if ratio > 0.4:
                    music_prob *= 0.3  # speech dominates

        out.append({
            "audio_energy": energy,
            "onset_strength": onset,
            "spectral_flux": sflux,
            "speech": ratio > 0.4,
            "speech_ratio": ratio,
            "music_prob": music_prob,
        })
    return out
