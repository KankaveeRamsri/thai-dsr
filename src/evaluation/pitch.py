"""Pitch / tone metrics against a clean reference on the same timeline (Praat via parselmouth).

    track = analyze(wav, sr)                    # F0 contour + mean HNR of one signal
    metrics = compare(track_ref, track_est)     # F0 correlation / RMSE / V-UV agreement

Both signals must already be on the same timeline and the same length (the
evaluation inverse-warps outputs onto the clean timeline first), so frame i of
one track is frame i of the other.

- F0: Praat autocorrelation pitch ("To Pitch (ac)"), 10 ms step, 75-500 Hz,
  otherwise Praat defaults. A frame is voiced when Praat returns F0 > 0.
- f0_corr: Pearson correlation of F0 in semitones on frames voiced in both.
- f0_rmse_st: RMS of 12 * log2(F0_est / F0_ref) on frames voiced in both.
- vuv_agree: share of all frames with the same voiced/unvoiced decision.
- hnr_db: mean harmonics-to-noise ratio, Praat "To Harmonicity (cc)" (10 ms,
  75 Hz, silence threshold 0.1, 1 period per window), averaged by Praat over
  frames it does not mark as silent. This is the same setting as
  scripts/measure_acoustics.py.
Clips with fewer than MIN_VOICED_BOTH frames voiced in both get NaN f0_corr and
f0_rmse_st.
"""

from dataclasses import dataclass

import numpy as np
import parselmouth
from parselmouth.praat import call

TIME_STEP = 0.01
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0
MIN_VOICED_BOTH = 10
PITCH_COLUMNS = ["f0_corr", "f0_rmse_st", "vuv_agree", "f0_n_voiced_both", "hnr_db", "hnr_clean_db"]


@dataclass
class PitchTrack:
    f0: np.ndarray  # Hz per 10 ms frame, 0 = unvoiced
    hnr_db: float


def _sound(wav, sr):
    return parselmouth.Sound(np.asarray(wav, dtype=np.float64), sampling_frequency=sr)


def analyze(wav, sr):
    sound = _sound(wav, sr)
    pitch = sound.to_pitch_ac(time_step=TIME_STEP, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ)
    f0 = np.nan_to_num(pitch.selected_array["frequency"], nan=0.0)
    harmonicity = call(sound, "To Harmonicity (cc)", TIME_STEP, PITCH_FLOOR_HZ, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    return PitchTrack(f0=f0, hnr_db=float(hnr) if np.isfinite(hnr) else float("nan"))


def compare(reference, estimate):
    """F0 / voicing agreement of estimate vs reference (same timeline)."""
    n = min(len(reference.f0), len(estimate.f0))
    if abs(len(reference.f0) - len(estimate.f0)) > 1:
        raise ValueError(f"pitch tracks differ by {abs(len(reference.f0) - len(estimate.f0))} frames: "
                         "signals are not on the same timeline")
    ref, est = reference.f0[:n], estimate.f0[:n]
    v_ref, v_est = ref > 0, est > 0
    both = v_ref & v_est
    out = {"vuv_agree": float(np.mean(v_ref == v_est)) if n else float("nan"),
           "f0_n_voiced_both": int(both.sum()), "f0_corr": float("nan"), "f0_rmse_st": float("nan"),
           "hnr_db": estimate.hnr_db, "hnr_clean_db": reference.hnr_db}
    if both.sum() >= MIN_VOICED_BOTH:
        st_ref = 12 * np.log2(ref[both])
        st_est = 12 * np.log2(est[both])
        out["f0_rmse_st"] = float(np.sqrt(np.mean((st_est - st_ref) ** 2)))
        if np.std(st_ref) > 0 and np.std(st_est) > 0:
            out["f0_corr"] = float(np.corrcoef(st_ref, st_est)[0, 1])
    return out
