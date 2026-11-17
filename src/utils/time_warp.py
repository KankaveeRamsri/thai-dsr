"""Exact clean <-> distorted time maps from logged tempo segments.

``segment_tempo_perturbation`` (src/preprocessing/distortion.py) cuts the
pitch-perturbed clip, which has the clean clip's length, into contiguous
segments and time-stretches each with ``librosa.effects.time_stretch``, whose
output length is ``int(round(length / rate))``. The noise/reverb stage keeps the
length. Given the logged ``[start, length, rate]`` list (samples at the clean
sample rate), the map between the two timelines is therefore piecewise linear
with one knot per segment boundary.

Mel frames follow ``src.utils.mel.compute_mel``: audio is resampled to 22.05
kHz, reflect-padded by (n_fft - hop) / 2 and framed without centring, so frame
``j`` is centred at ``(j + 0.5) * hop`` samples and a clip of ``n`` samples at
``sr`` has ``ceil(n * 22050 / sr) // hop`` frames.
"""

import json
import math

import librosa
import numpy as np

from src.utils.mel import HIFIGAN_SAMPLE_RATE

MEL_HOP = 256


def mel_frame_count(n_samples, sr, hop=MEL_HOP):
    """Frames ``compute_mel`` returns for ``n_samples`` of audio at ``sr``."""
    resampled = n_samples if sr == HIFIGAN_SAMPLE_RATE else math.ceil(n_samples * HIFIGAN_SAMPLE_RATE / sr)
    return resampled // hop


def stretched_length(length, rate):
    return length if rate is None else int(round(length / rate))


class TempoWarp:
    """Piecewise-linear map between the clean and distorted timelines."""

    def __init__(self, segments, sr):
        if isinstance(segments, str):
            segments = json.loads(segments)
        self.segments = [(int(s), int(n), None if r is None else float(r)) for s, n, r in segments]
        self.sr = sr
        clean, dist = [0.0], [0.0]
        position = 0
        for start, length, rate in self.segments:
            if start != position:
                raise ValueError(f"segments are not contiguous at sample {start} (expected {position})")
            position += length
            clean.append(float(position))
            dist.append(dist[-1] + stretched_length(length, rate))
        self.clean_knots = np.asarray(clean)  # samples at self.sr
        self.dist_knots = np.asarray(dist)

    @property
    def clean_samples(self):
        return int(self.clean_knots[-1])

    @property
    def distorted_samples(self):
        return int(self.dist_knots[-1])

    def clean_to_distorted(self, seconds):
        x = np.asarray(seconds, dtype=np.float64) * self.sr
        return np.interp(x, self.clean_knots, self.dist_knots) / self.sr

    def distorted_to_clean(self, seconds):
        x = np.asarray(seconds, dtype=np.float64) * self.sr
        return np.interp(x, self.dist_knots, self.clean_knots) / self.sr

    def warp_frames(self, clean_frames, n_target=None, hop=MEL_HOP, frame_sr=HIFIGAN_SAMPLE_RATE):
        """Resample ``(T_clean, D)`` frames onto the distorted frame grid.

        Each distorted frame centre is mapped back to clean time and the clean
        frames are linearly interpolated there, segment by segment.
        """
        clean_frames = np.asarray(clean_frames)
        if n_target is None:
            n_target = mel_frame_count(self.distorted_samples, self.sr, hop)
        centres = (np.arange(n_target) + 0.5) * hop / frame_sr
        position = self.distorted_to_clean(centres) * frame_sr / hop - 0.5
        position = np.clip(position, 0, len(clean_frames) - 1)
        low = np.floor(position).astype(int)
        high = np.minimum(low + 1, len(clean_frames) - 1)
        weight = (position - low)[:, None]
        return ((1 - weight) * clean_frames[low] + weight * clean_frames[high]).astype(clean_frames.dtype)

    def stretch_waveform(self, clean):
        """Apply the logged per-segment time-stretch to a clean waveform.

        Uses the same phase-vocoder call and rates as the distortion's tempo
        stage, so the result is time-aligned with the distorted audio.
        """
        clean = np.asarray(clean, dtype=np.float32)
        if len(clean) != self.clean_samples:
            raise ValueError(f"waveform has {len(clean)} samples, segments cover {self.clean_samples}")
        pieces = []
        for start, length, rate in self.segments:
            piece = clean[start:start + length]
            if rate is not None:
                piece = librosa.effects.time_stretch(piece, rate=rate)
            pieces.append(piece)
        return np.concatenate(pieces).astype(np.float32)
