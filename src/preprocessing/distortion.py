# distortion.py
#
# Purpose:
#   Generate synthetic "dysarthric-like" distortions from clean Thai speech
#   audio, so the model has (distorted, clean) pairs to learn reconstruction
#   from when real dysarthric recordings are scarce.
#
# Responsibilities:
#   - Load clean audio from data/raw/
#   - Apply three distortion transforms, tunable by severity (mild/moderate/
#     severe):
#       1. Formant/pitch perturbation (parselmouth/Praat) - shifts F0,
#          adds per-frame F0 jitter, and shifts formants to simulate
#          imprecise articulation.
#       2. Segment-level tempo perturbation (librosa) - splits audio into
#          0.3-0.5s chunks and time-stretches each chunk independently to
#          simulate irregular speaking rate.
#       3. Additive noise + light reverberation (numpy) - adds Gaussian
#          noise at a target SNR plus a short synthetic reverb tail.
#   - Save distorted audio to data/distorted/ as
#     {original_name}_{severity}.wav
#   - Expose main() to batch-process every file in data/raw/

import argparse
import glob
import os
import random

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

# Directories (relative to project root).
DEFAULT_INPUT_DIR = os.path.join("data", "raw")
DEFAULT_OUTPUT_DIR = os.path.join("data", "distorted")

# Typical human pitch range used to guide Praat's pitch analysis.
PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0

SEVERITY_LEVELS = ("mild", "moderate", "severe")

# Per-severity distortion magnitudes. Ranges grow wider/harsher from mild
# to severe; "severe" reaches the outer bounds called out in the spec
# (F0 shift up to +-20%, tempo rate 0.6x-1.4x, SNR down to 15 dB).
SEVERITY_PARAMS = {
    "mild": {
        "f0_shift_range": (0.03, 0.08),
        "jitter_std": 0.005,
        "formant_shift_range": (0.02, 0.05),
        "segment_duration_range": (0.3, 0.5),
        "tempo_rate_range": (0.85, 1.15),
        "snr_db_range": (20.0, 25.0),
        "reverb_wet_mix": 0.05,
        "reverb_decay": 0.05,
    },
    "moderate": {
        "f0_shift_range": (0.08, 0.15),
        "jitter_std": 0.012,
        "formant_shift_range": (0.05, 0.09),
        "segment_duration_range": (0.3, 0.5),
        "tempo_rate_range": (0.7, 1.3),
        "snr_db_range": (18.0, 22.0),
        "reverb_wet_mix": 0.15,
        "reverb_decay": 0.15,
    },
    "severe": {
        "f0_shift_range": (0.15, 0.20),
        "jitter_std": 0.07,
        "formant_shift_range": (0.09, 0.14),
        "segment_duration_range": (0.3, 0.5),
        "tempo_rate_range": (0.6, 1.4),
        "snr_db_range": (15.0, 18.0),
        "reverb_wet_mix": 0.30,
        "reverb_decay": 0.30,
    },
}


# Random draws. With ``rng=None`` these make exactly the same calls on the
# global ``random`` / ``np.random`` state as before, so seeded runs of the
# original pipeline are unchanged. With an ``np.random.Generator`` every draw
# comes from that generator, so a per-file seed makes the parameters
# independent of processing order.
def _sign(rng):
    return random.choice([-1, 1]) if rng is None else int(rng.choice([-1, 1]))


def _uniform(rng, low, high):
    return random.uniform(low, high) if rng is None else float(rng.uniform(low, high))


def _normal(rng, scale, size=None):
    if rng is None:
        if size is None:
            return np.random.normal(0, scale)
        return np.random.normal(0, scale, size=size)
    return rng.normal(0, scale, size=size)


def _randn(rng, n):
    return np.random.randn(n) if rng is None else rng.standard_normal(n)


def perturb_pitch_and_formants(y, sr, f0_shift_range, jitter_std, formant_shift_range,
                               rng=None, record=None):
    """Shift F0, add per-frame F0 jitter, and shift formants via Praat.

    Formants are shifted first with Praat's "Change gender" (formant ratio
    only, pitch left untouched), then F0 is shifted and jittered by editing
    the pitch tier of a Manipulation object and resynthesizing.

    If ``record`` is a dict, the drawn parameters are stored in it, including
    ``pitch_status``: "ok", "no_pitch_points" (formant shift only, no F0
    change) or "praat_error" (original audio returned unchanged).
    """
    record = {} if record is None else record
    record.update(pitch_status="praat_error", formant_shift_ratio=None,
                  f0_shift_factor=None, jitter_std=jitter_std, pitch_points=0,
                  jitter_mean=None, jitter_sd=None, pitch_error="")
    try:
        sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)

        formant_shift_ratio = 1.0 + _sign(rng) * _uniform(rng, *formant_shift_range)
        record["formant_shift_ratio"] = formant_shift_ratio
        formant_shifted = call(
            sound, "Change gender", PITCH_FLOOR_HZ, PITCH_CEILING_HZ,
            formant_shift_ratio, 0, 1.0, 1.0,
        )

        manipulation = call(formant_shifted, "To Manipulation", 0.01, PITCH_FLOOR_HZ, PITCH_CEILING_HZ)
        pitch_tier = call(manipulation, "Extract pitch tier")
        n_points = call(pitch_tier, "Get number of points")
        record["pitch_points"] = int(n_points)

        if n_points == 0:
            # No voiced frames detected (e.g. silence/noise); nothing to jitter.
            record["pitch_status"] = "no_pitch_points"
            return formant_shifted.values[0].astype(np.float32)

        f0_shift_factor = 1.0 + _sign(rng) * _uniform(rng, *f0_shift_range)
        record["f0_shift_factor"] = f0_shift_factor

        new_tier = call("Create PitchTier", "jittered", sound.xmin, sound.xmax)
        jitters = []
        for i in range(1, n_points + 1):
            t = call(pitch_tier, "Get time from index", i)
            f = call(pitch_tier, "Get value at index", i)
            shifted = f * f0_shift_factor
            jitter = _normal(rng, jitter_std)
            jitters.append(jitter)
            jittered = shifted * (1.0 + jitter)
            jittered = max(jittered, PITCH_FLOOR_HZ * 0.5)
            call(new_tier, "Add point", t, jittered)
        record["jitter_mean"] = float(np.mean(jitters))
        record["jitter_sd"] = float(np.std(jitters))

        call([manipulation, new_tier], "Replace pitch tier")
        resynthesized = call(manipulation, "Get resynthesis (overlap-add)")
        record["pitch_status"] = "ok"
        return resynthesized.values[0].astype(np.float32)
    except Exception as exc:
        print(f"  warning: pitch/formant perturbation failed ({exc}); using original audio")
        record["pitch_status"] = "praat_error"
        record["pitch_error"] = str(exc)
        return y.astype(np.float32)


def segment_tempo_perturbation(y, sr, segment_duration_range, tempo_rate_range,
                               rng=None, record=None):
    """Time-stretch short, randomly-sized segments at independent rates.

    Unlike a single uniform time-stretch, each 0.3-0.5s chunk gets its own
    random rate, producing the irregular, non-uniform speaking rate typical
    of dysarthric speech.

    If ``record`` is a dict, ``tempo_segments`` is set to a list of
    ``[start_sample, length, rate]`` (rate is None for unstretched segments).
    """
    min_dur, max_dur = segment_duration_range
    min_stretchable_samples = 2048  # librosa's default STFT window; shorter segments warn

    segments = []
    log = []
    start = 0
    n_samples = len(y)
    while start < n_samples:
        seg_len = max(int(_uniform(rng, min_dur, max_dur) * sr), 1)
        end = min(start + seg_len, n_samples)
        segment = y[start:end]

        rate = None
        if len(segment) >= min_stretchable_samples:
            rate = _uniform(rng, *tempo_rate_range)
            try:
                segment = librosa.effects.time_stretch(segment.astype(np.float32), rate=rate)
            except Exception as exc:
                print(f"  warning: time-stretch failed on segment ({exc}); keeping it unstretched")
                rate = None

        log.append([start, end - start, rate])
        segments.append(segment)
        start = end

    if record is not None:
        record["tempo_segments"] = log
    if not segments:
        return y
    return np.concatenate(segments).astype(np.float32)


def add_noise_and_reverb(y, sr, snr_db_range, reverb_wet_mix, reverb_decay,
                         rng=None, record=None):
    """Add Gaussian noise at a random target SNR, then a short reverb tail."""
    signal_power = np.mean(y ** 2)
    if signal_power <= 0:
        signal_power = 1e-10

    snr_db = _uniform(rng, *snr_db_range)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = _normal(rng, np.sqrt(noise_power), size=y.shape).astype(np.float32)
    y_noisy = y + noise

    ir_len = max(int(sr * 0.05), 1)  # 50ms synthetic impulse response
    t = np.arange(ir_len)
    impulse = _randn(rng, ir_len) * np.exp(-t / (sr * reverb_decay + 1e-6))
    impulse /= np.max(np.abs(impulse)) + 1e-8

    wet = np.convolve(y_noisy, impulse, mode="full")[: len(y_noisy)]
    y_out = (1 - reverb_wet_mix) * y_noisy + reverb_wet_mix * wet

    peak = np.max(np.abs(y_out))
    if peak > 0.99:
        y_out = y_out / peak * 0.99
    if record is not None:
        record.update(snr_db=snr_db, reverb_wet_mix=reverb_wet_mix,
                      reverb_decay=reverb_decay, peak_before_norm=float(peak),
                      peak_normalized=bool(peak > 0.99))
    return y_out.astype(np.float32)


def apply_distortion(y, sr, severity, rng=None, record=None):
    """Run the full mild/moderate/severe distortion pipeline on one clip.

    ``rng``: optional ``np.random.Generator``. When None (default) the global
    ``random`` / ``np.random`` state is used exactly as before. When given,
    it is split into one child generator per stage, so the tempo and noise
    draws do not depend on how many pitch points Praat finds.
    ``record``: optional dict that receives every drawn parameter and
    ``pitch_status`` (see ``perturb_pitch_and_formants``).
    """
    if severity not in SEVERITY_PARAMS:
        raise ValueError(f"Unknown severity '{severity}', expected one of {SEVERITY_LEVELS}")
    params = SEVERITY_PARAMS[severity]
    if record is not None:
        record.update(severity=severity, input_samples=len(y))
    pitch_rng, tempo_rng, noise_rng = (None, None, None) if rng is None else rng.spawn(3)

    y = perturb_pitch_and_formants(
        y, sr, params["f0_shift_range"], params["jitter_std"], params["formant_shift_range"],
        rng=pitch_rng, record=record,
    )
    y = segment_tempo_perturbation(
        y, sr, params["segment_duration_range"], params["tempo_rate_range"],
        rng=tempo_rng, record=record,
    )
    y = add_noise_and_reverb(
        y, sr, params["snr_db_range"], params["reverb_wet_mix"], params["reverb_decay"],
        rng=noise_rng, record=record,
    )
    if record is not None:
        record["output_samples"] = len(y)
    return y


def distort_file(input_path, output_dir, severities=SEVERITY_LEVELS):
    """Load one clean .wav file and write one distorted file per severity."""
    y, sr = librosa.load(input_path, sr=None, mono=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    for severity in severities:
        distorted = apply_distortion(y, sr, severity)
        out_path = os.path.join(output_dir, f"{stem}_{severity}.wav")
        sf.write(out_path, distorted, sr)
        print(f"  [{severity}] -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply mild/moderate/severe dysarthric-like distortions to clean speech."
    )
    parser.add_argument(
        "--input_dir", type=str, default=DEFAULT_INPUT_DIR,
        help=f"Directory of clean .wav files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write distorted .wav files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--severities", type=str, nargs="+", default=list(SEVERITY_LEVELS), choices=SEVERITY_LEVELS,
        help="Which severity levels to generate (default: all three)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    wav_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    if not wav_paths:
        print(f"No .wav files found in {args.input_dir}")
        return

    print(f"Found {len(wav_paths)} file(s) in {args.input_dir}")
    for wav_path in wav_paths:
        print(f"Processing: {wav_path}")
        distort_file(wav_path, args.output_dir, args.severities)

    print(f"Done. Distorted files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
