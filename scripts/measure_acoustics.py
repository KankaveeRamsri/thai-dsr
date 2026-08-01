#!/usr/bin/env python3
"""Measure jitter/shimmer/HNR/F0/speaking-rate for data/raw/ vs data/distorted/.

For each clean utterance in data/raw/ this pulls its mild/moderate/severe
counterparts from data/distorted/ (named {stem}_{severity}.wav, as produced
by src/preprocessing/distortion.py) and prints a comparison table so we can
check whether the synthetic distortion pipeline actually lands in the
acoustic range reported for real dysarthric speech.
"""
import argparse
import glob
import math
import os
import re

import pandas as pd
import parselmouth
from parselmouth.praat import call

RAW_DIR = os.path.join("data", "raw")
DISTORTED_DIR = os.path.join("data", "distorted")
OUTPUT_CSV = os.path.join("results", "acoustic_measurements.csv")

PITCH_FLOOR_HZ = 75.0
PITCH_CEILING_HZ = 500.0

CONDITIONS = ("raw", "mild", "moderate", "severe")
SEVERITY_SUFFIX_RE = re.compile(r"^(.*)_(mild|moderate|severe)$")

FEATURES = ("jitter_local_pct", "shimmer_local_pct", "hnr_db", "mean_f0_hz", "voiced_rate_per_sec")

FEATURE_LABELS = {
    "jitter_local_pct": "Jitter (local) [%]",
    "shimmer_local_pct": "Shimmer (local) [%]",
    "hnr_db": "HNR [dB]",
    "mean_f0_hz": "Mean F0 [Hz]",
    "voiced_rate_per_sec": "Speaking rate [voiced frames/s]",
}

# Reference ranges reported for real dysarthric speech.
DYSARTHRIC_TARGET_RANGE = {
    "jitter_local_pct": "> 3 %",
    "shimmer_local_pct": "> 6 %",
    "hnr_db": "< 15 dB",
    "mean_f0_hz": "-",
    "voiced_rate_per_sec": "-",
}


def measure_acoustics(wav_path, pitch_floor=PITCH_FLOOR_HZ, pitch_ceiling=PITCH_CEILING_HZ):
    """Return jitter/shimmer/HNR/F0/speaking-rate for one .wav file.

    Jitter, shimmer, and HNR follow the standard Praat "voice report" call
    sequence (PointProcess for jitter/shimmer, Harmonicity (cc) for HNR).
    Speaking rate is approximated as voiced Pitch frames per second.
    """
    try:
        sound = parselmouth.Sound(wav_path)

        pitch = call(sound, "To Pitch", 0.0, pitch_floor, pitch_ceiling)
        mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")

        point_process = call(sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling)
        jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer_local = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )

        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, pitch_floor, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        n_frames = int(call(pitch, "Get number of frames"))
        voiced_frames = sum(
            1 for i in range(1, n_frames + 1)
            if not math.isnan(call(pitch, "Get value in frame", i, "Hertz"))
        )
        voiced_rate = voiced_frames / sound.duration if sound.duration > 0 else float("nan")

        return {
            "jitter_local_pct": jitter_local * 100 if not math.isnan(jitter_local) else float("nan"),
            "shimmer_local_pct": shimmer_local * 100 if not math.isnan(shimmer_local) else float("nan"),
            "hnr_db": hnr,
            "mean_f0_hz": mean_f0,
            "voiced_rate_per_sec": voiced_rate,
        }
    except Exception as exc:
        print(f"  warning: measurement failed for {wav_path} ({exc})")
        return {feature: float("nan") for feature in FEATURES}


def collect_file_groups(raw_dir, distorted_dir):
    """Map each utterance stem to its raw/mild/moderate/severe .wav paths."""
    groups = {}

    for raw_path in sorted(glob.glob(os.path.join(raw_dir, "*.wav"))):
        stem = os.path.splitext(os.path.basename(raw_path))[0]
        groups.setdefault(stem, {c: None for c in CONDITIONS})
        groups[stem]["raw"] = raw_path

    for distorted_path in sorted(glob.glob(os.path.join(distorted_dir, "*.wav"))):
        stem_with_severity = os.path.splitext(os.path.basename(distorted_path))[0]
        match = SEVERITY_SUFFIX_RE.match(stem_with_severity)
        if not match:
            continue
        base_stem, severity = match.groups()
        groups.setdefault(base_stem, {c: None for c in CONDITIONS})
        groups[base_stem][severity] = distorted_path

    return groups


def measure_all(groups, pitch_floor, pitch_ceiling):
    rows = []
    for stem in sorted(groups):
        for condition in CONDITIONS:
            path = groups[stem][condition]
            if path is None:
                continue
            print(f"  measuring [{condition}] {os.path.basename(path)}")
            features = measure_acoustics(path, pitch_floor, pitch_ceiling)
            rows.append({"file": stem, "condition": condition, **features})
    return pd.DataFrame(rows)


def print_detailed_table(df):
    print("\n=== Per-file measurements ===")
    ordered = df.copy()
    ordered["condition"] = pd.Categorical(ordered["condition"], categories=CONDITIONS, ordered=True)
    ordered = ordered.sort_values(["file", "condition"])
    ordered = ordered.round({f: 2 for f in FEATURES})
    print(ordered.to_string(index=False))


def print_comparison_table(df):
    print("\n=== Comparison: raw vs mild/moderate/severe vs dysarthric target ===")
    header = ["Feature"] + [c.capitalize() for c in CONDITIONS] + ["Dysarthric target"]
    table_rows = [header]

    for feature in FEATURES:
        row = [FEATURE_LABELS[feature]]
        for condition in CONDITIONS:
            values = df.loc[df["condition"] == condition, feature].dropna()
            row.append(f"{values.mean():.2f} +/- {values.std():.2f}" if len(values) else "-")
        row.append(DYSARTHRIC_TARGET_RANGE[feature])
        table_rows.append(row)

    widths = [max(len(str(row[i])) for row in table_rows) for i in range(len(header))]
    for i, row in enumerate(table_rows):
        print(" | ".join(str(cell).ljust(w) for cell, w in zip(row, widths)))
        if i == 0:
            print("-+-".join("-" * w for w in widths))


def print_target_check(df):
    print("\n=== Does 'severe' reach the reference dysarthric range? ===")
    severe = df.loc[df["condition"] == "severe"]
    if severe.empty:
        print("  (no severe-condition files found)")
        return

    checks = [
        ("Jitter (local) > 3%", severe["jitter_local_pct"].mean() > 3.0),
        ("Shimmer (local) > 6%", severe["shimmer_local_pct"].mean() > 6.0),
        ("HNR < 15 dB", severe["hnr_db"].mean() < 15.0),
    ]
    for label, passed in checks:
        mark = "PASS" if passed else "below target"
        print(f"  [{mark}] {label}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure jitter/shimmer/HNR/F0/speaking-rate for raw vs distorted speech."
    )
    parser.add_argument("--raw_dir", type=str, default=RAW_DIR)
    parser.add_argument("--distorted_dir", type=str, default=DISTORTED_DIR)
    parser.add_argument("--output_csv", type=str, default=OUTPUT_CSV)
    parser.add_argument("--pitch_floor", type=float, default=PITCH_FLOOR_HZ)
    parser.add_argument("--pitch_ceiling", type=float, default=PITCH_CEILING_HZ)
    args = parser.parse_args()

    groups = collect_file_groups(args.raw_dir, args.distorted_dir)
    if not groups:
        print(f"No .wav files found in {args.raw_dir} or {args.distorted_dir}")
        return

    print(f"Found {len(groups)} utterance(s); measuring acoustic features...")
    df = measure_all(groups, args.pitch_floor, args.pitch_ceiling)

    print_detailed_table(df)
    print_comparison_table(df)
    print_target_check(df)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved raw measurements to: {args.output_csv}")


if __name__ == "__main__":
    main()
