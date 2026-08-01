# build_manifest.py
#
# Purpose:
#   Scan data/raw/ and data/distorted/ to build data/manifest.csv, mapping
#   each (clean, distorted) utterance pair to its metadata so downstream
#   code (src/training/dataset.py) has a single source of truth for what
#   data exists.
#
# Responsibilities:
#   - Walk data/raw/ for clean utterances and data/distorted/ for their
#     mild/moderate/severe counterparts (named {utterance_id}_{severity}.wav,
#     as produced by src/preprocessing/distortion.py)
#   - Derive speaker_id and transcript from each filename
#   - Measure each distorted clip's duration with soundfile
#   - Write one row per (utterance, severity) pair to data/manifest.csv,
#     sorted by utterance_id then severity

import glob
import os

import pandas as pd
import soundfile as sf

RAW_DIR = os.path.join("data", "raw")
DISTORTED_DIR = os.path.join("data", "distorted")
MANIFEST_PATH = os.path.join("data", "manifest.csv")

# Single-speaker dataset for now; revisit once more speakers are recorded.
SPEAKER_ID = "spk001"

SEVERITIES = ("mild", "moderate", "severe")

COLUMNS = [
    "utterance_id",
    "speaker_id",
    "clean_path",
    "distorted_path",
    "severity",
    "transcript",
    "duration_sec",
]


def extract_transcript(utterance_id):
    """Recover the Thai transcript from a filename like '001_ฉันชื่อ...'."""
    if "_" not in utterance_id:
        return ""
    return utterance_id.split("_", 1)[1]


def build_manifest_rows(raw_dir, distorted_dir):
    rows = []
    raw_paths = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))

    for raw_path in raw_paths:
        utterance_id = os.path.splitext(os.path.basename(raw_path))[0]
        transcript = extract_transcript(utterance_id)

        for severity in SEVERITIES:
            distorted_path = os.path.join(distorted_dir, f"{utterance_id}_{severity}.wav")
            if not os.path.exists(distorted_path):
                print(f"  warning: missing distorted file, skipping: {distorted_path}")
                continue

            rows.append({
                "utterance_id": utterance_id,
                "speaker_id": SPEAKER_ID,
                "clean_path": raw_path,
                "distorted_path": distorted_path,
                "severity": severity,
                "transcript": transcript,
                "duration_sec": round(sf.info(distorted_path).duration, 4),
            })

    return rows


def build_manifest_df(raw_dir, distorted_dir):
    rows = build_manifest_rows(raw_dir, distorted_dir)
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["utterance_id", "severity"]).reset_index(drop=True)


def print_summary(df):
    print(f"\nWrote {len(df)} row(s) covering {df['utterance_id'].nunique()} utterance(s) to {MANIFEST_PATH}")
    counts = df["severity"].value_counts().reindex(SEVERITIES, fill_value=0)
    for severity, count in counts.items():
        print(f"  {severity}: {count} rows")
    print(f"  total distorted audio duration: {df['duration_sec'].sum():.1f} sec")


def main():
    print(f"Scanning {RAW_DIR} and {DISTORTED_DIR} ...")
    df = build_manifest_df(RAW_DIR, DISTORTED_DIR)

    if df.empty:
        print("No rows generated; check that data/raw/ and data/distorted/ are populated.")
        return

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8")

    print_summary(df)


if __name__ == "__main__":
    main()
