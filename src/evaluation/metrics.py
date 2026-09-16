# metrics.py
#
# Purpose:
#   Compute objective speech quality/intelligibility metrics comparing
#   reconstructed audio against ground-truth clean audio.
#
# Expected responsibilities:
#   - Compute STOI via pystoi
#   - Compute PESQ via pesq
#   - Aggregate metrics across a results/audio_samples/ folder or manifest
#     and report summary statistics

import argparse
import json
import os

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq
from pystoi import stoi

MANIFEST_PATH = os.path.join("data", "manifest.csv")
RECONSTRUCTED_DIR = os.path.join("results", "audio_samples", "v2")
OUTPUT_JSON = os.path.join("results", "metrics.json")

SEVERITY = "severe"
TARGET_SAMPLE_RATE = 16000  # required by both pystoi and PESQ wideband mode
PESQ_MODE = "wb"

CONDITIONS = ("distorted", "reconstructed")
METRIC_LABELS = {"stoi": "STOI", "pesq": "PESQ", "snr": "SNR (dB)"}


def load_audio_16k(wav_path):
    """Load a .wav file, downmix to mono, and resample to TARGET_SAMPLE_RATE."""
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return audio


def align_length(reference, estimate):
    """Truncate both signals to their shared length.

    STOI/PESQ compare signals frame-for-frame and require equal length;
    distortion/reconstruction can shift duration slightly, so a simple
    truncation to the common length is used rather than time-alignment.
    """
    n = min(len(reference), len(estimate))
    return reference[:n], estimate[:n]


def compute_stoi(reference, estimate):
    reference, estimate = align_length(reference, estimate)
    return stoi(reference, estimate, TARGET_SAMPLE_RATE, extended=False)


def compute_pesq(reference, estimate):
    reference, estimate = align_length(reference, estimate)
    return pesq(TARGET_SAMPLE_RATE, reference, estimate, PESQ_MODE)


def compute_snr(reference, estimate):
    """SNR in dB, treating (estimate - reference) as the noise term."""
    reference, estimate = align_length(reference, estimate)
    noise = estimate - reference
    signal_power = np.sum(reference ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def reconstructed_path(utterance_id, severity, reconstructed_dir=RECONSTRUCTED_DIR):
    return os.path.join(reconstructed_dir, f"{utterance_id}_{severity}_reconstructed.wav")


def evaluate(
    manifest_path=MANIFEST_PATH,
    reconstructed_dir=RECONSTRUCTED_DIR,
    severity=SEVERITY,
):
    """Compute STOI/PESQ/SNR for distorted-vs-clean and reconstructed-vs-clean, per utterance."""
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["severity"] == severity].reset_index(drop=True)

    rows = []
    for _, row in manifest.iterrows():
        utterance_id = row["utterance_id"]
        recon_path = reconstructed_path(
            utterance_id, row["severity"], reconstructed_dir=reconstructed_dir
        )
        if not os.path.exists(recon_path):
            print(f"WARNING: missing reconstructed file, skipping {utterance_id}: {recon_path}")
            continue

        clean = load_audio_16k(row["clean_path"])
        distorted = load_audio_16k(row["distorted_path"])
        reconstructed = load_audio_16k(recon_path)

        rows.append({
            "utterance_id": utterance_id,
            "severity": row["severity"],
            "stoi_distorted": compute_stoi(clean, distorted),
            "stoi_reconstructed": compute_stoi(clean, reconstructed),
            "pesq_distorted": compute_pesq(clean, distorted),
            "pesq_reconstructed": compute_pesq(clean, reconstructed),
            "snr_distorted": compute_snr(clean, distorted),
            "snr_reconstructed": compute_snr(clean, reconstructed),
        })

    return pd.DataFrame(rows)


def summarize(df):
    summary = {}
    for metric in METRIC_LABELS:
        summary[metric] = {}
        for condition in CONDITIONS:
            values = df[f"{metric}_{condition}"]
            summary[metric][condition] = {"mean": float(values.mean()), "std": float(values.std())}
    return summary


def print_summary_table(summary, n, severity=SEVERITY):
    print(f"\nEvaluation summary (n={n} utterances, severity='{severity}')")
    header = f"{'Metric':<10} {'Distorted (mean ± std)':<26} {'Reconstructed (mean ± std)':<26}"
    print(header)
    print("-" * len(header))
    for metric, label in METRIC_LABELS.items():
        d = summary[metric]["distorted"]
        r = summary[metric]["reconstructed"]
        d_str = f"{d['mean']:.4f} ± {d['std']:.4f}"
        r_str = f"{r['mean']:.4f} ± {r['std']:.4f}"
        print(f"{label:<10} {d_str:<26} {r_str:<26}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstructed Thai speech.")
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--reconstructed_dir", default=RECONSTRUCTED_DIR)
    parser.add_argument("--severity", default=SEVERITY)
    parser.add_argument("--output_json", default=OUTPUT_JSON)
    args = parser.parse_args()

    df = evaluate(
        manifest_path=args.manifest,
        reconstructed_dir=args.reconstructed_dir,
        severity=args.severity,
    )
    if df.empty:
        raise ValueError(
            f"No reconstructed files evaluated from: {args.reconstructed_dir}"
        )
    summary = summarize(df)
    print_summary_table(summary, len(df), severity=args.severity)

    output = {
        "severity": args.severity,
        "num_utterances": len(df),
        "per_utterance": df.to_dict(orient="records"),
        "summary": summary,
    }

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
