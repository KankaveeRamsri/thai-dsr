# metrics_en.py
#
# Purpose:
#   Same STOI/PESQ/SNR evaluation as metrics.py, but for ad-hoc folders that
#   have no data/manifest.csv entry (e.g. the English comparison set).
#   Matches files by stem instead of reading a manifest: for a clean file
#   `data/raw_en/X.wav`, expects `data/distorted_en/X_<severity>.wav` and
#   `results/audio_samples/en_v2/X_<severity>_reconstructed.wav`.

import argparse
import glob
import json
import os

import pandas as pd

from src.evaluation.metrics import (
    CONDITIONS,
    METRIC_LABELS,
    compute_pesq,
    compute_snr,
    compute_stoi,
    load_audio_16k,
    print_summary_table,
    summarize,
)


def evaluate(clean_dir, distorted_dir, reconstructed_dir, severity):
    rows = []
    for clean_path in sorted(glob.glob(os.path.join(clean_dir, "*.wav"))):
        utterance_id = os.path.splitext(os.path.basename(clean_path))[0]
        distorted_path = os.path.join(distorted_dir, f"{utterance_id}_{severity}.wav")
        recon_path = os.path.join(reconstructed_dir, f"{utterance_id}_{severity}_reconstructed.wav")

        if not os.path.exists(distorted_path):
            print(f"WARNING: missing distorted file, skipping {utterance_id}: {distorted_path}")
            continue
        if not os.path.exists(recon_path):
            print(f"WARNING: missing reconstructed file, skipping {utterance_id}: {recon_path}")
            continue

        clean = load_audio_16k(clean_path)
        distorted = load_audio_16k(distorted_path)
        reconstructed = load_audio_16k(recon_path)

        rows.append({
            "utterance_id": utterance_id,
            "severity": severity,
            "stoi_distorted": compute_stoi(clean, distorted),
            "stoi_reconstructed": compute_stoi(clean, reconstructed),
            "pesq_distorted": compute_pesq(clean, distorted),
            "pesq_reconstructed": compute_pesq(clean, reconstructed),
            "snr_distorted": compute_snr(clean, distorted),
            "snr_reconstructed": compute_snr(clean, reconstructed),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute STOI/PESQ/SNR for a clean/distorted/reconstructed folder set (no manifest needed)."
    )
    parser.add_argument("--clean_dir", type=str, default=os.path.join("data", "raw_en"))
    parser.add_argument("--distorted_dir", type=str, default=os.path.join("data", "distorted_en"))
    parser.add_argument("--reconstructed_dir", type=str, default=os.path.join("results", "audio_samples", "en_v2"))
    parser.add_argument("--severity", type=str, default="severe")
    parser.add_argument("--output_json", type=str, default=os.path.join("results", "metrics_en.json"))
    args = parser.parse_args()

    df = evaluate(args.clean_dir, args.distorted_dir, args.reconstructed_dir, args.severity)
    if df.empty:
        print("No matched clean/distorted/reconstructed triples found.")
        return

    summary = summarize(df)
    print_summary_table(summary, len(df))

    output = {
        "severity": args.severity,
        "num_utterances": len(df),
        "per_utterance": df.to_dict(orient="records"),
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
