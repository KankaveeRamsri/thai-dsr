#!/usr/bin/env python3
"""Compare completed W5 wav2vec2 layer-analysis runs."""

import json
from pathlib import Path


LAYERS = (6, 8, 9, 10, 12)
RESULT_TEMPLATE = "results/metrics_layer{layer}.json"


def load_result(layer):
    path = Path(RESULT_TEMPLATE.format(layer=layer))
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    reconstructed = {
        metric: payload["summary"][metric]["reconstructed"]["mean"]
        for metric in ("stoi", "pesq", "snr")
    }
    return {
        "layer": layer,
        "val_loss": float(payload["val_loss"]),
        **reconstructed,
    }


def main():
    missing = [
        RESULT_TEMPLATE.format(layer=layer)
        for layer in LAYERS
        if not Path(RESULT_TEMPLATE.format(layer=layer)).is_file()
    ]
    if missing:
        raise SystemExit("ผลวิเคราะห์ยังไม่ครบ:\n  " + "\n  ".join(missing))

    results = [load_result(layer) for layer in LAYERS]
    print(f"{'Layer':<8} {'Val Loss':>10} {'STOI':>10} {'PESQ':>10} {'SNR (dB)':>10}")
    print("-" * 52)
    for row in results:
        current = " ← current" if row["layer"] == 9 else ""
        print(
            f"{row['layer']:<8} {row['val_loss']:>10.4f} {row['stoi']:>10.4f} "
            f"{row['pesq']:>10.4f} {row['snr']:>10.2f}{current}"
        )

    best = min(results, key=lambda row: row["val_loss"])
    print(
        f"\nแนะนำ layer {best['layer']}: มี validation loss ต่ำที่สุด "
        f"({best['val_loss']:.4f}) บนชุด W5"
    )


if __name__ == "__main__":
    main()
