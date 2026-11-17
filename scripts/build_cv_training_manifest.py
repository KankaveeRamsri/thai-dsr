#!/usr/bin/env python3
"""Turn the frozen CV 27.0 derived set into a training manifest and split file.

Inputs (all tracked in data/manifests/):
  cv27_th_speaker_split.csv    clip -> client_id, split, sentence
  cv27_th_distortion_params.csv one row per distorted file (kept/excluded, path)
  cv27_th_clean_16k_index.csv   clean 16 kHz wav per clip

Outputs:
  --manifest  build_manifest-style CSV (utterance_id, speaker_id, clean_path,
              distorted_path, severity, transcript, duration_sec), one row per
              kept distorted file. The tracked CSVs store paths relative to the
              data root ($DATA_ROOT or configs/data.yaml cv27_data_root); this
              per-machine manifest gets them resolved to absolute paths, so it
              is written inside the data root by default, outside the repo.
  --splits    {"splits": {train, val, test}} of utterance ids, copied from the
              speaker-level split (use with train.py --fixed-splits).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import cv_data_root, resolve_data_path  # noqa: E402

MANIFESTS = Path("data/manifests")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <data root>/training_manifest.csv")
    parser.add_argument("--splits", type=Path, default=MANIFESTS / "cv27_th_splits.json")
    args = parser.parse_args()
    root = cv_data_root()
    if args.manifest is None:
        args.manifest = root / "training_manifest.csv"

    split = pd.read_csv(MANIFESTS / "cv27_th_speaker_split.csv", dtype=str)
    params = pd.read_csv(MANIFESTS / "cv27_th_distortion_params.csv")
    clean = pd.read_csv(MANIFESTS / "cv27_th_clean_16k_index.csv")

    kept = params[params["kept"]]
    rows = kept.merge(clean[["clip_id", "clean_path"]], on="clip_id", validate="many_to_one")
    rows = rows.merge(split[["clip_id", "sentence"]], on="clip_id", validate="many_to_one")
    manifest = pd.DataFrame({
        "utterance_id": rows["clip_id"],
        "speaker_id": rows["client_id"],
        "clean_path": [str(resolve_data_path(p, root)) for p in rows["clean_path"]],
        "distorted_path": [str(resolve_data_path(p, root)) for p in rows["distorted_path"]],
        "severity": rows["severity"],
        "transcript": rows["sentence"],
        "duration_sec": (rows["output_samples"] / 16000).round(4),
    }).sort_values(["utterance_id", "severity"])

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest, index=False, encoding="utf-8")

    used = set(manifest["utterance_id"])
    splits = {name: sorted(set(split.loc[split["split"] == name, "clip_id"]) & used)
              for name in ("train", "val", "test")}
    payload = {
        "source": "data/manifests/cv27_th_speaker_split.csv",
        "level": "speaker (client_id); no client_id or sentence shared between splits",
        "splits": splits,
    }
    args.splits.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.manifest}: {len(manifest)} rows")
    print(manifest.groupby("severity").size().to_dict())
    print(f"wrote {args.splits}: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
