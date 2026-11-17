#!/usr/bin/env python3
"""Precompute clean and tempo-warped target mels for the CV 27.0 training manifest.

Writes <out-dir>/clean/<utterance_id>.npy (clean mel, (T, 80)) and
<out-dir>/tempo/<utterance_id>_<severity>.npy (clean mel warped onto the
distorted timeline) using DysarthricDataset's own methods, so training with
--mel-cache-dir gives exactly the items it would compute on the fly.
Default out-dir: <data root>/mel_cache (outside the repository).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.training.dataset import DysarthricDataset  # noqa: E402
from src.utils.config import cv_data_root  # noqa: E402

_DATASET = None


def _init(manifest, params):
    global _DATASET
    _DATASET = DysarthricDataset(manifest_path=manifest, embeddings_dir="/unused", severity=None,
                                 layer=9, alignment="tempo", distortion_params=params)


def _cache_clip(args):
    clip_id, indices, out_dir = args
    row = _DATASET.manifest.iloc[indices[0]]
    clean = _DATASET._extract_clean_mel(row["clean_path"]).astype(np.float32)
    np.save(Path(out_dir) / "clean" / f"{clip_id}.npy", clean)
    for index in indices:
        row = _DATASET.manifest.iloc[index]
        target = _DATASET.aligned_target(row, clean).astype(np.float32)
        np.save(Path(out_dir) / "tempo" / f"{clip_id}_{row['severity']}.npy", target)
    return len(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <data root>/training_manifest.csv")
    parser.add_argument("--params", type=Path, default=ROOT / "data/manifests/cv27_th_distortion_params.csv")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data root>/mel_cache")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    root = cv_data_root()
    manifest = args.manifest or root / "training_manifest.csv"
    out_dir = args.out_dir or root / "mel_cache"
    for sub in ("clean", "tempo"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    _init(manifest, args.params)
    groups = _DATASET.manifest.groupby("utterance_id").indices
    jobs = [(clip, list(idx), str(out_dir)) for clip, idx in sorted(groups.items())]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(manifest, args.params)) as pool:
        targets = sum(pool.map(_cache_clip, jobs, chunksize=16))
    print(f"cached {len(jobs)} clean mels and {targets} tempo targets in {out_dir}")


if __name__ == "__main__":
    main()
