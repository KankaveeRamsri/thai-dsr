#!/usr/bin/env python3
"""Convert the speaker-split Common Voice clips to 16 kHz wav and distort them.

Reads data/manifests/cv27_th_speaker_split.csv and, for each clip:
  1. decodes <clips-dir>/<clip_id>.mp3, downmixes to mono, resamples to
     16 kHz and writes <out-dir>/clean_16k/<clip_id>.wav (PCM_16)
  2. distorts that wav: severe only for train/val, mild + moderate + severe
     for test, under <out-dir>/distorted/<severity>/<clip_id>_<severity>.wav

Each (clip, severity) gets its own seed, sha256("<seed>:<clip_id>:<severity>"),
so the parameters do not depend on processing order or on --workers. Every
drawn parameter goes to <out-dir>/distortion_params.csv. A file whose
pitch_status is not "ok" (Praat error, or no pitch points so F0 was not
changed) is not written and is marked kept=False.

All audio stays under --out-dir, which should be outside the repository.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocessing.distortion import apply_distortion  # noqa: E402

TARGET_SR = 16000
SEVERITIES_BY_SPLIT = {
    "train": ("severe",),
    "val": ("severe",),
    "test": ("mild", "moderate", "severe"),
}
PARAM_COLUMNS = [
    "clip_id", "client_id", "split", "severity", "seed", "kept", "pitch_status",
    "pitch_error", "formant_shift_ratio", "f0_shift_factor", "jitter_std",
    "jitter_mean", "jitter_sd", "pitch_points", "n_tempo_segments",
    "tempo_rate_min", "tempo_rate_max", "tempo_segments", "snr_db",
    "reverb_wet_mix", "reverb_decay", "peak_before_norm", "peak_normalized",
    "input_samples", "output_samples", "distorted_path",
]
CLEAN_COLUMNS = ["clip_id", "split", "source_sr", "source_channels",
                 "clean_samples", "clean_sec", "clean_path"]


def file_seed(base_seed, clip_id, severity):
    digest = hashlib.sha256(f"{base_seed}:{clip_id}:{severity}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def write_atomic(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.wav")
    sf.write(tmp, audio, TARGET_SR, subtype="PCM_16")
    os.replace(tmp, path)


def convert(mp3_path, wav_path):
    audio, sr = sf.read(mp3_path, dtype="float32", always_2d=True)
    channels = audio.shape[1]
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    write_atomic(wav_path, audio)
    return sr, channels


def process_clip(row, clips_dir, out_dir, base_seed):
    clip_id, split = row["clip_id"], row["split"]
    clean_path = out_dir / "clean_16k" / f"{clip_id}.wav"
    if clean_path.is_file():
        source_sr, channels = "", ""
    else:
        source_sr, channels = convert(clips_dir / f"{clip_id}.mp3", clean_path)
    clean, sr = sf.read(clean_path, dtype="float32")
    clean_row = {"clip_id": clip_id, "split": split, "source_sr": source_sr,
                 "source_channels": channels, "clean_samples": len(clean),
                 "clean_sec": round(len(clean) / sr, 4), "clean_path": str(clean_path)}

    param_rows = []
    for severity in SEVERITIES_BY_SPLIT[split]:
        seed = file_seed(base_seed, clip_id, severity)
        record = {}
        distorted = apply_distortion(clean, sr, severity,
                                     rng=np.random.default_rng(seed), record=record)
        out_path = out_dir / "distorted" / severity / f"{clip_id}_{severity}.wav"
        kept = record["pitch_status"] == "ok"
        if kept:
            write_atomic(out_path, distorted)
        elif out_path.exists():
            out_path.unlink()
        rates = [r for _, _, r in record["tempo_segments"] if r is not None]
        param_rows.append({
            **{k: record.get(k) for k in PARAM_COLUMNS if k in record},
            "clip_id": clip_id, "client_id": row["client_id"], "split": split,
            "severity": severity, "seed": seed, "kept": kept,
            "n_tempo_segments": len(record["tempo_segments"]),
            "tempo_rate_min": min(rates) if rates else None,
            "tempo_rate_max": max(rates) if rates else None,
            "tempo_segments": json.dumps(record["tempo_segments"]),
            "distorted_path": str(out_path) if kept else "",
        })
    return clean_row, param_rows


def pick_rows(rows, per_split):
    if per_split is None:
        return rows
    limits = dict(zip(("train", "val", "test"), per_split))
    ordered = sorted(rows, key=lambda r: hashlib.sha256(r["clip_id"].encode()).hexdigest())
    picked = []
    for name, limit in limits.items():
        picked += [r for r in ordered if r["split"] == name][:limit]
    return picked


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/manifests/cv27_th_speaker_split.csv"))
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-split", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                        help="pilot: only the first N clips of each split (hash order)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if out_dir == repo or repo in out_dir.parents:
        parser.error(f"--out-dir must be outside the repository ({repo})")

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = pick_rows(list(csv.DictReader(handle)), args.per_split)
    print(f"{len(rows)} clips -> {out_dir} with {args.workers} worker(s)", flush=True)

    start = time.time()
    clean_rows, param_rows = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_clip, r, args.clips_dir.expanduser(), out_dir, args.seed)
                   for r in rows]
        for i, future in enumerate(futures, 1):
            clean_row, params = future.result()
            clean_rows.append(clean_row)
            param_rows.extend(params)
            if i % 200 == 0 or i == len(futures):
                print(f"  {i}/{len(futures)} clips, {time.time() - start:.0f}s", flush=True)

    write_csv(out_dir / "clean_16k_index.csv", CLEAN_COLUMNS, clean_rows)
    write_csv(out_dir / "distortion_params.csv", PARAM_COLUMNS, param_rows)

    print("\nfiles per split x severity (kept / excluded) and pitch_status:")
    summary = {}
    for p in param_rows:
        key = (p["split"], p["severity"])
        summary.setdefault(key, {"kept": 0, "excluded": 0, "status": {}})
        summary[key]["kept" if p["kept"] else "excluded"] += 1
        status = summary[key]["status"]
        status[p["pitch_status"]] = status.get(p["pitch_status"], 0) + 1
    for (split, severity), s in sorted(summary.items()):
        print(f"  {split:5s} {severity:8s} kept {s['kept']:5d}  excluded {s['excluded']:3d}  {s['status']}")
    print(f"\nwrote {out_dir / 'distortion_params.csv'} and {out_dir / 'clean_16k_index.csv'}")


if __name__ == "__main__":
    main()
