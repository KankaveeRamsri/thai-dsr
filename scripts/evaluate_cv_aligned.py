#!/usr/bin/env python3
"""Evaluate on the CV 27.0 speaker split with time-aligned references.

Systems
  distorted  the unprocessed distorted input (floor)
  copysynth  clean mel warped onto the distorted timeline -> HiFi-GAN (ceiling:
             the exact training target of --alignment tempo, vocoded)
  copysynth_ref  mel of the aligned reference waveform -> HiFi-GAN (ceiling
             that isolates the vocoder from the frame-interpolation warp)
  mapper     wav2vec2 layer-9 embedding -> mapper -> HiFi-GAN (--checkpoint)

Every system works on the distorted timeline: the mapper input is the cached
embedding interpolated only for the frame-rate change, so the output has the
distorted clip's mel frame count.

Metrics per (clip, severity)
  mel_l1          mean |mel - aligned clean mel| (mapper: unclamped prediction;
                  distorted: the distorted mel itself; copysynth: 0 by design)
  stoi / pesq     against the ALIGNED reference: the clean waveform
                  time-stretched per tempo segment with the same phase vocoder
                  as the distortion (TempoWarp.stretch_waveform), so both
                  signals share one timeline. The vocoder output differs from
                  the reference length only by mel-frame rounding (< 1 hop);
                  that residual is zero-padded or trimmed and reported.
  stoi_trunc /    the original protocol (src/evaluation/metrics.py): the
  pesq_trunc      unwarped clean reference, both signals truncated to the
                  shorter length, no time alignment. Reported separately.

Writes <out-dir>/<name>/per_utterance.csv (and vocoded audio) under the data
root, and prints mean metrics per severity.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import PESQ_MODE, TARGET_SAMPLE_RATE, compute_pesq, compute_stoi, load_audio_16k  # noqa: E402
from src.inference.run import MEL_CLAMP_MAX, MEL_CLAMP_MIN, get_device, load_mapper  # noqa: E402
from src.evaluation.mapper_eval import load_vocoder  # noqa: E402
from src.utils.config import DEFAULT_MODEL_CONFIG_PATH, DEFAULT_TRAIN_CONFIG_PATH, cv_data_root  # noqa: E402
from src.utils.mel import HIFIGAN_SAMPLE_RATE, compute_mel  # noqa: E402
from src.utils.time_warp import TempoWarp, mel_frame_count  # noqa: E402

SEVERITY_ORDER = ["mild", "moderate", "severe"]


def fit_length(signal, n):
    if len(signal) >= n:
        return signal[:n]
    return np.pad(signal, (0, n - len(signal)))


def safe_pesq(reference, estimate, fn):
    try:
        return float(fn(reference, estimate)), ""
    except Exception as exc:  # pesq raises on e.g. no detected utterance
        return float("nan"), str(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=("distorted", "copysynth", "copysynth_ref", "mapper"), required=True)
    parser.add_argument("--checkpoint", help="mapper checkpoint (for --system mapper)")
    parser.add_argument("--name", help="output folder name (default: the system name)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="default: <data root>/training_manifest.csv")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/manifests/cv27_th_splits.json")
    parser.add_argument("--params", type=Path, default=ROOT / "data/manifests/cv27_th_distortion_params.csv")
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="default: <data root>/embeddings")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data root>/eval")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--limit", type=int, default=0, help="only the first N rows (smoke test)")
    parser.add_argument("--device", default=None, help="default: cuda, then mps, then cpu")
    args = parser.parse_args()
    if args.system == "mapper" and not args.checkpoint:
        parser.error("--system mapper needs --checkpoint")

    root = cv_data_root()
    manifest = pd.read_csv(args.manifest or root / "training_manifest.csv")
    ids = set(json.loads(args.splits.read_text(encoding="utf-8"))["splits"][args.split])
    rows = manifest[manifest["utterance_id"].isin(ids)].sort_values(["severity", "utterance_id"])
    if args.limit:
        rows = rows.head(args.limit)
    segments = pd.read_csv(args.params).set_index(["clip_id", "severity"])["tempo_segments"]
    embeddings_dir = args.embeddings_dir or root / "embeddings"
    out = (args.out_dir or root / "eval") / (args.name or args.system)
    (out / "audio").mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else get_device()
    vocoder = load_vocoder(device) if args.system != "distorted" else None
    mapper = None
    if args.system == "mapper":
        mapper = load_mapper(args.checkpoint, device, DEFAULT_MODEL_CONFIG_PATH,
                             DEFAULT_TRAIN_CONFIG_PATH, layer=args.layer)
    print(f"{args.system}: {len(rows)} rows ({args.split}) on {device} -> {out}", flush=True)

    results, start = [], time.time()
    for i, row in enumerate(rows.itertuples(), 1):
        clean = load_audio_16k(row.clean_path)
        distorted = load_audio_16k(row.distorted_path)
        warp = TempoWarp(segments.loc[(row.utterance_id, row.severity)], TARGET_SAMPLE_RATE)
        reference = warp.stretch_waveform(clean)
        if len(reference) != len(distorted):
            raise ValueError(f"{row.utterance_id}: aligned reference length mismatch")
        n_frames = mel_frame_count(len(distorted), TARGET_SAMPLE_RATE)
        target = warp.warp_frames(compute_mel(clean, sr=TARGET_SAMPLE_RATE).T, n_target=n_frames)

        if args.system == "distorted":
            mel = compute_mel(distorted, sr=TARGET_SAMPLE_RATE).T
            estimate = distorted
        else:
            if args.system == "copysynth":
                mel = target
            elif args.system == "copysynth_ref":
                mel = compute_mel(reference, sr=TARGET_SAMPLE_RATE).T
            else:
                emb = torch.from_numpy(np.load(
                    embeddings_dir / f"{row.utterance_id}_{row.severity}_layer{args.layer:02d}.npy")).float()
                emb = F.interpolate(emb.T[None], size=n_frames, mode="linear", align_corners=False)[0].T
                with torch.no_grad():
                    pred = mapper(emb[None].to(device), torch.tensor([n_frames], device=device))
                mel = pred[0].cpu().numpy()
            voc_in = torch.from_numpy(np.ascontiguousarray(mel.T))[None].float()
            with torch.no_grad():
                wav = vocoder(voc_in.clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX).to(device)).squeeze().cpu().numpy()
            path = out / "audio" / f"{row.utterance_id}_{row.severity}.wav"
            sf.write(path, np.clip(wav, -1, 1), HIFIGAN_SAMPLE_RATE, subtype="PCM_16")
            estimate = load_audio_16k(path)

        residual = len(estimate) - len(reference)
        aligned = fit_length(estimate, len(reference))
        pesq_aligned, pesq_err = safe_pesq(
            reference, aligned, lambda r, e: pesq(TARGET_SAMPLE_RATE, r, e, PESQ_MODE))
        pesq_trunc, _ = safe_pesq(clean, estimate, compute_pesq)
        results.append({
            "utterance_id": row.utterance_id, "severity": row.severity,
            "mel_l1": float(np.abs(mel - target).mean()),
            "stoi": float(stoi(reference, aligned, TARGET_SAMPLE_RATE, extended=False)),
            "pesq": pesq_aligned,
            "stoi_trunc": float(compute_stoi(clean, estimate)),
            "pesq_trunc": pesq_trunc,
            "length_residual": residual, "pesq_error": pesq_err,
        })
        if i % 100 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} {time.time() - start:.0f}s", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(out / "per_utterance.csv", index=False)
    metrics = ["mel_l1", "stoi", "pesq", "stoi_trunc", "pesq_trunc"]
    summary = df.groupby("severity")[metrics].mean().reindex(
        [s for s in SEVERITY_ORDER if s in set(df["severity"])])
    summary.loc["all"] = df[metrics].mean()
    summary["n"] = df.groupby("severity").size().reindex(summary.index).fillna(len(df)).astype(int)
    summary["pesq_failed"] = df.groupby("severity")["pesq"].apply(lambda s: s.isna().sum()) \
        .reindex(summary.index).fillna(df["pesq"].isna().sum()).astype(int)
    print(f"\n{args.name or args.system} ({args.split}); max |length residual| "
          f"{df['length_residual'].abs().max()} samples")
    print(summary.round(4).to_string())
    summary.round(4).to_csv(out / "summary.csv")


if __name__ == "__main__":
    main()
