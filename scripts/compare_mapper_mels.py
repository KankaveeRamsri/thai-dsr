"""Read-only mapper diagnostic; no vocoder inference or training."""
import json
import hashlib
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.run import (load_mapper, compute_mel_frame_count,
                               MEL_CLAMP_MIN, MEL_CLAMP_MAX)
from src.models.encoder import Wav2Vec2ContentEncoder
from src.training.dataset import interpolate_embedding
from src.utils.mel import compute_mel


def stats(mel):
    return dict(shape=[1, *mel.shape], min=float(mel.min()), max=float(mel.max()),
                mean=float(mel.mean()), std=float(mel.std()),
                mean_temporal_std=float(mel.std(axis=1).mean()))


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device("cpu")
    provenance = json.loads((ROOT / "results/audio_compare/w6_vocoder_ft/provenance.json").read_text())
    output = ROOT / "results/mel_compare"
    output.mkdir(parents=True, exist_ok=True)
    encoder = Wav2Vec2ContentEncoder(device=device, layer=9)
    mapper = load_mapper(ROOT / provenance["mapper"], device,
                         ROOT / "configs/model.yaml", ROOT / "configs/train.yaml", layer=9)
    records = []
    log = ["Mapper mel comparison: same eight utterances and mapper_layer9.pt; CPU.",
           "All reported shapes use vocoder layout (1, 80, T); native mapper output is (1, T, 80).",
           "Inference interpolates embeddings to distorted-audio mel length; no mel time cropping/padding before vocoder.",
           "Training interpolates embeddings to CLEAN mel length. Evaluation truncates waveforms to shared length.",
           "Plots use the same fixed natural-log color scale [-12, 2] and shared seconds axis, without time warping."]
    for row in provenance["utterances"]:
        uid = row["utterance_id"]
        embedding = encoder.encode(str(ROOT / row["distorted_path"]))
        frames = compute_mel_frame_count(str(ROOT / row["distorted_path"]))
        inputs = interpolate_embedding(embedding, frames).unsqueeze(0).to(device)
        with torch.inference_mode():
            native = mapper(inputs)
            clamped = native.clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX).transpose(1, 2)
        raw = native[0].T.cpu().numpy()
        predicted = clamped[0].cpu().numpy()
        audio, sr = sf.read(ROOT / row["clean_path"], dtype="float32", always_2d=True)
        clean = compute_mel(audio.mean(axis=1), sr=sr)
        assert all(np.isfinite(m).all() for m in (raw, predicted, clean))
        record = dict(utterance_id=uid, predicted_raw=stats(raw), predicted_vocoder=stats(predicted),
                      clean=stats(clean), clamped_fraction=float(np.mean(raw != predicted)),
                      time_difference_frames=frames-clean.shape[1],
                      matches_previous_mel_hash=hashlib.sha256(clamped.cpu().numpy().tobytes()).hexdigest() == row["mel_sha256"])
        records.append(record)
        line = uid + "\n" + "\n".join(f"  {k}: {record[k]}" for k in
            ("predicted_raw", "predicted_vocoder", "clean", "clamped_fraction", "time_difference_frames", "matches_previous_mel_hash"))
        print(line, flush=True)
        log.append(line)
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
        duration = max(frames, clean.shape[1]) * 256 / 22050
        for ax, mel, title in zip(axes, (predicted, clean), ("Predicted mel (after pipeline clamp)", "Ground-truth clean mel")):
            im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest",
                           extent=(0, mel.shape[1]*256/22050, 0, 80), cmap="magma", vmin=-12, vmax=2)
            ax.set_xlim(0, duration)
            ax.set_facecolor("#dddddd")
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Mel bin")
            ax.set_title(title + f"\n(1, 80, {mel.shape[1]})  min={mel.min():.2f} max={mel.max():.2f}\nmean={mel.mean():.2f} std={mel.std():.2f}", fontsize=10)
        # ASCII title avoids unavailable Thai glyphs; full ID is preserved in the filename.
        fig.suptitle(uid if uid.isascii() else uid.split("_")[0] + " (Thai recording)")
        fig.colorbar(im, ax=axes, label="Natural log mel magnitude")
        fig.savefig(output / f"{uid}.png", dpi=150)
        plt.close(fig)
    (output / "stats.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    (output / "report.txt").write_text("\n\n".join(log) + "\n")


if __name__ == "__main__":
    main()
