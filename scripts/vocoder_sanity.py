"""Diagnose HiFi-GAN with clean reference mels, without an encoder or mapper."""

import csv
import json
from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/hifi-gan"))
from env import AttrDict
from models import Generator
from src.utils.mel import compute_mel
from src.evaluation.metrics import compute_stoi, compute_pesq, load_audio_16k


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device("cpu")
    previous = ROOT / "results/audio_compare/w6_vocoder_ft"
    output = ROOT / "results/audio_compare/w6_vocoder_sanity"
    provenance = json.loads((previous / "provenance.json").read_text())
    rows = provenance["utterances"]
    assert len(rows) == 8
    assert {r["utterance_id"] for r in rows} == {p.name for p in previous.iterdir() if p.is_dir()}
    generators = {}
    configs = {}
    for label, entry in provenance["checkpoints"].items():
        checkpoint = ROOT / entry["path"]
        config = AttrDict(json.loads(checkpoint.with_name("config.json").read_text()))
        configs[label] = config
        model = Generator(config).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["generator"])
        model.eval()
        model.remove_weight_norm()
        generators[label] = model
    assert configs["baseline"] == configs["finetuned"]
    config = configs["baseline"]
    assert (config.sampling_rate, config.num_mels, config.n_fft, config.hop_size,
            config.win_size, config.fmin, config.fmax) == (22050, 80, 1024, 256, 1024, 0, 8000)
    results = []
    details = []
    log = ["Ground-truth mel -> HiFi-GAN sanity test (CPU)",
           "No encoder or mapper loaded. Existing compute_mel defaults; no mel clamping or normalization.",
           "Metrics use saved PCM_16 output and previous ground_truth.wav, resampled to 16 kHz.",
           "Existing shared-length truncation; no time or gain alignment."]
    for row in rows:
        uid = row["utterance_id"]
        folder = output / uid
        folder.mkdir(parents=True, exist_ok=True)
        audio, sr = sf.read(ROOT / row["clean_path"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        # Verify the source reproduces the reference used in the preceding test.
        reference_22k = audio if sr == config.sampling_rate else librosa.resample(
            audio, orig_sr=sr, target_sr=config.sampling_rate)
        old_reference, old_sr = sf.read(previous / uid / "ground_truth.wav", dtype="float32")
        assert old_sr == config.sampling_rate and len(old_reference) == len(reference_22k)
        assert np.max(np.abs(np.clip(reference_22k, -1, 1) - old_reference)) <= 1 / 32768 + 1e-7
        mel = compute_mel(audio, sr=sr)
        assert mel.ndim == 2 and mel.shape[0] == config.num_mels
        assert np.isfinite(mel).all() and float(mel.std()) > 0
        tensor = torch.from_numpy(mel).unsqueeze(0).to(device)
        stats = {"utterance_id": uid, "clean_path": row["clean_path"],
                 "shape": list(tensor.shape), "min": float(mel.min()),
                 "max": float(mel.max()), "mean": float(mel.mean()), "std": float(mel.std()),
                 "finite": bool(np.isfinite(mel).all())}
        line = (f"{uid}: mel shape={tuple(tensor.shape)}, min={mel.min():.6f}, "
                f"max={mel.max():.6f}, mean={mel.mean():.6f}, std={mel.std():.6f}, finite=True")
        print(line, flush=True)
        log.append(line)
        result = {"utterance_id": uid}
        reference = load_audio_16k(previous / uid / "ground_truth.wav")
        for label, generator in generators.items():
            with torch.inference_mode():
                waveform = generator(tensor).squeeze().cpu().numpy()
            assert np.isfinite(waveform).all()
            assert len(waveform) == mel.shape[1] * config.hop_size
            stats[label] = {"peak": float(np.abs(waveform).max()),
                            "rms": float(np.sqrt(np.mean(waveform ** 2))),
                            "clipped_fraction": float(np.mean(np.abs(waveform) > 1))}
            path = folder / f"vocoder_only_{label}.wav"
            sf.write(path, np.clip(waveform, -1, 1), config.sampling_rate, subtype="PCM_16")
            estimate = load_audio_16k(path)
            result[f"stoi_{label}"] = float(compute_stoi(reference, estimate))
            result[f"pesq_{label}"] = float(compute_pesq(reference, estimate))
            line = (f"  {label}: STOI={result['stoi_' + label]:.6f}, "
                    f"PESQ={result['pesq_' + label]:.6f}, waveform={stats[label]}")
            print(line, flush=True)
            log.append(line)
        results.append(result)
        details.append(stats)
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output / "mel_stats.json").write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n")
    for metric in ("stoi", "pesq"):
        for label in generators:
            values = np.array([r[f"{metric}_{label}"] for r in results])
            assert np.isfinite(values).all()
            line = f"{metric.upper()} {label}: {values.mean():.6f} +/- {values.std(ddof=1):.6f} (sample std)"
            print(line, flush=True)
            log.append(line)
    log.append("No subjective listening judgment performed; use WAV files for audible confirmation.")
    (output / "sanity_report.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
