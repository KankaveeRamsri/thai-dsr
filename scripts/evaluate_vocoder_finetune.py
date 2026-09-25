"""Compare two vocoders on identical layer-9 mapper predictions (no training)."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference.run import (
    AttrDict, Generator, MEL_CLAMP_MAX, MEL_CLAMP_MIN,
    Wav2Vec2ContentEncoder, compute_mel_frame_count, get_device,
    interpolate_embedding, load_mapper,
)
from src.evaluation.metrics import (
    compute_pesq, compute_snr, compute_stoi, load_audio_16k,
)

OUTPUT = ROOT / "results/audio_compare/w6_vocoder_ft"
MAPPER = ROOT / "results/checkpoints/mapper_layer9.pt"
CHECKPOINTS = {
    "baseline": ROOT / "vendor/hifi-gan/checkpoints/UNIVERSAL_V1/g_02500000",
    "finetuned": ROOT / "results/checkpoints/hifigan_thai/g_00010000",
}
METRICS = {"stoi": compute_stoi, "pesq": compute_pesq, "snr": compute_snr}


def digest(path):
    checksum = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def select_rows():
    splits = json.loads((ROOT / "data/splits_w5.json").read_text())["splits"]
    test = set(splits["test"])
    assert not test.intersection(splits["train"] + splits["val"])
    with (ROOT / "data/manifest_w5.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["utterance_id"] in test and r["severity"] == "severe"]
    rows.sort(key=lambda r: (float(r["duration_sec"]), r["utterance_id"]))
    # Cover duration quantiles, then ensure every available speaker group appears.
    selected = [rows[i] for i in np.rint(np.linspace(0, len(rows) - 1, 8)).astype(int)]
    for speaker in sorted({r["speaker_id"] for r in rows}):
        if any(r["speaker_id"] == speaker for r in selected):
            continue
        candidate = next(r for r in rows if r["speaker_id"] == speaker)
        replaceable = [i for i, r in enumerate(selected)
                       if sum(s["speaker_id"] == r["speaker_id"] for s in selected) > 1]
        index = min(replaceable, key=lambda i: abs(
            float(selected[i]["duration_sec"]) - float(candidate["duration_sec"])))
        selected[index] = candidate
    assert len({r["utterance_id"] for r in selected}) == 8
    return sorted(selected, key=lambda r: float(r["duration_sec"]))


def main():
    os.chdir(ROOT)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = get_device()
    rows = select_rows()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configs = {k: json.loads(p.with_name("config.json").read_text())
               for k, p in CHECKPOINTS.items()}
    assert configs["baseline"] == configs["finetuned"], "Vocoder configs differ"
    config = AttrDict(configs["baseline"])
    sample_rate = int(config.sampling_rate)
    print(f"Device: {device}; output: {sample_rate} Hz PCM_16; layer: 9", flush=True)
    encoder = Wav2Vec2ContentEncoder(device=device, layer=9)
    mapper = load_mapper(MAPPER, device, ROOT / "configs/model.yaml",
                         ROOT / "configs/train.yaml", layer=9)
    vocoders = {}
    for label, path in CHECKPOINTS.items():
        generator = Generator(config).to(device)
        generator.load_state_dict(torch.load(path, map_location=device, weights_only=True)["generator"])
        generator.eval()
        generator.remove_weight_norm()
        vocoders[label] = generator
    provenance = {
        "device": str(device), "layer": 9, "severity": "severe",
        "mapper": str(MAPPER.relative_to(ROOT)), "mapper_sha256": digest(MAPPER),
        "checkpoints": {k: {"path": str(p.relative_to(ROOT)), "sha256": digest(p)}
                        for k, p in CHECKPOINTS.items()},
        "selection": "Eight duration quantiles, ensuring both manifest speaker groups appear",
        "speaker_limitation": "common_voice is a pooled label, not a verified speaker identity",
        "metric_protocol": "Saved PCM_16 audio resampled to 16 kHz; shared-length truncation; no time/gain alignment",
        "utterances": [],
    }
    results = []
    for index, row in enumerate(rows, 1):
        uid = row["utterance_id"]
        folder = OUTPUT / uid
        folder.mkdir(exist_ok=True)
        distorted = ROOT / row["distorted_path"]
        embedding = encoder.encode(str(distorted))
        frames = compute_mel_frame_count(str(distorted))
        inputs = interpolate_embedding(embedding, frames).unsqueeze(0).to(device)
        with torch.inference_mode():
            mel = mapper(inputs).clamp(MEL_CLAMP_MIN, MEL_CLAMP_MAX).transpose(1, 2)
            for label, generator in vocoders.items():
                audio = generator(mel).squeeze().cpu().numpy().astype(np.float32)
                assert np.isfinite(audio).all()
                sf.write(folder / f"{label}.wav", np.clip(audio, -1, 1), sample_rate, subtype="PCM_16")
        clean, sr = sf.read(ROOT / row["clean_path"], dtype="float32", always_2d=True)
        clean = clean.mean(axis=1)
        if sr != sample_rate:
            clean = librosa.resample(clean, orig_sr=sr, target_sr=sample_rate)
        sf.write(folder / "ground_truth.wav", clean, sample_rate, subtype="PCM_16")
        reference = load_audio_16k(folder / "ground_truth.wav")
        result = {"utterance_id": uid}
        for label in CHECKPOINTS:
            estimate = load_audio_16k(folder / f"{label}.wav")
            for metric, function in METRICS.items():
                result[f"{metric}_{label}"] = float(function(reference, estimate))
        assert all(np.isfinite(v) for k, v in result.items() if k != "utterance_id")
        results.append(result)
        provenance["utterances"].append({**row, "distorted_sha256": digest(distorted),
            "embedding_sha256": hashlib.sha256(embedding.tobytes()).hexdigest(),
            "mel_sha256": hashlib.sha256(mel.cpu().numpy().tobytes()).hexdigest(),
            "mel_frames": frames})
        print(f"[{index}/8] {uid}: " + " | ".join(
            f"{m} {result[m + '_baseline']:.4f} -> {result[m + '_finetuned']:.4f}"
            for m in METRICS), flush=True)
    columns = ["utterance_id"] + [f"{m}_{label}" for m in METRICS for label in CHECKPOINTS]
    with (ROOT / "results/w6_vocoder_finetune_eval.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    lines = ["W6 vocoder fine-tuning evaluation", "Evaluated: 8 TEST utterances; severe distortion.",
             "Pipeline: Thai XLSR layer 9 -> mapper_layer9.pt -> HiFi-GAN.",
             "Baseline: UNIVERSAL_V1/g_02500000; fine-tuned: hifigan_thai/g_00010000.",
             "Each pair shares the exact same distorted input, embedding, and predicted mel.",
             "Audio: 22050 Hz PCM_16. Metrics: 16 kHz, PESQ wideband, existing W5 functions.",
             "Alignment: truncate to common length; no time or gain alignment (SNR is waveform-sensitive).",
             "Selection spans duration quantiles and both manifest speaker groups; Common Voice speaker IDs are pooled.",
             "", "Metric       Baseline mean +/- std   Finetuned mean +/- std    Delta (FT - baseline)"]
    deltas = []
    for metric in METRICS:
        base = np.array([r[f"{metric}_baseline"] for r in results])
        fine = np.array([r[f"{metric}_finetuned"] for r in results])
        delta = float((fine - base).mean())
        deltas.append(delta)
        lines.append(f"{metric.upper():<12} {base.mean():.6f} +/- {base.std(ddof=1):.6f}    "
                     f"{fine.mean():.6f} +/- {fine.std(ddof=1):.6f}    {delta:+.6f}")
    verdict = "better" if all(d > 0 for d in deltas) else "worse" if all(d < 0 for d in deltas) else "mixed"
    lines += ["", "Std is sample standard deviation (ddof=1).", f"Overall objective result: {verdict}.",
              "Notable regressions (screening thresholds: STOI <= -0.02, PESQ <= -0.10, or SNR <= -1 dB):"]
    flagged = []
    for row in results:
        changes = {m: row[f"{m}_finetuned"] - row[f"{m}_baseline"] for m in METRICS}
        if any(changes[m] <= limit for m, limit in {"stoi": -0.02, "pesq": -0.1, "snr": -1}.items()):
            flagged.append(row["utterance_id"] + ": " + ", ".join(f"delta {m}={d:+.6f}" for m, d in changes.items()))
    lines += flagged or ["None at the stated thresholds."]
    lines += ["These flags identify candidates for listening, not confirmed audible artifacts.",
              "No subjective listening judgment was performed. Eight samples are exploratory, not a full test-set estimate."]
    report = "\n".join(lines) + "\n"
    report_path = ROOT / "results/logs/w6_vocoder_eval_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
