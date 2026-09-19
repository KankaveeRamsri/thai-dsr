# run.py
#
# Purpose:
#   CLI/script entry point to run the full inference pipeline on a single
#   audio file or a folder: encode -> map -> vocode -> save reconstructed
#   audio.
#
# Expected responsibilities:
#   - Load a trained checkpoint from results/checkpoints/
#   - Run encoder.py -> mapper.py -> vocoder.py on input audio
#   - Save reconstructed output to results/audio_samples/
#   - Accept CLI args for input path, checkpoint path, output path

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIFIGAN_DIR = os.path.join(REPO_ROOT, "vendor", "hifi-gan")

sys.path.insert(0, REPO_ROOT)
from src.models.encoder import Wav2Vec2ContentEncoder  # noqa: E402
from src.models.mapper import build_mapper_from_config  # noqa: E402
from src.training.dataset import interpolate_embedding  # noqa: E402
from src.utils.config import (  # noqa: E402
    DEFAULT_MODEL_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    get_selected_layer,
    load_yaml_config,
)
from src.utils.mel import LOG_CLIP_VALUE, compute_mel  # noqa: E402

# vendor/hifi-gan's models.py/env.py use bare top-level imports, so its
# directory must be on sys.path to import them (same as scripts/test_hifigan.py).
sys.path.insert(0, HIFIGAN_DIR)
from env import AttrDict  # noqa: E402
from models import Generator  # noqa: E402

DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "results", "checkpoints", "best_mapper.pt")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "audio_samples")
DEFAULT_DISTORTED_DIR = os.path.join(REPO_ROOT, "data", "distorted")

# HiFi-GAN was trained on mel-spectrograms in roughly this range; the mapper's
# log(mel + 1e-9) output can stray outside it, so clamp before vocoding.
MEL_CLAMP_MIN = float(np.log(LOG_CLIP_VALUE))
MEL_CLAMP_MAX = 2


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mapper(checkpoint_path, device, model_config_path, train_config_path, layer=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_config = checkpoint.get("model_config") or load_yaml_config(model_config_path)
    model = build_mapper_from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    checkpoint_train_config = checkpoint.get("train_config") or checkpoint.get("config", {})
    checkpoint_layer = checkpoint_train_config.get("data", {}).get("layer")
    configured_layer = get_selected_layer(train_config_path) if layer is None else int(layer)
    if checkpoint_layer is not None and int(checkpoint_layer) != configured_layer:
        raise ValueError(
            "Checkpoint/config layer mismatch: "
            f"checkpoint={checkpoint_layer}, configs/train.yaml={configured_layer}"
        )
    model.eval()
    return model


def load_hifigan(model_dir, device):
    with open(os.path.join(model_dir, "config.json")) as f:
        config = AttrDict(json.load(f))

    generator = Generator(config).to(device)
    checkpoint_path = os.path.join(model_dir, [
        name for name in os.listdir(model_dir) if name.startswith(("generator", "g_"))
    ][0])
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict["generator"])
    generator.eval()
    generator.remove_weight_norm()
    return generator, config


def compute_mel_frame_count(wav_path):
    """Compute the frame count using the same HiFi-GAN mel path as training."""
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return compute_mel(audio, sr=sample_rate).shape[1]


def reconstruct(wav_path, output_dir, encoder, mapper, hifigan, hifigan_config, device):
    """Run the full distorted -> clean-sounding pipeline on one .wav file."""
    name = os.path.splitext(os.path.basename(wav_path))[0]
    output_path = os.path.join(output_dir, f"{name}_reconstructed.wav")

    timings = {}
    t_start = time.perf_counter()

    t0 = time.perf_counter()
    target_mel_len = compute_mel_frame_count(wav_path)
    timings["1. load audio"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    embedding = encoder.encode(wav_path)  # (T_emb, 1024)
    timings["2. extract wav2vec2 embedding"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    embedding = interpolate_embedding(embedding, target_length=target_mel_len)  # (T_mel, 1024)
    embedding = embedding.unsqueeze(0).to(device)  # (1, T_mel, 1024)
    timings["3. interpolate to mel frame rate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        predicted_mel = mapper(embedding)  # (1, T_mel, 80)
    predicted_mel = predicted_mel.clamp(min=MEL_CLAMP_MIN, max=MEL_CLAMP_MAX)
    timings["4. mapper inference"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mel_for_hifigan = predicted_mel.transpose(1, 2)  # (1, 80, T_mel)
    with torch.no_grad():
        waveform = hifigan(mel_for_hifigan)  # (1, 1, T_mel * hop_size)
    timings["5. hifigan vocoding"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    audio = waveform.squeeze().cpu().numpy().astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    os.makedirs(output_dir, exist_ok=True)
    sf.write(output_path, audio, hifigan_config.sampling_rate, subtype="PCM_16")
    timings["6. save output"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_start

    print(f"\nProcessing: {os.path.basename(wav_path)}")
    for step, elapsed in timings.items():
        print(f"  {step:<32} {elapsed:.3f}s")
    print(f"  Saved to: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct clean-sounding speech from dysarthric-distorted audio."
    )
    parser.add_argument("--input", type=str, default=None, help="Path to a distorted .wav file.")
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save reconstructed .wav files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help=f"Process every .wav file in --input_dir instead of a single --input file.",
    )
    parser.add_argument(
        "--input_dir", type=str, default=DEFAULT_DISTORTED_DIR,
        help=f"Directory of distorted .wav files to process in --batch mode (default: {DEFAULT_DISTORTED_DIR})",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Mapper checkpoint path.")
    parser.add_argument("--manifest", type=str, default=None, help="Optional manifest used to select batch inputs.")
    parser.add_argument("--splits", type=str, default=None, help="Optional utterance split JSON for batch filtering.")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None)
    parser.add_argument("--layer", type=int, default=None, help="Override the wav2vec2 layer.")
    parser.add_argument(
        "--hifigan_model",
        type=str,
        default=None,
        help="Optional HiFi-GAN folder name overriding configs/model.yaml.",
    )
    parser.add_argument("--train_config", type=str, default=os.fspath(DEFAULT_TRAIN_CONFIG_PATH))
    parser.add_argument("--model_config", type=str, default=os.fspath(DEFAULT_MODEL_CONFIG_PATH))
    args = parser.parse_args()

    if not args.batch and args.input is None:
        parser.error("Provide --input path/to/audio.wav, or use --batch to process data/distorted/.")

    device = get_device()
    print(f"Using device: {device}")
    model_config = load_yaml_config(args.model_config)
    vocoder_config = model_config["vocoder"]
    if str(vocoder_config["type"]).lower() != "hifigan":
        raise ValueError(f"Unsupported vocoder type: {vocoder_config['type']}")

    t0 = time.perf_counter()
    configured_layer = get_selected_layer(args.train_config) if args.layer is None else args.layer
    encoder = Wav2Vec2ContentEncoder(
        device=device,
        train_config_path=args.train_config,
        model_config_path=args.model_config,
        layer=configured_layer,
    )
    print(f"Loaded wav2vec2 encoder (layer {configured_layer}) in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    mapper = load_mapper(
        args.checkpoint, device, args.model_config, args.train_config,
        layer=configured_layer,
    )
    print(f"Loaded mapper checkpoint from {args.checkpoint} in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    if args.hifigan_model:
        hifigan_dir = os.path.join(HIFIGAN_DIR, "checkpoints", args.hifigan_model)
    else:
        hifigan_dir = vocoder_config["checkpoint"]
        if not os.path.isabs(hifigan_dir):
            hifigan_dir = os.path.join(REPO_ROOT, hifigan_dir)
    hifigan, hifigan_config = load_hifigan(hifigan_dir, device)
    if int(hifigan_config.sampling_rate) != int(vocoder_config["sample_rate"]):
        raise ValueError(
            "HiFi-GAN/config sample-rate mismatch: "
            f"checkpoint={hifigan_config.sampling_rate}, config={vocoder_config['sample_rate']}"
        )
    print(f"Loaded HiFi-GAN ({hifigan_dir}) in {time.perf_counter() - t0:.2f}s")

    if args.batch:
        if args.manifest:
            selected_ids = None
            if args.splits or args.split:
                if not (args.splits and args.split):
                    parser.error("--splits and --split must be provided together")
                with open(args.splits, encoding="utf-8") as handle:
                    selected_ids = set(json.load(handle)["splits"][args.split])
            with open(args.manifest, encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            wav_paths = sorted({
                row["distorted_path"] for row in rows
                if selected_ids is None or row["utterance_id"] in selected_ids
            })
            print(f"\nBatch mode: {len(wav_paths)} files selected from {args.manifest}")
        else:
            wav_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
            print(f"\nBatch mode: {len(wav_paths)} files found in {args.input_dir}")
    else:
        wav_paths = [args.input]

    for wav_path in wav_paths:
        reconstruct(wav_path, args.output_dir, encoder, mapper, hifigan, hifigan_config, device)


if __name__ == "__main__":
    main()
