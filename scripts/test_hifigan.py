#!/usr/bin/env python3
# test_hifigan.py
#
# Purpose:
#   Smoke-test the vendored HiFi-GAN checkpoint (vendor/hifi-gan/) so we know
#   the vocoder loads and runs before wiring it into the inference pipeline.
#
# Expected responsibilities:
#   - Load the pretrained HiFi-GAN generator (LJ_FT_T2_V1 by default)
#   - Run inference on a dummy mel-spectrogram
#   - Print the resulting waveform shape
#   - Save the generated audio to results/audio_samples/hifigan_test.wav

import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIFIGAN_DIR = os.path.join(REPO_ROOT, "vendor", "hifi-gan")
CHECKPOINTS_DIR = os.path.join(HIFIGAN_DIR, "checkpoints")
DEFAULT_MODEL = "LJ_FT_T2_V1"
OUTPUT_WAV = os.path.join(REPO_ROOT, "results", "audio_samples", "hifigan_test.wav")

# vendor/hifi-gan's models.py/env.py use bare top-level imports (`from models
# import Generator`), so its directory must be on sys.path to import them.
sys.path.insert(0, HIFIGAN_DIR)
from env import AttrDict  # noqa: E402
from models import Generator  # noqa: E402


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_generator(model_name, device):
    """Load a HiFi-GAN generator checkpoint from vendor/hifi-gan/checkpoints/<model_name>/."""
    model_dir = os.path.join(CHECKPOINTS_DIR, model_name)

    with open(os.path.join(model_dir, "config.json")) as f:
        config = AttrDict(json.load(f))

    generator = Generator(config).to(device)

    checkpoint_path = os.path.join(model_dir, [
        name for name in os.listdir(model_dir) if name.startswith(("generator", "g_"))
    ][0])
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict["generator"])

    generator.eval()
    generator.remove_weight_norm()
    return generator, config


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the vendored HiFi-GAN generator.")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Checkpoint folder name under {CHECKPOINTS_DIR} (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--output", type=str, default=OUTPUT_WAV, help="Path to save the test .wav")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    generator, config = load_generator(args.model, device)

    # Dummy mel-spectrogram: (batch=1, n_mels=80, T=100), matching HiFi-GAN's
    # expected (B, num_mels, T) input layout.
    dummy_mel = torch.randn(1, 80, 100, device=device)

    with torch.no_grad():
        waveform = generator(dummy_mel)

    print(f"Input mel shape:    {tuple(dummy_mel.shape)}")
    print(f"Output waveform shape: {tuple(waveform.shape)}")

    audio = waveform.squeeze().cpu().numpy().astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sf.write(args.output, audio, config.sampling_rate, subtype="PCM_16")
    print(f"Saved test audio to: {args.output}")


if __name__ == "__main__":
    main()
