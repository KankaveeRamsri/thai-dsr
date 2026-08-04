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
import glob
import json
import os
import sys
import time

import librosa
import numpy as np
import soundfile as sf
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HIFIGAN_DIR = os.path.join(REPO_ROOT, "vendor", "hifi-gan")

sys.path.insert(0, REPO_ROOT)
from src.models.encoder import Wav2Vec2ContentEncoder  # noqa: E402
from src.models.mapper import MapperModel  # noqa: E402
from src.training.dataset import HOP_LENGTH, MEL_SAMPLE_RATE, interpolate_embedding  # noqa: E402

# vendor/hifi-gan's models.py/env.py use bare top-level imports, so its
# directory must be on sys.path to import them (same as scripts/test_hifigan.py).
sys.path.insert(0, HIFIGAN_DIR)
from env import AttrDict  # noqa: E402
from models import Generator  # noqa: E402

DEFAULT_LAYER = 9
DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "results", "checkpoints", "best_mapper.pt")
DEFAULT_HIFIGAN_MODEL = "LJ_FT_T2_V1"
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "results", "audio_samples")
DEFAULT_DISTORTED_DIR = os.path.join(REPO_ROOT, "data", "distorted")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_mapper(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MapperModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_hifigan(model_name, device):
    model_dir = os.path.join(HIFIGAN_DIR, "checkpoints", model_name)

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
    """Number of mel frames librosa would produce for this audio at MEL_SAMPLE_RATE.

    Mirrors librosa's center=True STFT frame formula (1 + len(y) // hop_length)
    used by DysarthricDataset._extract_clean_mel during training, so the mapper
    receives an embedding interpolated to the same frame rate it was trained on.
    """
    audio, sample_rate = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != MEL_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=MEL_SAMPLE_RATE)
    return 1 + len(audio) // HOP_LENGTH


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
        help=f"Process every .wav file in {DEFAULT_DISTORTED_DIR} instead of a single --input file.",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Mapper checkpoint path.")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="wav2vec2 layer to extract.")
    parser.add_argument("--hifigan_model", type=str, default=DEFAULT_HIFIGAN_MODEL, help="HiFi-GAN checkpoint folder name.")
    args = parser.parse_args()

    if not args.batch and args.input is None:
        parser.error("Provide --input path/to/audio.wav, or use --batch to process data/distorted/.")

    device = get_device()
    print(f"Using device: {device}")

    t0 = time.perf_counter()
    encoder = Wav2Vec2ContentEncoder(layer=args.layer, device=device)
    print(f"Loaded wav2vec2 encoder (layer {args.layer}) in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    mapper = load_mapper(args.checkpoint, device)
    print(f"Loaded mapper checkpoint from {args.checkpoint} in {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    hifigan, hifigan_config = load_hifigan(args.hifigan_model, device)
    print(f"Loaded HiFi-GAN ({args.hifigan_model}) in {time.perf_counter() - t0:.2f}s")

    if args.batch:
        wav_paths = sorted(glob.glob(os.path.join(DEFAULT_DISTORTED_DIR, "*.wav")))
        print(f"\nBatch mode: {len(wav_paths)} files found in {DEFAULT_DISTORTED_DIR}")
    else:
        wav_paths = [args.input]

    for wav_path in wav_paths:
        reconstruct(wav_path, args.output_dir, encoder, mapper, hifigan, hifigan_config, device)


if __name__ == "__main__":
    main()
