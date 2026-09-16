# extract_embedding.py
#
# Purpose:
#   Extract content/phonetic embeddings from Thai speech audio using the
#   pretrained wav2vec2 XLSR-53 model fine-tuned on Thai
#   (airesearch/wav2vec2-large-xlsr-53-th), saving the selected hidden state
#   for training or all transformer layers for explicit layer experiments.
#
# Expected responsibilities:
#   - Load audio from a given .wav path and resample to 16 kHz
#   - Run the audio through the pretrained wav2vec2 model with
#     output_hidden_states=True
#   - Save the layer selected in configs/train.yaml by default
#   - Optionally save all layers for layer-selection experiments

import argparse
import glob
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from src.utils.config import (
    DEFAULT_MODEL_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    get_selected_layer,
    load_yaml_config,
)

# Target sample rate expected by the model.
TARGET_SAMPLE_RATE = 16000
NUM_HIDDEN_STATE_LAYERS = 25

# Where per-layer embeddings get written.
DEFAULT_OUTPUT_DIR = os.path.join("data", "embeddings")


def load_model(device):
    """Load the pretrained feature extractor and wav2vec2 model onto `device`."""
    model_name = load_yaml_config(DEFAULT_MODEL_CONFIG_PATH)["encoder"]["content_model"]
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model.to(device)
    model.eval()  # inference mode: disables dropout, etc.
    return feature_extractor, model


def load_audio(wav_path):
    """Load a .wav file and resample it to TARGET_SAMPLE_RATE if needed."""
    audio_data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_data.T)

    # Collapse to mono by averaging channels, if the file is stereo.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    # Return a 1-D numpy array as expected by the HuggingFace feature extractor.
    return waveform.squeeze(0).numpy()


def extract_all_hidden_states(wav_path, feature_extractor, model, device):
    """Run inference and return a tuple of hidden states, one per layer.

    Each element has shape (batch=1, time_steps, hidden_dim). Layer 0 is the
    output of the CNN feature encoder (before any transformer block), and
    the remaining layers are the outputs of each transformer block.
    """
    audio_array = load_audio(wav_path)

    inputs = feature_extractor(
        audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)

    # No gradients needed since we are only doing feature extraction.
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    # outputs.hidden_states is a tuple: (embeddings, layer_1, layer_2, ..., layer_N)
    hidden_states = outputs.hidden_states
    return hidden_states


def save_hidden_states(hidden_states, wav_path, output_dir, layer=None):
    """Save one selected layer, or every layer for layer-selection research."""
    os.makedirs(output_dir, exist_ok=True)

    utterance_id = os.path.splitext(os.path.basename(wav_path))[0]
    layer_indices = range(len(hidden_states)) if layer is None else (layer,)
    for layer_idx in layer_indices:
        layer_output = hidden_states[layer_idx]
        # Move to CPU and drop the batch dimension before saving.
        layer_array = layer_output.squeeze(0).cpu().numpy()

        print(f"Layer {layer_idx:02d} shape: {layer_array.shape}")

        out_path = os.path.join(
            output_dir, f"{utterance_id}_layer{layer_idx:02d}.npy"
        )
        np.save(out_path, layer_array)


def cache_exists(wav_path, output_dir, layer=None):
    """Return whether the requested cache format is already complete."""
    utterance_id = os.path.splitext(os.path.basename(wav_path))[0]
    layer_indices = range(NUM_HIDDEN_STATE_LAYERS) if layer is None else (layer,)
    return all(
        os.path.isfile(
            os.path.join(output_dir, f"{utterance_id}_layer{layer_idx:02d}.npy")
        )
        for layer_idx in layer_indices
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-layer wav2vec2 (Thai XLSR-53) embeddings from a .wav file."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to the input .wav file, e.g. path/to/audio.wav",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Process every .wav file in this directory with one model load.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save .npy embeddings (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to save (default: configs/train.yaml data.layer).",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Save all 25 layers for layer-selection experiments (uses much more disk).",
    )
    args = parser.parse_args()

    selected_layer = get_selected_layer(DEFAULT_TRAIN_CONFIG_PATH) if args.layer is None else args.layer
    if not 0 <= selected_layer <= 24:
        parser.error(f"--layer must be between 0 and 24, got {selected_layer}")

    layer_to_save = None if args.all_layers else selected_layer
    if args.input_dir:
        wav_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
        if not wav_paths:
            parser.error(f"No .wav files found in {args.input_dir}")
    else:
        wav_paths = [args.input]

    pending_paths = [
        path for path in wav_paths
        if not cache_exists(path, args.output_dir, layer=layer_to_save)
    ]
    skipped = len(wav_paths) - len(pending_paths)
    print(f"Found {len(wav_paths)} input file(s); skipping {skipped} cached file(s)")
    if not pending_paths:
        print("All requested embeddings are already cached.")
        return

    # Use CUDA automatically when available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = load_yaml_config(DEFAULT_MODEL_CONFIG_PATH)["encoder"]["content_model"]
    print(f"Loading model: {model_name}")
    feature_extractor, model = load_model(device)

    print(f"Extracting embeddings from {len(pending_paths)} uncached file(s)")
    for index, wav_path in enumerate(pending_paths, start=1):
        print(f"[{index}/{len(pending_paths)}] {wav_path}")
        hidden_states = extract_all_hidden_states(
            wav_path, feature_extractor, model, device
        )
        save_hidden_states(hidden_states, wav_path, args.output_dir, layer=layer_to_save)

    if args.all_layers:
        print(f"Saved all layer embeddings to: {args.output_dir}")
    else:
        print(f"Saved selected layer {selected_layer} embedding to: {args.output_dir}")


if __name__ == "__main__":
    main()
