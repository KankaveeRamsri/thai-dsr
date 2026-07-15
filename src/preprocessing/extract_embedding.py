# extract_embedding.py
#
# Purpose:
#   Extract content/phonetic embeddings from Thai speech audio using the
#   pretrained wav2vec2 XLSR-53 model fine-tuned on Thai
#   (airesearch/wav2vec2-large-xlsr-53-th), saving the hidden state of
#   every transformer layer so downstream experiments can pick the layer
#   that works best.
#
# Expected responsibilities:
#   - Load audio from a given .wav path and resample to 16 kHz
#   - Run the audio through the pretrained wav2vec2 model with
#     output_hidden_states=True
#   - Save each layer's hidden state as a separate .npy file in
#     data/embeddings/
#   - Print the shape of every layer so the outputs can be inspected

import argparse
import os

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Pretrained Thai wav2vec2 XLSR-53 checkpoint on HuggingFace.
MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"

# Target sample rate expected by the model.
TARGET_SAMPLE_RATE = 16000

# Where per-layer embeddings get written.
DEFAULT_OUTPUT_DIR = os.path.join("data", "embeddings")


def load_model(device):
    """Load the pretrained feature extractor and wav2vec2 model onto `device`."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()  # inference mode: disables dropout, etc.
    return feature_extractor, model


def load_audio(wav_path):
    """Load a .wav file and resample it to TARGET_SAMPLE_RATE if needed."""
    waveform, sample_rate = torchaudio.load(wav_path)

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


def save_hidden_states(hidden_states, wav_path, output_dir):
    """Save each layer's hidden state to its own .npy file and print its shape."""
    os.makedirs(output_dir, exist_ok=True)

    utterance_id = os.path.splitext(os.path.basename(wav_path))[0]

    for layer_idx, layer_output in enumerate(hidden_states):
        # Move to CPU and drop the batch dimension before saving.
        layer_array = layer_output.squeeze(0).cpu().numpy()

        print(f"Layer {layer_idx:02d} shape: {layer_array.shape}")

        out_path = os.path.join(
            output_dir, f"{utterance_id}_layer{layer_idx:02d}.npy"
        )
        np.save(out_path, layer_array)


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-layer wav2vec2 (Thai XLSR-53) embeddings from a .wav file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input .wav file, e.g. path/to/audio.wav",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save .npy embeddings (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Use CUDA automatically when available, otherwise fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model: {MODEL_NAME}")
    feature_extractor, model = load_model(device)

    print(f"Extracting hidden states from: {args.input}")
    hidden_states = extract_all_hidden_states(
        args.input, feature_extractor, model, device
    )
    print(f"Model produced {len(hidden_states)} hidden state layers (including input embeddings).")

    save_hidden_states(hidden_states, args.input, args.output_dir)
    print(f"Saved all layer embeddings to: {args.output_dir}")


if __name__ == "__main__":
    main()
