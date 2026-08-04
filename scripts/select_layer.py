#!/usr/bin/env python3
# select_layer.py
#
# Purpose:
#   Pick the wav2vec2 (Thai XLSR-53) layer that best captures *content*
#   rather than acoustic noise, by measuring how much each layer's
#   representation changes when the same utterance is spoken with
#   dysarthric-style distortion (mild/moderate/severe) instead of cleanly.
#
# Expected responsibilities:
#   - Load the clean per-layer embeddings already extracted in W1
#     (data/embeddings/{utterance_id}_layer{NN}.npy)
#   - Extract matching per-layer embeddings for every distorted .wav in
#     data/distorted/, using the same model as
#     src/preprocessing/extract_embedding.py, caching them to
#     data/embeddings/distorted/ so re-runs are fast
#   - For each layer (0-24), compute cosine similarity, L2 distance, and
#     variance across utterances between clean and distorted embeddings
#   - Plot all three metrics across layers to results/layer_comparison.png
#   - Print the layer with the highest clean-vs-distorted cosine similarity
#     as the recommended content-representation layer

import argparse
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

# Same pretrained checkpoint used for the clean W1 embeddings, so distorted
# embeddings live in the same space and are directly comparable.
MODEL_NAME = "airesearch/wav2vec2-large-xlsr-53-th"
TARGET_SAMPLE_RATE = 16000
NUM_LAYERS = 25  # CNN feature encoder output (layer 0) + 24 transformer blocks

EMBEDDINGS_DIR = os.path.join("data", "embeddings")
DISTORTED_DIR = os.path.join("data", "distorted")
DISTORTED_EMBEDDINGS_DIR = os.path.join("data", "embeddings", "distorted")
DEFAULT_OUTPUT_PLOT = os.path.join("results", "layer_comparison.png")

SEVERITIES = ("mild", "moderate", "severe")
SEVERITY_SUFFIX_RE = re.compile(r"^(.*)_(mild|moderate|severe)$")

# Palette + chrome from the project's dataviz reference (validated categorical
# slots 1/2/3 and neutral chrome for a light chart surface).
COLOR_COSINE = "#2a78d6"  # blue
COLOR_L2 = "#eb6834"  # orange
COLOR_VARIANCE = "#1baf7a"  # aqua
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_model(device):
    """Load the pretrained feature extractor and wav2vec2 model onto `device`."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return feature_extractor, model


def load_audio(wav_path):
    """Load a .wav file and resample it to TARGET_SAMPLE_RATE if needed.

    Uses soundfile rather than torchaudio.load, since torchaudio>=2.9 needs
    the optional torchcodec backend for file I/O that isn't installed here.
    """
    audio_data, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio_data.T)  # (channels, time)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def extract_all_hidden_states(wav_path, feature_extractor, model, device):
    """Run inference and return a tuple of per-layer hidden states."""
    audio_array = load_audio(wav_path)

    inputs = feature_extractor(
        audio_array, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)

    return outputs.hidden_states


def discover_utterance_ids():
    """List utterance ids from the clean embeddings already saved in W1."""
    pattern = os.path.join(EMBEDDINGS_DIR, "*_layer00.npy")
    ids = sorted(
        os.path.basename(p)[: -len("_layer00.npy")] for p in glob.glob(pattern)
    )
    if not ids:
        raise RuntimeError(f"No clean embeddings found in {EMBEDDINGS_DIR}")
    return ids


def discover_distorted_files(utterance_id):
    """Return {severity: wav_path} for one utterance from data/distorted/."""
    files = {}
    for wav_path in glob.glob(os.path.join(DISTORTED_DIR, f"{utterance_id}_*.wav")):
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        match = SEVERITY_SUFFIX_RE.match(stem)
        if match and match.group(1) == utterance_id:
            files[match.group(2)] = wav_path
    return files


def ensure_distorted_embeddings(utterance_ids, force=False):
    """Extract wav2vec2 embeddings for every distorted .wav, caching to disk.

    Mirrors src/preprocessing/extract_embedding.py's save convention, so the
    distorted embeddings can be reused across runs the same way the clean
    W1 embeddings already are.
    """
    os.makedirs(DISTORTED_EMBEDDINGS_DIR, exist_ok=True)

    todo = []
    for utterance_id in utterance_ids:
        distorted_files = discover_distorted_files(utterance_id)
        for severity in SEVERITIES:
            wav_path = distorted_files.get(severity)
            if wav_path is None:
                raise RuntimeError(
                    f"Missing {severity} distorted audio for '{utterance_id}' in {DISTORTED_DIR}"
                )
            stem = f"{utterance_id}_{severity}"
            last_layer_path = os.path.join(
                DISTORTED_EMBEDDINGS_DIR, f"{stem}_layer{NUM_LAYERS - 1:02d}.npy"
            )
            if force or not os.path.exists(last_layer_path):
                todo.append((stem, wav_path))

    if not todo:
        print("Distorted embeddings already cached, skipping extraction.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")
    feature_extractor, model = load_model(device)

    for stem, wav_path in todo:
        print(f"Extracting distorted embeddings: {wav_path}")
        hidden_states = extract_all_hidden_states(wav_path, feature_extractor, model, device)
        for layer_idx, layer_output in enumerate(hidden_states):
            layer_array = layer_output.squeeze(0).cpu().numpy()
            out_path = os.path.join(
                DISTORTED_EMBEDDINGS_DIR, f"{stem}_layer{layer_idx:02d}.npy"
            )
            np.save(out_path, layer_array)


def load_pooled(path):
    """Mean-pool a (time, hidden_dim) embedding over time into one vector."""
    array = np.load(path)
    if array.ndim == 2:
        array = array.mean(axis=0)
    return array


def cosine_similarity(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def l2_distance(a, b):
    return float(np.linalg.norm(a - b))


def compute_layer_metrics(utterance_ids):
    """For each layer, compare clean vs. distorted pooled embeddings.

    Returns a list of dicts, one per layer:
      - cosine_similarity: mean cosine sim between clean and distorted,
        averaged over every (utterance, severity) pair. Higher means the
        layer's content representation survives dysarthric distortion.
      - l2_distance: mean L2 distance over the same pairs. Lower is better.
      - variance: variance, across the 10 utterances, of each utterance's
        mean cosine similarity (averaged over its 3 severities). Lower means
        the layer's robustness is consistent regardless of what was said,
        not just good on average.
    """
    results = []
    for layer_idx in range(NUM_LAYERS):
        cos_sims = []
        l2_dists = []
        per_utterance_cos = []

        for utterance_id in utterance_ids:
            clean_path = os.path.join(
                EMBEDDINGS_DIR, f"{utterance_id}_layer{layer_idx:02d}.npy"
            )
            clean_vec = load_pooled(clean_path)

            utt_cos = []
            for severity in SEVERITIES:
                dist_path = os.path.join(
                    DISTORTED_EMBEDDINGS_DIR,
                    f"{utterance_id}_{severity}_layer{layer_idx:02d}.npy",
                )
                dist_vec = load_pooled(dist_path)

                cos = cosine_similarity(clean_vec, dist_vec)
                l2 = l2_distance(clean_vec, dist_vec)

                cos_sims.append(cos)
                l2_dists.append(l2)
                utt_cos.append(cos)

            per_utterance_cos.append(float(np.mean(utt_cos)))

        results.append(
            {
                "layer": layer_idx,
                "cosine_similarity": float(np.mean(cos_sims)),
                "l2_distance": float(np.mean(l2_dists)),
                "variance": float(np.var(per_utterance_cos)),
            }
        )
    return results


def plot_metrics(results, output_path):
    """Plot cosine similarity, L2 distance, and variance as small multiples.

    Each metric has a different scale (similarity in [-1, 1], unbounded L2
    distance, small variance), so they get separate stacked axes rather than
    a dual-axis overlay.
    """
    layers = [r["layer"] for r in results]
    cosine = [r["cosine_similarity"] for r in results]
    l2 = [r["l2_distance"] for r in results]
    variance = [r["variance"] for r in results]

    best_idx = int(np.argmax(cosine))
    best_layer = layers[best_idx]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, facecolor=SURFACE)

    specs = [
        (axes[0], cosine, COLOR_COSINE, "Cosine similarity (clean vs. distorted)", "higher = better"),
        (axes[1], l2, COLOR_L2, "L2 distance (clean vs. distorted)", "lower = better"),
        (axes[2], variance, COLOR_VARIANCE, "Variance across utterances", "lower = more consistent"),
    ]

    for ax, values, color, title, subtitle in specs:
        ax.set_facecolor(SURFACE)
        ax.plot(layers, values, color=color, linewidth=2, marker="o", markersize=4)
        ax.axvline(best_layer, color=MUTED, linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(f"{title}  ({subtitle})", loc="left", fontsize=11, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED)

    axes[0].scatter(
        [best_layer], [cosine[best_idx]],
        color=COLOR_COSINE, s=60, zorder=5, edgecolor=INK, linewidth=0.8,
    )
    axes[0].annotate(
        f"layer {best_layer}",
        xy=(best_layer, cosine[best_idx]),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=INK,
    )

    axes[-1].set_xlabel("wav2vec2 layer", color=INK)
    axes[-1].set_xticks(layers)

    fig.suptitle(
        "wav2vec2 layer selection: content robustness to dysarthric distortion",
        fontsize=13,
        color=INK,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select the wav2vec2 layer most robust to dysarthric distortion."
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract distorted embeddings even if already cached.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PLOT,
        help=f"Path to save the comparison plot (default: {DEFAULT_OUTPUT_PLOT})",
    )
    args = parser.parse_args()

    utterance_ids = discover_utterance_ids()
    print(f"Found {len(utterance_ids)} utterances with clean embeddings.")

    ensure_distorted_embeddings(utterance_ids, force=args.force_extract)

    results = compute_layer_metrics(utterance_ids)

    print(f"\n{'Layer':>5} {'CosineSim':>10} {'L2Dist':>10} {'Variance':>10}")
    for r in results:
        print(
            f"{r['layer']:>5} {r['cosine_similarity']:>10.4f} "
            f"{r['l2_distance']:>10.4f} {r['variance']:>10.6f}"
        )

    plot_metrics(results, args.output)

    print(
        "\nNote: raw embedding norms grow across transformer depth and layer 24 "
        "is shrunk back down by the model's final LayerNorm, so L2 distance is "
        "not directly comparable across layers -- cosine similarity (scale-"
        "invariant) is the primary signal for the recommendation below."
    )

    best = max(results, key=lambda r: r["cosine_similarity"])
    print(
        f"\nRecommended layer: {best['layer']} "
        f"(cosine similarity = {best['cosine_similarity']:.4f}, "
        f"L2 distance = {best['l2_distance']:.4f}, "
        f"variance across utterances = {best['variance']:.6f})"
    )
    print(
        "This layer's embeddings stay most similar between clean and dysarthric-"
        "distorted speech, meaning it captures phonetic/content information rather "
        "than acoustic noise -- the best candidate layer for content representation."
    )


if __name__ == "__main__":
    main()
