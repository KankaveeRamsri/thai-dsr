# dataset.py
#
# Purpose:
#   PyTorch Dataset/DataLoader definitions that read the manifest produced
#   by build_manifest.py and yield (distorted, clean, speaker_embedding)
#   tuples for training.
#
# Expected responsibilities:
#   - Load manifest (CSV/JSON) and split by train/val/test
#   - Load audio/embeddings referenced in each manifest row
#   - Apply padding/collation for variable-length audio in batches
#   - Read dataset-related settings from configs/data.yaml

import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.utils.config import DEFAULT_TRAIN_CONFIG_PATH, REPO_ROOT, get_selected_layer
from src.utils.mel import compute_mel
from src.utils.time_warp import TempoWarp, mel_frame_count


MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"
DISTORTED_EMBEDDINGS_DIR = REPO_ROOT / "data" / "embeddings" / "distorted"

DEFAULT_SEVERITY = "severe"
ALIGNMENTS = ("linear", "tempo")


class DysarthricDataset(Dataset):
    """(distorted wav2vec2 embedding, clean mel-spectrogram) pairs for training the mapper.

    Each manifest row pairs one distorted utterance with its clean reference.
    __getitem__ loads the cached wav2vec2 embedding for the distorted audio
    (extracted by scripts/select_layer.py into data/embeddings/distorted/),
    computes a log-mel spectrogram from the clean .wav on the fly, and
    linearly interpolates the embedding along time so it lines up frame-for-
    frame with the mel (wav2vec2 runs at ~50Hz, mel frames at ~86Hz).

    ``alignment`` chooses how the distorted input and clean target meet:

    - ``"linear"`` (default, the original behaviour): the embedding is
      stretched over the clean mel's length, so the tempo perturbation is
      undone only on average.
    - ``"tempo"``: the target is the clean mel warped onto the distorted
      timeline with the logged tempo segments (``distortion_params``, a CSV
      with clip_id, severity and tempo_segments). The embedding is then
      interpolated only for the 50 Hz -> mel frame-rate change, so input and
      target are frame-aligned.

    ``mel_cache_dir`` (optional) holds precomputed ``clean/<utterance_id>.npy``
    and ``tempo/<utterance_id>_<severity>.npy`` targets written by
    scripts/cache_cv_mels.py with these same methods, so items are identical.
    ``preload=True`` reads every embedding and target into RAM once.
    """

    def __init__(
        self,
        manifest_path=MANIFEST_PATH,
        embeddings_dir=DISTORTED_EMBEDDINGS_DIR,
        severity=DEFAULT_SEVERITY,
        train_config_path=DEFAULT_TRAIN_CONFIG_PATH,
        layer=None,
        alignment="linear",
        distortion_params=None,
        mel_cache_dir=None,
        preload=False,
    ):
        if alignment not in ALIGNMENTS:
            raise ValueError(f"alignment must be one of {ALIGNMENTS}, got {alignment!r}")
        if alignment == "tempo" and distortion_params is None:
            raise ValueError("alignment='tempo' needs distortion_params")
        self.alignment = alignment
        self.embeddings_dir = os.fspath(embeddings_dir)
        self.layer = (
            get_selected_layer(train_config_path) if layer is None else int(layer)
        )

        manifest = pd.read_csv(manifest_path)
        if severity is not None:
            manifest = manifest[manifest["severity"] == severity]
        self.manifest = manifest.reset_index(drop=True)

        self.tempo_segments = {}
        if alignment == "tempo":
            params = pd.read_csv(distortion_params, usecols=["clip_id", "severity", "tempo_segments"])
            self.tempo_segments = {
                (row.clip_id, row.severity): row.tempo_segments for row in params.itertuples()
            }
            missing = [
                (u, s) for u, s in zip(self.manifest["utterance_id"], self.manifest["severity"])
                if (u, s) not in self.tempo_segments
            ]
            if missing:
                raise ValueError(f"{len(missing)} manifest row(s) have no tempo_segments, e.g. {missing[0]}")

        self.mel_cache_dir = None if mel_cache_dir is None else os.fspath(mel_cache_dir)
        self._preloaded = None
        if preload:
            self._preloaded = [self._load_item(index) for index in range(len(self.manifest))]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        if self._preloaded is not None:
            embedding, mel = self._preloaded[index]
        else:
            embedding, mel = self._load_item(index)

        embedding = interpolate_embedding(embedding, target_length=mel.shape[0])
        mel = torch.from_numpy(mel).float()

        return embedding, mel

    def _load_item(self, index):
        """Raw ``(embedding (T_emb, D), target mel (T, 80))`` for one row."""
        row = self.manifest.iloc[index]
        embedding = self._load_distorted_embedding(row["utterance_id"], row["severity"])
        if self.mel_cache_dir is not None:
            if self.alignment == "tempo":
                name = os.path.join("tempo", f"{row['utterance_id']}_{row['severity']}.npy")
            else:
                name = os.path.join("clean", f"{row['utterance_id']}.npy")
            return embedding, np.load(os.path.join(self.mel_cache_dir, name))
        mel = self._extract_clean_mel(row["clean_path"])
        if self.alignment == "tempo":
            mel = self.aligned_target(row, mel)
        return embedding, mel

    def aligned_target(self, row, clean_mel):
        """Warp a ``(T_clean, 80)`` clean mel onto the distorted mel frame grid."""
        info = sf.info(self._resolve(row["distorted_path"]))
        warp = TempoWarp(self.tempo_segments[(row["utterance_id"], row["severity"])], info.samplerate)
        if warp.distorted_samples != info.frames:
            raise ValueError(
                f"{row['distorted_path']}: {info.frames} samples, tempo segments give "
                f"{warp.distorted_samples}"
            )
        return warp.warp_frames(clean_mel, n_target=mel_frame_count(info.frames, info.samplerate))

    @staticmethod
    def _resolve(path):
        return path if os.path.isabs(path) else REPO_ROOT / path

    def _load_distorted_embedding(self, utterance_id, severity):
        """Load the cached wav2vec2 hidden state for one distorted utterance.

        Shape: (T_emb, 1024).
        """
        path = os.path.join(
            self.embeddings_dir, f"{utterance_id}_{severity}_layer{self.layer:02d}.npy"
        )
        return np.load(path)

    def _extract_clean_mel(self, clean_path):
        """Compute a HiFi-GAN-compatible log-mel from the clean reference.

        Shape: (T_mel, 80).
        """
        if not os.path.isabs(clean_path):
            clean_path = REPO_ROOT / clean_path
        audio, sample_rate = sf.read(clean_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return compute_mel(audio, sr=sample_rate).T


def interpolate_embedding(embedding, target_length):
    """Linearly interpolate a (T_emb, D) embedding sequence to (target_length, D).

    Bridges the wav2vec2 (~50Hz) vs. mel-spectrogram (~86Hz) frame-rate gap so
    every timestep of the mapper's input lines up with the target mel frame.
    """
    tensor = torch.from_numpy(embedding).float().T.unsqueeze(0)  # (1, D, T_emb)
    resized = F.interpolate(tensor, size=target_length, mode="linear", align_corners=False)
    return resized.squeeze(0).T  # (target_length, D)


def collate_fn(batch):
    """Pad a batch of (embedding, mel) pairs to the longest sequence in the batch.

    Returns:
      padded_embeddings: (B, T_max, 1024)
      padded_mels: (B, T_max, 80)
      lengths: (B,) true length of each sample before padding
    """
    embeddings, mels = zip(*batch)

    lengths = torch.tensor([e.shape[0] for e in embeddings], dtype=torch.long)
    max_len = int(lengths.max())
    batch_size = len(batch)
    embed_dim = embeddings[0].shape[1]
    mel_dim = mels[0].shape[1]

    padded_embeddings = torch.zeros(batch_size, max_len, embed_dim)
    padded_mels = torch.zeros(batch_size, max_len, mel_dim)

    for i, (embedding, mel) in enumerate(zip(embeddings, mels)):
        t = embedding.shape[0]
        padded_embeddings[i, :t] = embedding
        padded_mels[i, :t] = mel

    return padded_embeddings, padded_mels, lengths


if __name__ == "__main__":
    dataset = DysarthricDataset()
    print(
        f"Dataset size: {len(dataset)} samples "
        f"(severity='{DEFAULT_SEVERITY}', layer={dataset.layer})"
    )

    embedding, mel = dataset[0]
    print(f"First sample - embedding shape: {tuple(embedding.shape)}, mel shape: {tuple(mel.shape)}")
    assert embedding.shape[0] == mel.shape[0], "Embedding and mel time dimension mismatch!"
    print("OK: embedding and mel share the same T dimension.")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    padded_embeddings, padded_mels, lengths = next(iter(loader))
    print(
        f"Batch - embeddings: {tuple(padded_embeddings.shape)}, "
        f"mels: {tuple(padded_mels.shape)}, lengths: {lengths.tolist()}"
    )
