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

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MANIFEST_PATH = os.path.join("data", "manifest.csv")
DISTORTED_EMBEDDINGS_DIR = os.path.join("data", "embeddings", "distorted")

DEFAULT_SEVERITY = "severe"
DEFAULT_LAYER = 9  # wav2vec2 layer selected via scripts/select_layer.py

# Mel-spectrogram extraction settings for the clean target.
MEL_SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
LOG_EPS = 1e-9


class DysarthricDataset(Dataset):
    """(distorted wav2vec2 embedding, clean mel-spectrogram) pairs for training the mapper.

    Each manifest row pairs one distorted utterance with its clean reference.
    __getitem__ loads the cached wav2vec2 embedding for the distorted audio
    (extracted by scripts/select_layer.py into data/embeddings/distorted/),
    computes a log-mel spectrogram from the clean .wav on the fly, and
    linearly interpolates the embedding along time so it lines up frame-for-
    frame with the mel (wav2vec2 runs at ~50Hz, mel frames at ~86Hz).
    """

    def __init__(
        self,
        manifest_path=MANIFEST_PATH,
        embeddings_dir=DISTORTED_EMBEDDINGS_DIR,
        severity=DEFAULT_SEVERITY,
        layer=DEFAULT_LAYER,
    ):
        self.embeddings_dir = embeddings_dir
        self.layer = layer

        manifest = pd.read_csv(manifest_path)
        if severity is not None:
            manifest = manifest[manifest["severity"] == severity]
        self.manifest = manifest.reset_index(drop=True)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        row = self.manifest.iloc[index]

        embedding = self._load_distorted_embedding(row["utterance_id"], row["severity"])
        mel = self._extract_clean_mel(row["clean_path"])

        embedding = interpolate_embedding(embedding, target_length=mel.shape[0])
        mel = torch.from_numpy(mel).float()

        return embedding, mel

    def _load_distorted_embedding(self, utterance_id, severity):
        """Load the cached wav2vec2 hidden state for one distorted utterance.

        Shape: (T_emb, 1024).
        """
        path = os.path.join(
            self.embeddings_dir, f"{utterance_id}_{severity}_layer{self.layer:02d}.npy"
        )
        return np.load(path)

    def _extract_clean_mel(self, clean_path):
        """Compute a log-mel spectrogram from the clean reference .wav.

        Shape: (T_mel, 80).
        """
        audio, sample_rate = sf.read(clean_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != MEL_SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=MEL_SAMPLE_RATE)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=MEL_SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        log_mel = np.log(mel + LOG_EPS)
        return log_mel.T.astype(np.float32)  # (n_mels, T_mel) -> (T_mel, n_mels)


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
    print(f"Dataset size: {len(dataset)} samples (severity='{DEFAULT_SEVERITY}', layer={DEFAULT_LAYER})")

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
