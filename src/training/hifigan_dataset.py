"""Prepared clean Thai mel/waveform pairs for HiFi-GAN fine-tuning."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.utils.mel import HIFIGAN_SAMPLE_RATE, compute_mel


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest_w5.csv"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits_w5.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "hifigan_finetune"


def _project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def prepare_hifigan_dataset(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    splits_path: str | Path = DEFAULT_SPLITS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    sample_rate: int = HIFIGAN_SAMPLE_RATE,
    n_fft: int = 1024,
    hop_size: int = 256,
    win_size: int = 1024,
    num_mels: int = 80,
    fmax: int = 8000,
) -> dict:
    """Materialize one deduplicated clean pair per utterance and write an index."""
    manifest_path = _project_path(manifest_path)
    splits_path = _project_path(splits_path)
    output_dir = _project_path(output_dir)
    wav_dir = output_dir / "wavs"
    mel_dir = output_dir / "mels"
    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {}
    for row in rows:
        utterance_id = str(row["utterance_id"])
        clean_path = str(row["clean_path"])
        if utterance_id in by_id and by_id[utterance_id] != clean_path:
            raise ValueError(f"Conflicting clean paths for {utterance_id!r}")
        by_id[utterance_id] = clean_path

    with splits_path.open(encoding="utf-8") as handle:
        split_payload = json.load(handle)
    splits = split_payload["splits"]
    split_for_id = {
        str(utterance_id): split
        for split, utterance_ids in splits.items()
        for utterance_id in utterance_ids
    }
    if set(split_for_id) != set(by_id):
        missing = sorted(set(by_id) - set(split_for_id))
        extra = sorted(set(split_for_id) - set(by_id))
        raise ValueError(
            "Manifest/splits utterance mismatch: "
            f"missing_from_splits={len(missing)}, extra_in_splits={len(extra)}"
        )

    items = []
    for item_number, utterance_id in enumerate(sorted(by_id)):
        source_path = _project_path(by_id[utterance_id])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing clean waveform: {source_path}")
        audio, source_sr = sf.read(source_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError(f"Expected non-empty mono audio at {source_path}")
        if source_sr != sample_rate:
            audio = librosa.resample(
                audio, orig_sr=source_sr, target_sr=sample_rate
            ).astype(np.float32)
        else:
            audio = np.asarray(audio, dtype=np.float32)

        # This is the single canonical extraction path used by mapper targets too.
        mel = compute_mel(
            audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_size,
            win_length=win_size,
            n_mels=num_mels,
            fmax=fmax,
        )
        expected_samples = mel.shape[1] * hop_size
        if audio.size < expected_samples:
            raise ValueError(
                f"Mel/audio alignment failed for {utterance_id}: "
                f"{mel.shape[1]} frames require {expected_samples} samples, got {audio.size}"
            )

        cache_name = f"{item_number:04d}.npy"
        wav_path = wav_dir / cache_name
        mel_path = mel_dir / cache_name
        np.save(wav_path, audio, allow_pickle=False)
        np.save(mel_path, mel, allow_pickle=False)
        items.append(
            {
                "utterance_id": utterance_id,
                "split": split_for_id[utterance_id],
                "source_path": str(source_path.relative_to(REPO_ROOT)),
                "wav_path": str(wav_path.relative_to(REPO_ROOT)),
                "mel_path": str(mel_path.relative_to(REPO_ROOT)),
                "num_samples": int(audio.size),
                "num_frames": int(mel.shape[1]),
            }
        )

    payload = {
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
        "splits_path": str(splits_path.relative_to(REPO_ROOT)),
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_size": hop_size,
        "win_size": win_size,
        "num_mels": num_mels,
        "fmax": fmax,
        "split_counts": {
            split: sum(item["split"] == split for item in items)
            for split in ("train", "val", "test")
        },
        "items": items,
    }
    index_path = output_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return payload


class HiFiGANCleanDataset(Dataset):
    """Aligned cached clean mel/audio pairs, cropped on mel-frame boundaries."""

    def __init__(
        self,
        index_path: str | Path,
        split: str,
        segment_size: int,
        *,
        random_crop: bool,
        seed: int = 1234,
    ):
        index_path = _project_path(index_path)
        with index_path.open(encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        self.items = [item for item in self.metadata["items"] if item["split"] == split]
        if not self.items:
            raise ValueError(f"No items for split {split!r} in {index_path}")
        self.hop_size = int(self.metadata["hop_size"])
        self.num_mels = int(self.metadata["num_mels"])
        self.segment_size = int(segment_size)
        if self.segment_size % self.hop_size:
            raise ValueError("segment_size must be divisible by hop_size")
        self.frames_per_segment = self.segment_size // self.hop_size
        self.random_crop = random_crop
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        mel = torch.from_numpy(np.load(_project_path(item["mel_path"]))).float()
        audio = torch.from_numpy(np.load(_project_path(item["wav_path"]))).float()
        if mel.ndim != 2 or mel.shape[0] != self.num_mels or audio.ndim != 1:
            raise ValueError(
                f"Bad cached shapes for {item['utterance_id']}: "
                f"mel={tuple(mel.shape)}, audio={tuple(audio.shape)}"
            )

        if self.random_crop:
            if mel.shape[1] >= self.frames_per_segment:
                max_start = mel.shape[1] - self.frames_per_segment
                frame_start = self.rng.randint(0, max_start)
                mel = mel[:, frame_start : frame_start + self.frames_per_segment]
                sample_start = frame_start * self.hop_size
                audio = audio[sample_start : sample_start + self.segment_size]
            else:
                mel = F.pad(mel, (0, self.frames_per_segment - mel.shape[1]))
                audio = F.pad(audio, (0, self.segment_size - audio.shape[0]))
        else:
            # Generator emits exactly hop_size samples per input frame.
            audio = audio[: mel.shape[1] * self.hop_size]

        return mel, audio, item["utterance_id"]
