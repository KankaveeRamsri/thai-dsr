"""Per-dimension statistics of the mapper input, from the TRAIN split only.

The mapper normalises its input as (x - mean) / std (MapperModel input_norm).
mean and std are computed here over every frame the mapper sees for the train
clips: the cached embedding interpolated to the target mel frame count, i.e.
``dataset[i][0]`` exactly as the training loader yields it. Validation and test
rows are never read. See docs/ablation_plan.md, Amendment 1.
"""

import hashlib

import numpy as np
import torch

from src.training.splits import row_indices_for_split

STD_FLOOR = 1e-5


def train_clip_indices(manifest, splits):
    """Manifest row positions of the train split; fails if train overlaps val/test."""
    train = set(splits["train"])
    held_out = set(splits.get("val", [])) | set(splits.get("test", []))
    overlap = train & held_out
    if overlap:
        raise ValueError(f"{len(overlap)} clip(s) are in train and in val/test, e.g. {sorted(overlap)[0]}")
    indices = row_indices_for_split(manifest, splits["train"])
    used = set(manifest["utterance_id"].iloc[indices])
    if not used <= train or used & held_out:
        raise AssertionError("train rows include non-train clips")
    if not indices:
        raise ValueError("the train split has no rows in this manifest")
    return indices


def compute_train_input_stats(dataset, splits, layer=None, std_floor=STD_FLOOR):
    """(mean, std, meta) of the mapper input over the train split only.

    ``dataset`` must have a ``manifest`` DataFrame with ``utterance_id`` and
    return ``(embedding (T, D) tensor, target)`` from ``dataset[i]``. Per-clip
    moments are merged with Chan's parallel formula in float64.
    """
    indices = train_clip_indices(dataset.manifest, splits)
    count, mean, m2 = 0, None, None
    for index in indices:
        x = dataset[index][0]
        x = (x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)).astype(np.float64)
        n_b = x.shape[0]
        mean_b = x.mean(axis=0)
        m2_b = ((x - mean_b) ** 2).sum(axis=0)
        if mean is None:
            count, mean, m2 = n_b, mean_b, m2_b
            continue
        delta = mean_b - mean
        total = count + n_b
        mean = mean + delta * (n_b / total)
        m2 = m2 + m2_b + delta ** 2 * (count * n_b / total)
        count = total
    std = np.maximum(np.sqrt(m2 / count), std_floor)
    clip_ids = sorted(set(dataset.manifest["utterance_id"].iloc[indices]))
    meta = {
        "split": "train",
        "layer": None if layer is None else int(layer),
        "n_clips": len(indices),
        "n_frames": int(count),
        "clip_ids_sha256": hashlib.sha256("\n".join(clip_ids).encode()).hexdigest(),
        "std_floor": float(std_floor),
        "n_dims_at_floor": int((std == std_floor).sum()),
    }
    return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(std.astype(np.float32)), meta
