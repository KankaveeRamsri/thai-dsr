"""Reproducible utterance-level train/validation/test splitting."""

import json
import os
import random
from pathlib import Path


SPLIT_NAMES = ("train", "val", "test")


def _validate_ratios(ratios):
    values = [float(ratios[name]) for name in SPLIT_NAMES]
    if any(value <= 0 for value in values):
        raise ValueError(f"All split ratios must be positive, got {ratios}")
    if abs(sum(values) - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(values):.6f}")
    return dict(zip(SPLIT_NAMES, values))


def _split_counts(num_utterances, ratios):
    if num_utterances < len(SPLIT_NAMES):
        raise ValueError(
            f"At least {len(SPLIT_NAMES)} utterances are required for train/val/test, "
            f"got {num_utterances}"
        )

    raw_counts = [num_utterances * ratios[name] for name in SPLIT_NAMES]
    counts = [int(value) for value in raw_counts]
    remainder_order = sorted(
        range(len(SPLIT_NAMES)),
        key=lambda i: raw_counts[i] - counts[i],
        reverse=True,
    )
    for index in remainder_order[: num_utterances - sum(counts)]:
        counts[index] += 1

    for empty_index, count in enumerate(counts):
        if count == 0:
            donor = max(range(len(counts)), key=counts.__getitem__)
            counts[donor] -= 1
            counts[empty_index] += 1
    return dict(zip(SPLIT_NAMES, counts))


def create_utterance_splits(utterance_ids, ratios, seed):
    """Create deterministic, mutually exclusive split assignments."""
    ratios = _validate_ratios(ratios)
    unique_ids = sorted(set(utterance_ids))
    counts = _split_counts(len(unique_ids), ratios)
    random.Random(seed).shuffle(unique_ids)

    assignments = {}
    start = 0
    for name in SPLIT_NAMES:
        end = start + counts[name]
        assignments[name] = sorted(unique_ids[start:end])
        start = end
    return assignments


def _is_compatible(payload, utterance_ids, ratios, seed):
    if payload.get("seed") != seed or payload.get("ratios") != ratios:
        return False
    splits = payload.get("splits", {})
    if set(splits) != set(SPLIT_NAMES):
        return False
    assigned = [item for name in SPLIT_NAMES for item in splits[name]]
    return len(assigned) == len(set(assigned)) and set(assigned) == set(utterance_ids)


def load_or_create_splits(utterance_ids, ratios, seed, output_path):
    """Reuse a compatible split file, otherwise deterministically replace it."""
    ratios = _validate_ratios(ratios)
    output_path = Path(output_path)
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if _is_compatible(payload, utterance_ids, ratios, seed):
            return payload["splits"], False

    splits = create_utterance_splits(utterance_ids, ratios, seed)
    payload = {"seed": seed, "ratios": ratios, "splits": splits}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_path, output_path)
    return splits, True


def row_indices_for_split(manifest, utterance_ids):
    """Return manifest row positions belonging to the supplied utterance IDs."""
    wanted = set(utterance_ids)
    return [
        index
        for index, utterance_id in enumerate(manifest["utterance_id"])
        if utterance_id in wanted
    ]
