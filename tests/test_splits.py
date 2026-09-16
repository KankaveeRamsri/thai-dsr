import unittest

import pandas as pd

from src.training.splits import (
    SPLIT_NAMES,
    create_utterance_splits,
    row_indices_for_split,
)


class UtteranceSplitTest(unittest.TestCase):
    def test_split_is_deterministic_and_utterance_disjoint(self):
        utterance_ids = [f"utt-{index:03d}" for index in range(10)]
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        first = create_utterance_splits(utterance_ids, ratios, seed=42)
        second = create_utterance_splits(reversed(utterance_ids), ratios, seed=42)

        self.assertEqual(first, second)
        self.assertEqual({name: len(first[name]) for name in SPLIT_NAMES}, {
            "train": 7,
            "val": 2,
            "test": 1,
        })
        sets = {name: set(first[name]) for name in SPLIT_NAMES}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])

        manifest = pd.DataFrame({
            "utterance_id": [item for item in utterance_ids for _ in range(3)],
            "severity": ["mild", "moderate", "severe"] * len(utterance_ids),
        })
        row_ids = {
            name: set(manifest.iloc[row_indices_for_split(manifest, first[name])]["utterance_id"])
            for name in SPLIT_NAMES
        }
        self.assertEqual(row_ids, sets)


if __name__ == "__main__":
    unittest.main()
