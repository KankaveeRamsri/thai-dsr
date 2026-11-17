"""Leakage checks for the speaker-independent Common Voice split manifest."""

import csv
import unittest
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "data" / "manifests" / "cv27_th_speaker_split.csv"
COLUMNS = ["clip_id", "client_id", "split", "sentence", "duration", "gender", "age"]
SPLITS = {"train", "val", "test"}


def _splits_by(rows, key):
    found = defaultdict(set)
    for row in rows:
        found[row[key]].add(row["split"])
    return {value: splits for value, splits in found.items() if len(splits) > 1}


class SpeakerSplitManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with MANIFEST.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            cls.columns = reader.fieldnames
            cls.rows = list(reader)

    def test_columns_and_split_names(self):
        self.assertEqual(self.columns, COLUMNS)
        self.assertTrue(self.rows)
        self.assertEqual({row["split"] for row in self.rows}, SPLITS)

    def test_no_client_id_in_more_than_one_split(self):
        leaked = _splits_by(self.rows, "client_id")
        self.assertEqual(leaked, {}, f"{len(leaked)} client_id(s) appear in several splits")

    def test_no_clip_in_more_than_one_split(self):
        leaked = _splits_by(self.rows, "clip_id")
        self.assertEqual(leaked, {}, f"{len(leaked)} clip(s) appear in several splits")

    def test_no_sentence_in_more_than_one_split(self):
        leaked = _splits_by(self.rows, "sentence")
        self.assertEqual(leaked, {}, f"{len(leaked)} sentence(s) appear in several splits")

    def test_clips_per_speaker_between_4_and_8(self):
        counts = Counter(row["client_id"] for row in self.rows)
        outside = {s: n for s, n in counts.items() if not 4 <= n <= 8}
        self.assertEqual(outside, {})

    def test_clip_ids_are_unique(self):
        clip_ids = [row["clip_id"] for row in self.rows]
        self.assertEqual(len(clip_ids), len(set(clip_ids)))

    def test_leak_check_detects_shared_speaker(self):
        rows = [
            {"clip_id": "a", "client_id": "spk", "split": "train"},
            {"clip_id": "b", "client_id": "spk", "split": "test"},
        ]
        self.assertEqual(_splits_by(rows, "client_id"), {"spk": {"train", "test"}})

    def test_leak_check_detects_shared_sentence(self):
        rows = [
            {"clip_id": "a", "client_id": "s1", "split": "train", "sentence": "x"},
            {"clip_id": "b", "client_id": "s2", "split": "val", "sentence": "x"},
        ]
        self.assertEqual(_splits_by(rows, "sentence"), {"x": {"train", "val"}})


if __name__ == "__main__":
    unittest.main()
