import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.training.hifigan_dataset import HiFiGANCleanDataset


class HiFiGANDatasetTest(unittest.TestCase):
    def test_split_selection_and_frame_aligned_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mel_path = root / "mel.npy"
            wav_path = root / "wav.npy"
            np.save(mel_path, np.ones((80, 40), dtype=np.float32))
            np.save(wav_path, np.arange(40 * 256 + 17, dtype=np.float32))
            index_path = root / "index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "hop_size": 256,
                        "num_mels": 80,
                        "items": [
                            {
                                "utterance_id": "train-item",
                                "split": "train",
                                "mel_path": str(mel_path),
                                "wav_path": str(wav_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            dataset = HiFiGANCleanDataset(
                index_path, "train", segment_size=8192, random_crop=True, seed=7
            )
            mel, audio, utterance_id = dataset[0]

            self.assertEqual(utterance_id, "train-item")
            self.assertEqual(tuple(mel.shape), (80, 32))
            self.assertEqual(tuple(audio.shape), (8192,))
            self.assertEqual(float(audio[0]) % 256, 0.0)
