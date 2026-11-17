"""Resuming from --save-latest must reproduce an uninterrupted run exactly."""

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.training.train import capture_rng_state, restore_rng_state

REPO = Path(__file__).resolve().parents[1]


def _make_data(root):
    rng = np.random.default_rng(0)
    rows, ids = [], []
    for i in range(8):
        uid = f"utt{i}"
        ids.append(uid)
        sf.write(root / f"{uid}.wav", (0.1 * rng.standard_normal(8000)).astype(np.float32), 16000)
        np.save(root / f"{uid}_severe_layer09.npy", rng.standard_normal((25, 1024)).astype(np.float32))
        rows.append({"utterance_id": uid, "speaker_id": f"s{i}", "clean_path": str(root / f"{uid}.wav"),
                     "distorted_path": str(root / f"{uid}.wav"), "severity": "severe",
                     "transcript": "", "duration_sec": 0.5})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    splits = {"splits": {"train": ids[:4], "val": ids[4:6], "test": ids[6:]}}
    (root / "splits.json").write_text(json.dumps(splits))


def _train(root, run, epochs, extra=()):
    out = root / run
    cmd = [sys.executable, "-m", "src.training.train",
           "--manifest", str(root / "manifest.csv"), "--splits", str(root / "splits.json"),
           "--fixed-splits", "--embeddings-dir", str(root), "--num-epochs", str(epochs),
           "--device", "cpu", "--num-workers", "0",
           "--checkpoint-dir", str(out), "--log-dir", str(out),
           "--best-checkpoint", str(out / "best.pt"), "--loss-curve", str(out / "loss.png"),
           "--save-latest", str(out / "latest.pt"), *extra]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(result.stdout[-2000:] + result.stderr[-2000:])
    return result.stdout


class RngStateTest(unittest.TestCase):
    def test_round_trip(self):
        generator = torch.Generator().manual_seed(1)
        state = capture_rng_state(generator)
        expected = (random.random(), np.random.rand(), torch.rand(1).item(),
                    torch.rand(1, generator=generator).item())
        restore_rng_state(state, generator)
        got = (random.random(), np.random.rand(), torch.rand(1).item(),
               torch.rand(1, generator=generator).item())
        self.assertEqual(got, expected)

    def test_state_survives_weights_only_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            torch.save({"rng_state": capture_rng_state(torch.Generator())}, path)
            torch.load(path, weights_only=True)


class ResumeMatchesUninterruptedTest(unittest.TestCase):
    def test_resume_after_two_of_three_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_data(root)
            _train(root, "full", 3)
            _train(root, "split", 2)
            log = _train(root, "split", 3, extra=("--auto-resume",))
            self.assertIn("Resumed from", log)

            full = torch.load(root / "full" / "latest.pt", weights_only=True)
            resumed = torch.load(root / "split" / "latest.pt", weights_only=True)
            self.assertEqual(resumed["epoch"], 3)
            self.assertEqual(resumed["train_losses"], full["train_losses"])
            self.assertEqual(resumed["val_losses"], full["val_losses"])
            self.assertEqual(resumed["best_val_loss"], full["best_val_loss"])
            self.assertEqual(resumed["best_epoch"], full["best_epoch"])
            for name, tensor in full["model_state_dict"].items():
                self.assertTrue(torch.equal(tensor, resumed["model_state_dict"][name]), name)
            self.assertEqual(resumed["scheduler_state_dict"], full["scheduler_state_dict"])

    def test_auto_resume_without_checkpoint_starts_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_data(root)
            log = _train(root, "fresh", 1, extra=("--auto-resume",))
            self.assertNotIn("Resumed from", log)
            self.assertEqual(torch.load(root / "fresh" / "latest.pt", weights_only=True)["epoch"], 1)


if __name__ == "__main__":
    unittest.main()
