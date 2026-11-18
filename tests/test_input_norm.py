"""Input normalisation: statistics come only from train clips, travel with the
checkpoint, and are applied identically in training and inference."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from src.inference.run import load_mapper
from src.models.mapper import MapperModel
from src.training.dataset import DysarthricDataset
from src.training.input_norm import compute_train_input_stats

REPO = Path(__file__).resolve().parents[1]
TRAIN_OFFSET, HELD_OUT_OFFSET = 2.0, 1000.0


class RecordingDataset:
    """dataset[i] -> (embedding, None); records which rows were read."""

    def __init__(self, ids, values):
        self.manifest = pd.DataFrame({"utterance_id": ids})
        self.values = values
        self.read = []

    def __getitem__(self, index):
        self.read.append(index)
        return torch.from_numpy(self.values[index]), None


class TrainOnlyStatsTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ids = [f"c{i}" for i in range(9)]
        self.splits = {"train": self.ids[:5], "val": self.ids[5:7], "test": self.ids[7:]}
        self.values = [rng.standard_normal((10 + i, 6)).astype(np.float32)
                       + (TRAIN_OFFSET if i < 5 else HELD_OUT_OFFSET) for i in range(9)]

    def test_only_train_rows_are_read_and_used(self):
        dataset = RecordingDataset(self.ids, self.values)
        mean, std, meta = compute_train_input_stats(dataset, self.splits, layer=3)
        self.assertEqual(sorted(set(dataset.read)), [0, 1, 2, 3, 4])
        train = np.concatenate(self.values[:5]).astype(np.float64)
        np.testing.assert_allclose(mean.numpy(), train.mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(std.numpy(), train.std(axis=0), rtol=1e-5)
        self.assertLess(float(mean.max()), 10.0)  # nowhere near the held-out offset
        self.assertEqual((meta["split"], meta["layer"], meta["n_clips"], meta["n_frames"]),
                         ("train", 3, 5, len(train)))
        self.assertEqual(meta["clip_ids_sha256"],
                         hashlib.sha256("\n".join(sorted(self.splits["train"])).encode()).hexdigest())

    def test_manifest_order_does_not_matter(self):
        order = [8, 0, 6, 3, 1, 7, 4, 2, 5]
        shuffled = RecordingDataset([self.ids[i] for i in order], [self.values[i] for i in order])
        a = compute_train_input_stats(RecordingDataset(self.ids, self.values), self.splits)
        b = compute_train_input_stats(shuffled, self.splits)
        np.testing.assert_allclose(a[0].numpy(), b[0].numpy(), rtol=1e-6)
        np.testing.assert_allclose(a[1].numpy(), b[1].numpy(), rtol=1e-6)
        self.assertEqual(a[2]["clip_ids_sha256"], b[2]["clip_ids_sha256"])

    def test_train_overlapping_held_out_is_rejected(self):
        splits = {**self.splits, "val": self.ids[4:7]}  # c4 in train and val
        with self.assertRaises(ValueError):
            compute_train_input_stats(RecordingDataset(self.ids, self.values), splits)


class MapperInputNormTest(unittest.TestCase):
    def test_forward_normalises_inside_the_model(self):
        torch.manual_seed(0)
        plain = MapperModel(input_dim=6, lstm_hidden_size=8, projection_hidden_dim=8, mel_dim=4)
        normed = MapperModel(input_dim=6, lstm_hidden_size=8, projection_hidden_dim=8, mel_dim=4,
                             input_norm=True)
        normed.load_state_dict({**plain.state_dict(), "input_mean": torch.full((6,), 3.0),
                                "input_std": torch.full((6,), 2.0)})
        plain.eval(), normed.eval()
        x = torch.randn(2, 7, 6) * 2 + 3
        torch.testing.assert_close(normed(x), plain((x - 3) / 2))

    def test_without_input_norm_has_no_buffers(self):
        self.assertNotIn("input_mean", MapperModel(input_dim=6).state_dict())


def _make_data(root):
    """8 clips; val/test embeddings are offset by +1000 so any leak would show."""
    rng = np.random.default_rng(0)
    rows, ids = [], []
    for i in range(8):
        uid = f"utt{i}"
        ids.append(uid)
        sf.write(root / f"{uid}.wav", (0.1 * rng.standard_normal(8000)).astype(np.float32), 16000)
        offset = TRAIN_OFFSET if i < 4 else HELD_OUT_OFFSET
        np.save(root / f"{uid}_severe_layer09.npy", (rng.standard_normal((25, 1024)) + offset).astype(np.float32))
        rows.append({"utterance_id": uid, "speaker_id": f"s{i}", "clean_path": str(root / f"{uid}.wav"),
                     "distorted_path": str(root / f"{uid}.wav"), "severity": "severe",
                     "transcript": "", "duration_sec": 0.5})
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    splits = {"train": ids[:4], "val": ids[4:6], "test": ids[6:]}
    (root / "splits.json").write_text(json.dumps({"splits": splits}))
    return splits


def _train(root, run, extra=()):
    out = root / run
    cmd = [sys.executable, "-m", "src.training.train",
           "--manifest", str(root / "manifest.csv"), "--splits", str(root / "splits.json"),
           "--fixed-splits", "--embeddings-dir", str(root), "--num-epochs", "1",
           "--device", "cpu", "--num-workers", "0",
           "--checkpoint-dir", str(out), "--log-dir", str(out),
           "--best-checkpoint", str(out / "best.pt"), "--loss-curve", str(out / "loss.png"),
           "--save-latest", str(out / "latest.pt"), *extra]
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(result.stdout[-2000:] + result.stderr[-2000:])
    return out


class TrainScriptTest(unittest.TestCase):
    def test_checkpoint_stats_come_from_train_clips_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = _make_data(root)
            out = _train(root, "norm")
            best = torch.load(out / "best.pt", weights_only=True)
            state = best["model_state_dict"]

            dataset = DysarthricDataset(manifest_path=root / "manifest.csv", embeddings_dir=root, layer=9)
            mean, std, meta = compute_train_input_stats(dataset, splits, layer=9)
            torch.testing.assert_close(state["input_mean"], mean)
            torch.testing.assert_close(state["input_std"], std)
            self.assertLess(float(state["input_mean"].abs().max()), 10.0)  # held-out clips sit at +1000
            self.assertEqual(best["input_norm"], meta)
            self.assertEqual(best["input_norm"]["n_clips"], 4)
            self.assertTrue(best["model_config"]["mapper"]["input_norm"])
            self.assertTrue(best["train_config"]["data"]["input_norm"])

            saved = np.load(out / "input_norm.npz")
            np.testing.assert_array_equal(saved["mean"], mean.numpy())
            self.assertEqual(json.loads(str(saved["meta"])), meta)

            # Inference: the same buffers, applied by the same forward().
            model = load_mapper(out / "best.pt", torch.device("cpu"), REPO / "configs/model.yaml",
                                REPO / "configs/train.yaml", layer=9)
            self.assertTrue(model.input_norm)
            torch.testing.assert_close(model.input_mean, mean)
            with self.assertRaises(ValueError):
                load_mapper(out / "best.pt", torch.device("cpu"), REPO / "configs/model.yaml",
                            REPO / "configs/train.yaml", layer=3)

    def test_no_input_norm_reproduces_w2_and_old_checkpoints_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _make_data(root)
            out = _train(root, "plain", extra=("--no-input-norm",))
            best = torch.load(out / "best.pt", weights_only=True)
            self.assertNotIn("input_mean", best["model_state_dict"])
            self.assertFalse((out / "input_norm.npz").exists())
            model = load_mapper(out / "best.pt", torch.device("cpu"), REPO / "configs/model.yaml",
                                REPO / "configs/train.yaml", layer=9)
            self.assertFalse(model.input_norm)


if __name__ == "__main__":
    unittest.main()
