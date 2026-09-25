"""Contract tests for joint-training data boundaries and differentiable content."""
import json
from types import SimpleNamespace

import tempfile
from pathlib import Path
import unittest
import torch

from src.training.joint_finetune import FrozenContent, load_pairs, loss_mel




class TinyFrozenEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 4, 7, stride=3)
        self.requires_grad_(False)

    def forward(self, x, output_hidden_states):
        hidden = self.conv(x[:, None]).transpose(1, 2)
        return SimpleNamespace(hidden_states=(hidden,) * 10)


def check_content_resampling_normalization_preserves_waveform_gradient():
    # Exercise the production adapter, replacing only the large pretrained network.
    content = FrozenContent.__new__(FrozenContent)
    content.device = torch.device('cpu')
    content.model = TinyFrozenEncoder().eval()
    wave = torch.randn(1, 8192, requires_grad=True)
    content.encode_wave(wave).abs().mean().backward()
    assert wave.grad is not None and torch.isfinite(wave.grad).all()
    assert wave.grad.norm() > 0
    assert all(p.grad is None for p in content.model.parameters())


def check_waveform_mel_loss_backpropagates():
    wave = torch.randn(1, 1, 8192, requires_grad=True)
    result = loss_mel(wave)
    assert result.shape == (1, 80, 32)
    result.abs().mean().backward()
    assert torch.isfinite(wave.grad).all() and wave.grad.norm() > 0


class JointContracts(unittest.TestCase):
    def test_split_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            splits = root / 'splits.json'
            splits.write_text(json.dumps({'splits': {'train': ['x'], 'val': ['x'], 'test': []}}))
            with self.assertRaisesRegex(ValueError, 'Split leakage'):
                load_pairs(root / 'missing.csv', splits)

    def test_w5_pairs(self):
        pairs = load_pairs('data/manifest_w5.csv', 'data/splits_w5.json')
        self.assertEqual({k: len(v) for k, v in pairs.items()}, {'train': 147, 'val': 32, 'test': 31})

    def test_content_gradient(self):
        check_content_resampling_normalization_preserves_waveform_gradient()

    def test_mel_gradient(self):
        check_waveform_mel_loss_backpropagates()


if __name__ == '__main__':
    unittest.main()
