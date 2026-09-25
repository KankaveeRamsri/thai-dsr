"""Contract tests for joint-training data boundaries and differentiable content."""
import json
from types import SimpleNamespace

import tempfile
from pathlib import Path
import unittest
import torch

from src.training.joint_finetune import (
    FrozenContent, load_pairs, loss_mel, gan_scale,
    clip_optimizer_gradients, weighted_loss_gradients, parse_args,
)




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
    def test_gan_schedule_boundaries_and_resume(self):
        self.assertEqual([gan_scale(s, 2, 3) for s in (0, 1, 2)], [0, 0, 0])
        self.assertAlmostEqual(gan_scale(3, 2, 3), 1/3)
        self.assertAlmostEqual(gan_scale(4, 2, 3), 2/3)
        self.assertEqual(gan_scale(5, 2, 3), 1)
        self.assertEqual(gan_scale(100, 2, 3), 1)
        # A resumed run calls the same function with global steps, not steps since restart.
        full = [gan_scale(s, 2, 3) for s in range(1, 9)]
        resumed = [gan_scale(s, 2, 3) for s in range(4, 9)]
        self.assertEqual(full[3:], resumed)

    def test_optional_zero_length_schedule(self):
        self.assertEqual(gan_scale(1, 0, 0), 1)
        self.assertEqual(gan_scale(2, 2, 0), 0)
        self.assertEqual(gan_scale(3, 2, 0), 1)
        self.assertEqual(gan_scale(1, 0, 4), 0.25)
        with self.assertRaises(ValueError):
            gan_scale(1, -1, 10)

    def test_clip_combined_optimizer_before_update(self):
        modules = [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
        optimizer = torch.optim.SGD([p for m in modules for p in m.parameters()], lr=1)
        for module, norm in zip(modules, (3., 4.)):
            module.weight.data.zero_()
            module.weight.grad = torch.tensor([[norm]])
        before, after = clip_optimizer_gradients(modules, 1.)
        self.assertAlmostEqual(before, 5.)
        self.assertLessEqual(after, 1.000001)
        self.assertAlmostEqual(after, 1., places=5)
        optimizer.step()
        self.assertAlmostEqual(float(modules[0].weight.detach()), -0.6, places=5)
        self.assertAlmostEqual(float(modules[1].weight.detach()), -0.8, places=5)

    def test_clip_rejects_nonfinite_and_invalid_limit(self):
        module = torch.nn.Linear(1, 1, bias=False)
        for value in (float('nan'), float('inf')):
            module.weight.grad = torch.tensor([[value]])
            with self.assertRaises(RuntimeError):
                clip_optimizer_gradients([module], 1.)
        for limit in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                clip_optimizer_gradients([module], limit)

    def test_loss_gradient_diagnostics_do_not_modify_backward(self):
        predicted = torch.tensor([2.], requires_grad=True)
        terms = dict(mel=3*predicted.square().sum(), content=2*predicted.sum(),
                     adversarial=predicted.new_zeros(()), feature_matching=predicted.new_zeros(()))
        measured = weighted_loss_gradients(terms, predicted)
        self.assertEqual(measured, dict(mel=12., content=2., adversarial=0., feature_matching=0.))
        self.assertIsNone(predicted.grad)
        sum(terms.values()).backward()
        self.assertEqual(float(predicted.grad), 14.)

    def test_stability_cli_defaults(self):
        args = parse_args([])
        self.assertEqual((args.grad_clip, args.gan_warmup_steps, args.gan_ramp_steps), (1., 1000, 1000))
        self.assertEqual((args.validation_interval, args.val_items), (1000, 3))
        smoke = parse_args(['--gan-warmup-steps', '200', '--gan-ramp-steps', '200', '--max-steps', '600'])
        self.assertEqual(gan_scale(smoke.max_steps, smoke.gan_warmup_steps, smoke.gan_ramp_steps), 1.)

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
