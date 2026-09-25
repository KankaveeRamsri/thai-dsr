"""Postnet contracts; these small tensors test code, not speech-quality metrics."""
import unittest
import torch
from src.models.postnet import Postnet
from src.training.train_postnet import mel_l1


class PostnetContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1234)

    def test_exact_identity_in_train_and_eval(self):
        model = Postnet(hidden_channels=16)
        mel = torch.randn(1, 80, 31)
        snapshot = mel.clone()
        for training in (True, False):
            model.train(training)
            self.assertTrue(torch.equal(model.refine(mel), mel))
            self.assertEqual(model(mel).shape, mel.shape)
        self.assertTrue(torch.equal(mel, snapshot))

    def test_only_postnet_updates(self):
        frozen_mapper = torch.nn.Conv1d(80, 80, 1).eval().requires_grad_(False)
        before = {k: v.clone() for k, v in frozen_mapper.state_dict().items()}
        with torch.no_grad():
            cached = frozen_mapper(torch.randn(1, 80, 37))
        model = Postnet(hidden_channels=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        target = torch.randn(1, 80, 29)
        for _ in range(3):
            optimizer.zero_grad()
            loss = mel_l1(model.refine(cached), target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            optimizer.step()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in frozen_mapper.parameters()))
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in frozen_mapper.state_dict().items()))
        self.assertFalse(torch.equal(model.eval().refine(cached), cached))
        self.assertGreater(float(model.layers[0][0].weight.grad.norm()), 0)

    def test_loss_uses_existing_target_without_modifying_either_input(self):
        mel = torch.randn(1, 80, 41, requires_grad=True)
        target = torch.randn(1, 80, 35)
        before = target.clone()
        loss = mel_l1(mel, target)
        loss.backward()
        self.assertEqual(mel.grad.shape, mel.shape)
        self.assertTrue(torch.isfinite(mel.grad).all())
        self.assertTrue(torch.equal(target, before))

    def test_even_kernel_rejected(self):
        with self.assertRaises(ValueError):
            Postnet(kernel_size=4)


if __name__ == '__main__':
    unittest.main()
