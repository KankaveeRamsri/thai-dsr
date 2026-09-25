import unittest
from unittest.mock import patch
import torch
from src.models.mel_discriminator import (MelDiscriminator, adversarial_weight,
    score_mels, discriminator_loss, generator_loss)
from src.training.train_mapper_mel_adversarial import clip


class MelDiscriminatorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_schedule(self):
        for step, expected in ((0,0.),(50,0.),(51,.001),(100,.05),(150,.1),(300,.1)):
            self.assertAlmostEqual(adversarial_weight(step),expected)

    def test_clip_rescales_large_gradients(self):
        p=torch.nn.Parameter(torch.zeros(2))
        p.grad=torch.tensor([300.,400.])
        raw,post=clip([p])
        self.assertAlmostEqual(raw,500.)
        self.assertLessEqual(post,1.00001)
        torch.testing.assert_close(p.grad,torch.tensor([.6,.8]))

    def test_clip_tolerates_reduction_roundoff(self):
        p=torch.nn.Parameter(torch.zeros(2))
        p.grad=torch.tensor([3.,4.])
        # Simulate disagreement in the diagnostic reduction after real clipping.
        with patch('torch.linalg.vector_norm',side_effect=[torch.tensor(5.),torch.tensor(1.),torch.tensor(1.00002)]):
            raw,post=clip([p])
        self.assertEqual(raw,5.)
        self.assertGreater(post,1.00001)
        torch.testing.assert_close(p.grad,torch.tensor([.6,.8]))

    def test_clip_rejects_nonfinite_gradients(self):
        p=torch.nn.Parameter(torch.zeros(1))
        p.grad=torch.tensor([float('inf')])
        with self.assertRaises(RuntimeError):clip([p])

    def test_padding_and_gradient_isolation(self):
        d=MelDiscriminator()
        self.assertLess(sum(p.numel() for p in d.parameters()),30000)
        real=torch.randn(2,48,80)
        pred=torch.randn(2,48,80,requires_grad=True)
        lengths=torch.tensor([32,48])
        scores=score_mels(d,pred,lengths)
        changed=pred.detach().clone();changed[0,32:]=1000
        torch.testing.assert_close(scores[0],score_mels(d,changed,lengths)[0])
        loss=discriminator_loss(score_mels(d,real,lengths),score_mels(d,pred.detach(),lengths))
        loss.backward()
        self.assertIsNone(pred.grad)
        raw,post=clip(d.parameters());self.assertLessEqual(post,1.00001)
        d.zero_grad(set_to_none=True);d.requires_grad_(False)
        generator_loss(score_mels(d,pred,lengths)).backward()
        self.assertGreater(pred.grad.abs().sum().item(),0)
        self.assertEqual(pred.grad[0,32:].abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for p in d.parameters()))

    def test_zero_weight_no_adversarial_gradient(self):
        d=MelDiscriminator().requires_grad_(False)
        pred=torch.randn(1,32,80,requires_grad=True)
        (adversarial_weight(50)*generator_loss(score_mels(d,pred,[32]))).backward()
        self.assertEqual(pred.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
