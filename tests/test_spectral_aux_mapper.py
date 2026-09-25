import unittest
import numpy as np
import torch
from src.models.spectral_aux_mapper import SpectralAuxMapper, extract_log_mel
from src.models.mapper import build_mapper_from_config
from src.utils.config import load_yaml_config, DEFAULT_MODEL_CONFIG_PATH
from src.utils.mel import compute_mel


class SpectralAuxTests(unittest.TestCase):
    def test_architecture_alignment_and_gradient(self):
        torch.set_num_threads(2)
        cfg = load_yaml_config(DEFAULT_MODEL_CONFIG_PATH)
        base = build_mapper_from_config(cfg)
        model = SpectralAuxMapper(cfg).eval()
        self.assertEqual(model.mapper.lstm.input_size, 1104)
        for key, value in base.state_dict().items():
            new = model.mapper.state_dict()[key]
            if key in ('lstm.weight_ih_l0', 'lstm.weight_ih_l0_reverse'):
                self.assertEqual(new.shape, (2048, 1104))
            else:
                self.assertEqual(new.shape, value.shape)
        embedding = torch.randn(5, 1024)
        mel = torch.randn(9, 80, requires_grad=True)
        lengths = torch.tensor([7, 11])
        features = [(embedding, mel), (torch.randn(8,1024), torch.randn(11,80))]
        pred = model(features, lengths)
        self.assertEqual(pred.shape, (2,11,80))
        alone = model(features[:1], lengths[:1])
        torch.testing.assert_close(pred[0,:7], alone[0], atol=1e-6, rtol=1e-5)
        pred[0,:7].square().mean().backward()
        self.assertGreater(mel.grad.abs().sum().item(), 0)
        self.assertIsNone(embedding.grad)
        self.assertGreater(model.mapper.lstm.weight_ih_l0.grad[:,1024:].abs().sum().item(),0)
        restored = SpectralAuxMapper(model.model_config).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(restored(features, lengths), pred)
        aligned = model.align(mel.detach(), 9)
        torch.testing.assert_close(aligned, mel.detach())

    def test_source_target_same_mel(self):
        audio = np.random.default_rng(42).normal(0,.1,16000).astype(np.float32)
        actual = extract_log_mel(audio,16000)
        np.testing.assert_array_equal(actual.numpy(),compute_mel(audio,sr=16000).T)


if __name__ == '__main__':
    unittest.main()
