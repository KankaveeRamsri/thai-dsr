import unittest
import torch
from torch.nn import functional as F
from src.models.weighted_layer_mapper import WeightedLayerMapper

CFG={'mapper':dict(architecture='bilstm',input_dim=1024,lstm_hidden_size=8,lstm_num_layers=2,dropout=0.,projection_hidden_dim=8,mel_dim=80)}

class WeightedLayerTests(unittest.TestCase):
    def test_mix_gradient_and_mapper_gradient(self):
        torch.manual_seed(1)
        model=WeightedLayerMapper(CFG)
        hs=torch.randn(25,7,1024).half()
        torch.testing.assert_close(model.mix(hs),hs.float().mean(0))
        pred=model([hs,hs[:,:5]],torch.tensor([9,6]))
        self.assertEqual(tuple(pred.shape),(2,9,80))
        pred.square().mean().backward()
        self.assertGreater(float(model.layer_logits.grad.norm()),0)
        self.assertGreater(float(model.mapper.lstm.weight_ih_l0.grad.norm()),0)
        self.assertIsNone(hs.grad)
        before=model.layer_weights.detach().clone()
        torch.optim.Adam(model.parameters(),lr=.001).step()
        self.assertFalse(torch.equal(before,model.layer_weights))
        self.assertAlmostEqual(float(model.layer_weights.sum()),1.,places=6)

    def test_interpolation_commutes_and_padding_isolated(self):
        torch.manual_seed(2)
        m=WeightedLayerMapper(CFG).eval()
        with torch.no_grad():m.layer_logits.copy_(torch.randn(25))
        h=torch.randn(25,7,1024)
        a=F.interpolate(m.mix(h).T[None],size=11,mode='linear',align_corners=False)[0].T
        b=m.mix(F.interpolate(h.transpose(1,2),size=11,mode='linear',align_corners=False).transpose(1,2))
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
        short=m([h],torch.tensor([11]))[0]
        batched=m([h,torch.randn(25,15,1024)],torch.tensor([11,20]))[0,:11]
        torch.testing.assert_close(short,batched,atol=1e-6,rtol=1e-5)

if __name__=='__main__':unittest.main()
