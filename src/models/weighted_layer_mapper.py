"""Trainable convex combination of frozen XLSR hidden states + unchanged mapper."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.mapper import build_mapper_from_config


class WeightedLayerMapper(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(25))
        self.mapper = build_mapper_from_config(model_config)

    @property
    def layer_weights(self):
        return self.layer_logits.softmax(dim=0)

    def mix(self, layers):
        if layers.ndim != 3 or layers.shape[0] != 25 or layers.shape[-1] != 1024:
            raise ValueError('Expected (25, T, 1024) hidden states')
        # Cache is FP16; summation and its learnable gradient are FP32.
        return torch.einsum('l,ltd->td', self.layer_weights, layers.float())

    def forward(self, layers, lengths):
        # Mix at native encoder rate before interpolation (linear operations commute).
        embeddings = [F.interpolate(self.mix(h).T[None], size=int(n), mode='linear',
                                    align_corners=False)[0].T for h, n in zip(layers, lengths)]
        batch = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)
        return self.mapper(batch, lengths)
