"""Frozen layer-9 features plus distorted log-mel, with the original mapper."""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from src.models.mapper import build_mapper_from_config
from src.utils.mel import compute_mel

# Shared explicitly by source and target extraction. Matches existing experiments.
MEL_CONFIG = dict(n_fft=1024, hop_length=256, win_length=1024,
                  n_mels=80, fmax=8000, power=1)


def extract_log_mel(audio, sample_rate):
    """Mono waveform -> (T, 80), resampling to HiFi-GAN's 22050 Hz."""
    return torch.from_numpy(compute_mel(audio, sr=sample_rate, **MEL_CONFIG).T.copy())


class SpectralAuxMapper(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        config = deepcopy(model_config)
        if config['mapper']['input_dim'] not in (1024, 1024 + MEL_CONFIG['n_mels']) or config['mapper']['mel_dim'] != MEL_CONFIG['n_mels']:
            raise ValueError('Expected original 1024-dimensional encoder and 80-bin mel config')
        config['mapper']['input_dim'] = 1024 + MEL_CONFIG['n_mels']
        self.model_config = config
        self.mapper = build_mapper_from_config(config)

    @staticmethod
    def align(sequence, length):
        return F.interpolate(sequence.float().T[None], size=int(length),
                             mode='linear', align_corners=False)[0].T

    def forward(self, features, lengths):
        """features: unpadded (layer9[T,1024], distorted_mel[S,80]) pairs.

        Retain original training convention: interpolate both streams to target
        mel length. In inference use distorted mel length, without a clean input.
        This is global duration alignment, not phonetic/DTW alignment.
        """
        if len(features) != len(lengths):
            raise ValueError('Feature/length batch mismatch')
        inputs = []
        for (embedding, mel), length in zip(features, lengths):
            if embedding.ndim != 2 or embedding.shape[-1] != 1024 or mel.ndim != 2 or mel.shape[-1] != 80:
                raise ValueError('Expected (T,1024) layer 9 and (S,80) distorted mel')
            inputs.append(torch.cat((self.align(embedding, length),
                                     self.align(mel, length)), dim=-1))
        return self.mapper(nn.utils.rnn.pad_sequence(inputs, batch_first=True), lengths)
