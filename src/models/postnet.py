"""Tacotron2-style residual mel correction; forward returns the correction only."""
import torch
from torch import nn


class Postnet(nn.Module):
    def __init__(self, mel_channels=80, hidden_channels=512, layers=5, kernel_size=5):
        super().__init__()
        if layers < 2 or kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError('Require >=2 layers and a positive odd kernel size')
        self.config = dict(mel_channels=mel_channels, hidden_channels=hidden_channels,
                           layers=layers, kernel_size=kernel_size)
        blocks = []
        for index in range(layers):
            cin = mel_channels if index == 0 else hidden_channels
            cout = mel_channels if index == layers-1 else hidden_channels
            conv = nn.Conv1d(cin, cout, kernel_size, padding=kernel_size//2, bias=False)
            nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain('tanh') if index < layers-1 else 1.)
            norm = nn.BatchNorm1d(cout)
            block = [conv, norm]
            if index < layers-1:
                block.append(nn.Tanh())
            blocks.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*blocks)
        # Start with an exact zero residual, in both train/eval modes. The final BN
        # affine parameters learn immediately; earlier convolutions follow as scale opens.
        nn.init.zeros_(self.layers[-1][1].weight)
        nn.init.zeros_(self.layers[-1][1].bias)

    def forward(self, predicted_mel):
        if predicted_mel.ndim != 3 or predicted_mel.shape[1] != self.config['mel_channels']:
            raise ValueError('Expected (batch, 80 mel channels, frames)')
        return self.layers(predicted_mel)

    def refine(self, predicted_mel):
        return predicted_mel + self(predicted_mel)
