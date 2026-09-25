"""Small mel-only patch discriminator; no waveform or vocoder dependencies."""
import torch
from torch import nn


class MelDiscriminator(nn.Module):
    """Score local time-frequency patches of (B, 1, 80, T) log-mels.

    No batch statistics or sigmoid. A fixed affine input scale is shared by
    real and predicted mels. Trim padding before calling this module.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, mel):
        if mel.ndim != 4 or mel.shape[1] != 1 or mel.shape[2] != 80:
            raise ValueError('Expected (B, 1, 80, T) log-mel')
        return self.net((mel + 5.0) / 5.0)


def adversarial_weight(step, warmup=50, ramp=100, maximum=0.1):
    """Zero through warmup; first positive weight on warmup+1."""
    if warmup < 0 or ramp < 1 or maximum < 0:
        raise ValueError('Invalid GAN schedule')
    return maximum * min(1.0, max(0.0, (step - warmup) / ramp))


def score_mels(discriminator, mels, lengths):
    """Score each true-length utterance, never its batch padding."""
    return [discriminator(mels[i, :int(n)].T[None, None])
            for i, n in enumerate(lengths)]


def discriminator_loss(real_scores, fake_scores):
    return torch.stack([0.5 * ((r - 1).square().mean() + f.square().mean())
                        for r, f in zip(real_scores, fake_scores)]).mean()


def generator_loss(fake_scores):
    return torch.stack([(s - 1).square().mean() for s in fake_scores]).mean()
