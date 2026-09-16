"""HiFi-GAN-compatible mel-spectrogram extraction.

This mirrors ``vendor/hifi-gan/meldataset.py``: reflect padding, a Hann
window, a non-centred STFT, magnitude (not power) spectra, the librosa mel
filter bank, and natural-log dynamic-range compression.
"""

from functools import lru_cache

import librosa
import numpy as np
import torch
import torch.nn.functional as F


HIFIGAN_SAMPLE_RATE = 22050
HIFIGAN_FMIN = 0
LOG_CLIP_VALUE = 1e-5


@lru_cache(maxsize=16)
def _mel_basis(sr, n_fft, n_mels, fmax):
    return librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=HIFIGAN_FMIN,
        fmax=fmax,
    ).astype(np.float32)


def compute_mel(
    audio,
    sr=22050,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=80,
    fmax=8000,
    power=1,
):
    """Return a HiFi-GAN-compatible log-mel array shaped ``(n_mels, T)``.

    ``sr`` describes the input audio. Audio at another sample rate is
    resampled to the 22.05 kHz rate expected by the bundled universal
    HiFi-GAN checkpoint before the spectrogram is calculated.
    """
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"compute_mel expects mono 1-D audio, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("compute_mel received empty audio")
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")
    if power <= 0:
        raise ValueError(f"power must be positive, got {power}")

    if sr != HIFIGAN_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sr, target_sr=HIFIGAN_SAMPLE_RATE
        ).astype(np.float32)
        sr = HIFIGAN_SAMPLE_RATE

    pad = (n_fft - hop_length) // 2
    if pad < 0:
        raise ValueError("n_fft must be greater than or equal to hop_length")
    if waveform.size <= pad:
        raise ValueError(
            f"Audio is too short for HiFi-GAN reflect padding: {waveform.size} samples"
        )

    tensor = torch.from_numpy(waveform).unsqueeze(0)
    tensor = F.pad(tensor.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    window = torch.hann_window(win_length, dtype=tensor.dtype, device=tensor.device)
    complex_spectrum = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    # HiFi-GAN adds 1e-9 before sqrt rather than using complex.abs().
    spectrum = torch.sqrt(
        complex_spectrum.real.pow(2) + complex_spectrum.imag.pow(2) + 1e-9
    )
    if power != 1:
        spectrum = spectrum.pow(power)

    basis = torch.from_numpy(_mel_basis(sr, n_fft, n_mels, fmax)).to(
        device=tensor.device, dtype=tensor.dtype
    )
    mel = torch.matmul(basis, spectrum)
    log_mel = torch.log(torch.clamp(mel, min=LOG_CLIP_VALUE))
    return log_mel.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
