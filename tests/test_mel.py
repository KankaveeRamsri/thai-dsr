import os
import sys
import unittest
from unittest.mock import patch

import librosa
import numpy as np
import torch

from src.utils.mel import LOG_CLIP_VALUE, compute_mel


VENDOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor", "hifi-gan"))
sys.path.insert(0, VENDOR_DIR)
import meldataset  # noqa: E402


class MelCompatibilityTest(unittest.TestCase):
    def test_compute_mel_matches_hifigan_reference(self):
        sample_rate = 22050
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)

        actual = compute_mel(audio, sr=sample_rate)

        original_stft = torch.stft

        def legacy_stft(*args, **kwargs):
            return original_stft(*args, **kwargs, return_complex=False)

        def legacy_mel(sr, n_fft, n_mels, fmin, fmax):
            return librosa.filters.mel(
                sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
            )

        with patch.object(meldataset, "librosa_mel_fn", side_effect=legacy_mel), patch.object(
            meldataset.torch, "stft", side_effect=legacy_stft
        ):
            reference = meldataset.mel_spectrogram(
                torch.from_numpy(audio).unsqueeze(0),
                1024,
                80,
                sample_rate,
                256,
                1024,
                0,
                8000,
            ).squeeze(0).numpy()

        self.assertEqual(actual.shape, reference.shape)
        self.assertEqual(actual.shape[0], 80)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), float(np.log(LOG_CLIP_VALUE)) - 1e-6)
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
