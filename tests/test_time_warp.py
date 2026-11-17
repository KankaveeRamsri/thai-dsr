"""Clean <-> distorted time maps built from logged tempo segments."""

import unittest

import numpy as np

from src.preprocessing.distortion import segment_tempo_perturbation
from src.utils.mel import compute_mel
from src.utils.time_warp import MEL_HOP, TempoWarp, mel_frame_count

SR = 16000
FRAME_SR = 22050


def _modulated_noise(seconds=2.0, seed=0):
    """Noise carrier under a slowly varying (1-5 Hz) envelope, roughly speech-like."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    envelope = sum(rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * rng.uniform(1, 5) * t + rng.uniform(0, 6.3))
                   for _ in range(4))
    envelope = (envelope - envelope.min()) / (envelope.max() - envelope.min()) + 0.05
    return (0.3 * envelope * rng.standard_normal(len(t))).astype(np.float32)


def _log_energy(mel):
    return np.log(np.exp(mel).sum(axis=1))


class KnownSegmentsTest(unittest.TestCase):
    SEGMENTS = [[0, 8000, 0.8], [8000, 4000, None], [12000, 8000, 1.25]]

    def test_knots(self):
        warp = TempoWarp(self.SEGMENTS, SR)
        np.testing.assert_array_equal(warp.dist_knots, [0, 10000, 14000, 20400])
        self.assertAlmostEqual(float(warp.clean_to_distorted(0.5)), 10000 / SR)
        self.assertAlmostEqual(float(warp.clean_to_distorted(0.25)), 5000 / SR)
        self.assertAlmostEqual(float(warp.clean_to_distorted(1.0)), 17200 / SR)

    def test_round_trip_is_identity(self):
        warp = TempoWarp(self.SEGMENTS, SR)
        t = np.random.default_rng(1).uniform(0, warp.clean_samples / SR, 1000)
        np.testing.assert_allclose(warp.distorted_to_clean(warp.clean_to_distorted(t)), t, atol=1e-12)

    def test_rejects_non_contiguous_segments(self):
        with self.assertRaises(ValueError):
            TempoWarp([[0, 100, None], [150, 100, None]], SR)

    def test_warp_frames_alignment_error_near_zero(self):
        warp = TempoWarp(self.SEGMENTS, SR)
        n_clean = mel_frame_count(warp.clean_samples, SR)
        clean_time = (np.arange(n_clean) + 0.5) * MEL_HOP / FRAME_SR  # each frame stores its own time
        warped = warp.warp_frames(clean_time[:, None])[:, 0]
        target_time = warp.distorted_to_clean((np.arange(len(warped)) + 0.5) * MEL_HOP / FRAME_SR)
        inside = (target_time >= clean_time[0]) & (target_time <= clean_time[-1])
        self.assertGreater(inside.mean(), 0.95)
        self.assertLess(np.abs(warped[inside] - target_time[inside]).max(), 1e-9)


class AgainstDistortionTempoStageTest(unittest.TestCase):
    def setUp(self):
        self.clean = _modulated_noise()
        record = {}
        self.distorted = segment_tempo_perturbation(
            self.clean, SR, (0.3, 0.5), (0.6, 1.4), rng=np.random.default_rng(3), record=record)
        self.warp = TempoWarp(record["tempo_segments"], SR)

    def test_lengths_match_distorted_audio(self):
        self.assertEqual(self.warp.clean_samples, len(self.clean))
        self.assertEqual(self.warp.distorted_samples, len(self.distorted))
        self.assertEqual(mel_frame_count(len(self.distorted), SR),
                         compute_mel(self.distorted, sr=SR).shape[1])

    def test_stretched_reference_equals_tempo_stage(self):
        np.testing.assert_allclose(self.warp.stretch_waveform(self.clean), self.distorted, atol=1e-6)

    def test_warped_mel_is_frame_aligned_with_distorted_mel(self):
        """The warped clean envelope lines up with the distorted one at lag 0."""
        distorted_mel = compute_mel(self.distorted, sr=SR).T
        warped = self.warp.warp_frames(compute_mel(self.clean, sr=SR).T, n_target=len(distorted_mel))
        a, b = _log_energy(distorted_mel), _log_energy(warped)
        lags = np.arange(-4, 5)
        scores = np.array([np.corrcoef(np.roll(b, lag)[6:-6], a[6:-6])[0, 1] for lag in lags])
        best = int(np.argmax(scores))
        y0, y1, y2 = scores[best - 1:best + 2]
        lag = lags[best] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)  # sub-frame peak
        self.assertLess(abs(lag), 0.5)
        self.assertGreater(scores[lags == 0][0], 0.85)


if __name__ == "__main__":
    unittest.main()
