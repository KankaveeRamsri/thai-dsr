import unittest

import numpy as np

from src.evaluation.pitch import analyze, compare

SR = 16000


def glide(f_start, f_end, seconds=1.0, voiced=None):
    """Harmonic tone with a log-linear F0 glide; `voiced` (start, end) in seconds, silence elsewhere."""
    t = np.arange(int(SR * seconds)) / SR
    f0 = f_start * (f_end / f_start) ** (t / seconds)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    x = sum(np.sin(k * phase) / k for k in range(1, 6)) * 0.3
    if voiced is not None:
        mask = (t >= voiced[0]) & (t < voiced[1])
        x = x * mask
    return x


class PitchMetricsTest(unittest.TestCase):
    def test_identical_signal(self):
        track = analyze(glide(120, 240), SR)
        m = compare(track, track)
        self.assertAlmostEqual(m["f0_corr"], 1.0, places=6)
        self.assertAlmostEqual(m["f0_rmse_st"], 0.0, places=6)
        self.assertEqual(m["vuv_agree"], 1.0)
        self.assertGreater(m["f0_n_voiced_both"], 50)

    def test_one_semitone_shift(self):
        ref = analyze(glide(120, 240), SR)
        est = analyze(glide(120 * 2 ** (1 / 12), 240 * 2 ** (1 / 12)), SR)
        m = compare(ref, est)
        self.assertAlmostEqual(m["f0_rmse_st"], 1.0, delta=0.1)
        self.assertGreater(m["f0_corr"], 0.99)  # a constant offset keeps the contour

    def test_inverted_contour_and_voicing(self):
        ref = analyze(glide(120, 240, voiced=(0.0, 1.0)), SR)
        est = analyze(glide(240, 120, voiced=(0.0, 0.5)), SR)
        m = compare(ref, est)
        self.assertLess(m["f0_corr"], -0.9)
        self.assertAlmostEqual(m["vuv_agree"], 0.5, delta=0.06)

    def test_noise_has_lower_hnr_and_too_few_voiced_frames(self):
        tone = analyze(glide(150, 150), SR)
        noise = analyze(np.random.default_rng(0).normal(scale=0.1, size=SR), SR)
        self.assertGreater(tone.hnr_db, noise.hnr_db + 10)
        m = compare(tone, noise)
        if m["f0_n_voiced_both"] < 10:
            self.assertTrue(np.isnan(m["f0_corr"]) and np.isnan(m["f0_rmse_st"]))

    def test_rejects_different_timelines(self):
        with self.assertRaises(ValueError):
            compare(analyze(glide(120, 240, 1.0), SR), analyze(glide(120, 240, 1.5), SR))


if __name__ == "__main__":
    unittest.main()
