"""Speaker-clustered bootstrap used by scripts/summarize_cv_eval.py."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "summarize_cv_eval", Path(__file__).resolve().parents[1] / "scripts/summarize_cv_eval.py")
summarize = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summarize)


class ClusterBootstrapTest(unittest.TestCase):
    def test_mean_and_ci_cover_it(self):
        rng = np.random.default_rng(0)
        clusters = np.repeat(np.arange(40), 6)
        values = rng.normal(0, 1, 40)[clusters] + rng.normal(0, 0.1, len(clusters))
        n, mean, low, high = summarize.cluster_bootstrap(values, clusters, 2000, 42)
        self.assertEqual(n, len(values))
        self.assertAlmostEqual(mean, values.mean())
        self.assertLess(low, mean)
        self.assertGreater(high, mean)

    def test_wider_than_clip_bootstrap_when_clips_are_correlated(self):
        """Clips of one speaker move together, so resampling speakers must give a wider CI."""
        rng = np.random.default_rng(1)
        clusters = np.repeat(np.arange(30), 8)
        values = rng.normal(0, 1, 30)[clusters] + rng.normal(0, 0.05, len(clusters))
        _, _, low_spk, high_spk = summarize.cluster_bootstrap(values, clusters, 2000, 42)
        _, _, low_clip, high_clip = summarize.cluster_bootstrap(values, np.arange(len(values)), 2000, 42)
        self.assertGreater(high_spk - low_spk, 2 * (high_clip - low_clip))

    def test_ignores_nan(self):
        n, mean, _, _ = summarize.cluster_bootstrap([1.0, np.nan, 3.0], ["a", "a", "b"], 100, 42)
        self.assertEqual((n, mean), (2, 2.0))

    def test_reproducible(self):
        args = (np.arange(20.0), np.repeat(np.arange(5), 4), 500, 42)
        self.assertEqual(summarize.cluster_bootstrap(*args), summarize.cluster_bootstrap(*args))

    def test_wilcoxon_handles_degenerate_input(self):
        self.assertTrue(np.isnan(summarize.safe_wilcoxon([0.0, 0.0, 0.0])))
        self.assertTrue(np.isnan(summarize.safe_wilcoxon([1.0])))
        self.assertLess(summarize.safe_wilcoxon(np.linspace(0.1, 1.0, 20)), 0.001)


class ClusterRatioBootstrapTest(unittest.TestCase):
    def test_ratio_is_corpus_rate_with_ci(self):
        rng = np.random.default_rng(3)
        clusters = np.repeat(np.arange(40), 6)
        den = rng.integers(20, 40, len(clusters)).astype(float)
        num = np.round(den * rng.uniform(0.05, 0.4, 40)[clusters])
        n, ratio, low, high = summarize.cluster_ratio_bootstrap(num, den, clusters, 2000, 42)
        self.assertEqual(n, len(num))
        self.assertAlmostEqual(ratio, num.sum() / den.sum())
        self.assertLess(low, ratio)
        self.assertGreater(high, ratio)

    def test_same_resamples_as_mean_bootstrap(self):
        """With unit denominators the ratio bootstrap is the mean bootstrap."""
        rng = np.random.default_rng(4)
        clusters = np.repeat(np.arange(20), 5)
        values = rng.normal(size=len(clusters))
        a = summarize.cluster_bootstrap(values, clusters, 1000, 42)
        b = summarize.cluster_ratio_bootstrap(values, np.ones_like(values), clusters, 1000, 42)
        np.testing.assert_allclose(a, b)


if __name__ == "__main__":
    unittest.main()
