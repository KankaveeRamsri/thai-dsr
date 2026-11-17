"""The optional ``rng`` / ``record`` arguments must not change default behaviour.

Reference values were recorded from distortion.py before ``rng`` was added
(commit cfabcc3), on a fixed synthetic voiced signal with a fixed global seed.

Praat's "Change gender" is not bit-reproducible between runs (output varies
slightly with identical input), so whole-pipeline audio is compared by RMS
with a tolerance. Everything driven by the random draws is compared exactly:
the global RNG state after the run (same number and order of draws), output
lengths (set by the tempo segment draws), and the tempo and noise/reverb
stages, which do not use Praat.
"""

import random
import unittest

import numpy as np

from src.preprocessing import distortion as D

SR = 16000

REFERENCE = {
    "pipeline": {
        "mild": {"len": 20384, "rms": 0.2348228635537722},
        "moderate": {"len": 21118, "rms": 0.22662632072593059},
        "severe": {"len": 17907, "rms": 0.27583238524922393},
    },
    "next_py": 0.019299566309716853,
    "next_np": 0.8080242494457455,
    "tempo": {"len": 25820, "sum": 13.455880049809366, "sumsq": 915.7694883281229},
    "noise": {"len": 19200, "sum": 4.501660674340201, "sumsq": 1759.8372891573683},
}


def _voiced_signal():
    t = np.arange(int(1.2 * SR)) / SR
    f0 = 140 + 10 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    return (sum(np.sin(k * phase) / k for k in range(1, 6)) * 0.3).astype(np.float32)


def _stats(x):
    x = x.astype(np.float64)
    return {"len": len(x), "sum": float(x.sum()), "sumsq": float((x ** 2).sum())}


class DefaultBehaviourUnchangedTest(unittest.TestCase):
    def test_pipeline_matches_reference_with_global_seed(self):
        y = _voiced_signal()
        random.seed(1234)
        np.random.seed(1234)
        for severity in D.SEVERITY_LEVELS:
            out = D.apply_distortion(y, SR, severity)
            ref = REFERENCE["pipeline"][severity]
            self.assertEqual(len(out), ref["len"], severity)
            rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
            self.assertAlmostEqual(rms / ref["rms"], 1.0, delta=0.03, msg=severity)
        self.assertEqual(random.random(), REFERENCE["next_py"])
        self.assertEqual(float(np.random.random()), REFERENCE["next_np"])

    def test_tempo_and_noise_stages_match_reference(self):
        y = _voiced_signal()
        params = D.SEVERITY_PARAMS["severe"]
        random.seed(7)
        np.random.seed(7)
        tempo = D.segment_tempo_perturbation(
            y, SR, params["segment_duration_range"], params["tempo_rate_range"])
        random.seed(7)
        np.random.seed(7)
        noise = D.add_noise_and_reverb(
            y, SR, params["snr_db_range"], params["reverb_wet_mix"], params["reverb_decay"])
        for name, out in (("tempo", tempo), ("noise", noise)):
            got, ref = _stats(out), REFERENCE[name]
            self.assertEqual(got["len"], ref["len"], name)
            self.assertAlmostEqual(got["sum"], ref["sum"], places=6, msg=name)
            self.assertAlmostEqual(got["sumsq"], ref["sumsq"], places=6, msg=name)


class GeneratorTest(unittest.TestCase):
    def _run(self, seed, severity="severe"):
        record = {}
        out = D.apply_distortion(_voiced_signal(), SR, severity,
                                 rng=np.random.default_rng(seed), record=record)
        return out, record

    # Measured from Praat output (not drawn), so subject to Praat's run-to-run noise.
    MEASURED = {"peak_before_norm", "jitter_mean", "jitter_sd", "pitch_points"}

    def test_same_seed_gives_same_parameters_regardless_of_order(self):
        out_a, rec_a = self._run(11)
        self._run(99)  # an unrelated file processed in between
        out_b, rec_b = self._run(11)
        drawn = lambda r: {k: v for k, v in r.items() if k not in self.MEASURED}
        self.assertEqual(drawn(rec_a), drawn(rec_b))
        self.assertEqual(len(out_a), len(out_b))

    def test_generator_does_not_touch_global_state(self):
        random.seed(5)
        np.random.seed(5)
        py_state, np_state = random.getstate(), np.random.get_state()[1].copy()
        self._run(3)
        self.assertEqual(random.getstate(), py_state)
        np.testing.assert_array_equal(np.random.get_state()[1], np_state)

    def test_record_contains_parameters_and_status(self):
        _, record = self._run(21)
        self.assertEqual(record["pitch_status"], "ok")
        for key in ("formant_shift_ratio", "f0_shift_factor", "jitter_mean",
                    "tempo_segments", "snr_db", "reverb_wet_mix", "output_samples"):
            self.assertIsNotNone(record[key], key)
        low, high = D.SEVERITY_PARAMS["severe"]["snr_db_range"]
        self.assertTrue(low <= record["snr_db"] <= high)

    def test_silence_reports_no_pitch_points(self):
        record = {}
        D.apply_distortion(np.zeros(SR, dtype=np.float32), SR, "mild",
                           rng=np.random.default_rng(0), record=record)
        self.assertIn(record["pitch_status"], {"no_pitch_points", "praat_error"})


if __name__ == "__main__":
    unittest.main()
