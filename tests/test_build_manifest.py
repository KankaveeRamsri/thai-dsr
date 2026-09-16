import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.preprocessing.build_manifest import build_manifest_df


def _write_wav(path, duration=0.1, sample_rate=16000):
    samples = np.zeros(round(duration * sample_rate), dtype=np.float32)
    sf.write(path, samples, sample_rate)


class MultiDirectoryManifestTest(unittest.TestCase):
    def test_build_manifest_from_multiple_directory_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            cv_dir = root / "common_voice"
            distorted_dir = root / "distorted"
            distorted_cv_dir = root / "distorted_cv"
            for audio_dir in (raw_dir, cv_dir, distorted_dir, distorted_cv_dir):
                audio_dir.mkdir()

            raw_name = "001_ทดสอบ"
            cv_name = "cv_0000"
            _write_wav(raw_dir / f"{raw_name}.wav")
            _write_wav(cv_dir / f"{cv_name}.wav")
            _write_wav(distorted_dir / f"{raw_name}_severe.wav")
            _write_wav(distorted_cv_dir / f"{cv_name}_severe.wav")

            with (cv_dir / "transcripts.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["filename", "sentence"])
                writer.writeheader()
                writer.writerow({
                    "filename": f"{cv_name}.wav",
                    "sentence": "ข้อความจาก CSV",
                })

            manifest = build_manifest_df(
                [raw_dir, cv_dir],
                [distorted_dir, distorted_cv_dir],
                severities=["severe"],
            ).set_index("utterance_id")

            self.assertEqual(set(manifest.index), {raw_name, cv_name})
            self.assertEqual(manifest.loc[raw_name, "transcript"], "ทดสอบ")
            self.assertEqual(manifest.loc[raw_name, "speaker_id"], "spk001")
            self.assertEqual(manifest.loc[cv_name, "transcript"], "ข้อความจาก CSV")
            self.assertEqual(manifest.loc[cv_name, "speaker_id"], "common_voice")
            self.assertEqual(set(manifest["severity"]), {"severe"})


if __name__ == "__main__":
    unittest.main()
