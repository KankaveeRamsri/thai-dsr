import tempfile
import unittest
from pathlib import Path

import torch

from src.preprocessing.extract_embedding import save_hidden_states


class EmbeddingCacheTest(unittest.TestCase):
    def test_selected_layer_only_is_saved(self):
        hidden_states = tuple(torch.full((1, 3, 4), float(index)) for index in range(25))
        with tempfile.TemporaryDirectory() as directory:
            save_hidden_states(
                hidden_states,
                "example.wav",
                directory,
                layer=9,
            )
            files = sorted(path.name for path in Path(directory).glob("*.npy"))
        self.assertEqual(files, ["example_layer09.npy"])


if __name__ == "__main__":
    unittest.main()
