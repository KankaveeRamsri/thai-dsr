import tempfile
import unittest
from pathlib import Path

import torch

from src.training.finetune_hifigan import resume_checkpoint, save_checkpoint


def _optimizer_step(module, optimizer):
    optimizer.zero_grad()
    sum(parameter.square().sum() for parameter in module.parameters()).backward()
    optimizer.step()


class HiFiGANCheckpointTest(unittest.TestCase):
    def test_numbered_generators_and_latest_state_resume_together(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            generator = torch.nn.Linear(3, 2)
            mpd = torch.nn.Linear(2, 1)
            msd = torch.nn.Linear(2, 1)
            optim_g = torch.optim.AdamW(generator.parameters())
            optim_d = torch.optim.AdamW(
                list(mpd.parameters()) + list(msd.parameters())
            )
            _optimizer_step(generator, optim_g)
            _optimizer_step(mpd, optim_d)
            save_checkpoint(
                output_dir, 500, 3, generator, mpd, msd, optim_g, optim_d
            )
            expected = {
                name: tensor.detach().clone() for name, tensor in generator.state_dict().items()
            }
            save_checkpoint(
                output_dir, 1000, 6, generator, mpd, msd, optim_g, optim_d
            )

            self.assertTrue((output_dir / "g_00000500").is_file())
            self.assertTrue((output_dir / "g_00001000").is_file())
            self.assertTrue((output_dir / "do_latest").is_file())

            restored_g = torch.nn.Linear(3, 2)
            restored_mpd = torch.nn.Linear(2, 1)
            restored_msd = torch.nn.Linear(2, 1)
            restored_optim_g = torch.optim.AdamW(restored_g.parameters())
            restored_optim_d = torch.optim.AdamW(
                list(restored_mpd.parameters()) + list(restored_msd.parameters())
            )
            step, epoch, generator_path = resume_checkpoint(
                output_dir,
                restored_g,
                restored_mpd,
                restored_msd,
                restored_optim_g,
                restored_optim_d,
                torch.device("cpu"),
            )

            self.assertEqual((step, epoch), (1000, 6))
            self.assertEqual(generator_path.name, "g_00001000")
            for name, tensor in restored_g.state_dict().items():
                torch.testing.assert_close(tensor, expected[name])
            self.assertTrue(restored_optim_g.state_dict()["state"])
            self.assertTrue(restored_optim_d.state_dict()["state"])
