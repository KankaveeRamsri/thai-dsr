"""Fine-tune UNIVERSAL_V1 HiFi-GAN on clean Thai mel/waveform pairs.

The authoritative Week-5/6 corpus inputs are ``data/manifest_w5.csv`` and
``data/splits_w5.json`` (210 unique utterances: 147 train, 32 validation,
31 test). The older unsuffixed files describe only the original 10-utterance
dataset and are intentionally not the defaults here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.hifigan_dataset import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLITS,
    HiFiGANCleanDataset,
    prepare_hifigan_dataset,
)
from src.utils.mel import compute_mel_tensor


REPO_ROOT = Path(__file__).resolve().parents[2]
HIFIGAN_DIR = REPO_ROOT / "vendor" / "hifi-gan"
DEFAULT_PRETRAINED_DIR = HIFIGAN_DIR / "checkpoints" / "UNIVERSAL_V1"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "results" / "checkpoints" / "hifigan_thai"
sys.path.insert(0, os.fspath(HIFIGAN_DIR))

from env import AttrDict  # noqa: E402
from models import (  # noqa: E402
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    discriminator_loss,
    feature_loss,
    generator_loss,
)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(model_dir: Path) -> AttrDict:
    with (model_dir / "config.json").open() as handle:
        return AttrDict(json.load(handle))


def load_pretrained_generator(model_dir: Path, config: AttrDict, device: torch.device):
    candidates = sorted(
        path for path in model_dir.iterdir() if path.name.startswith(("g_", "generator"))
    )
    if not candidates:
        raise FileNotFoundError(f"No generator checkpoint found in {model_dir}")
    generator = Generator(config).to(device)
    checkpoint = torch.load(candidates[-1], map_location=device, weights_only=True)
    generator.load_state_dict(checkpoint["generator"])
    return generator, candidates[-1]


def save_checkpoint(output_dir, step, epoch, generator, mpd, msd, optim_g, optim_d):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_path = output_dir / f"g_{step:08d}"
    # Historical generator files are what we need for listening comparisons.
    # Keep only one large discriminator/optimizer state to avoid ~1 GiB per step.
    training_path = output_dir / "do_latest"
    training_path_tmp = output_dir / "do_latest.tmp"
    torch.save({"generator": generator.state_dict()}, generator_path)
    torch.save(
        {
            "mpd": mpd.state_dict(),
            "msd": msd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "steps": step,
            "epoch": epoch,
        },
        training_path_tmp,
    )
    os.replace(training_path_tmp, training_path)
    return generator_path, training_path


def resume_checkpoint(output_dir, generator, mpd, msd, optim_g, optim_d, device):
    """Restore the most recent numbered generator and matching training state."""
    output_dir = Path(output_dir)
    training_path = output_dir / "do_latest"
    if not training_path.is_file():
        raise FileNotFoundError(f"Cannot resume; missing {training_path}")
    state = torch.load(training_path, map_location=device, weights_only=True)
    step = int(state["steps"])
    epoch = int(state["epoch"])
    generator_path = output_dir / f"g_{step:08d}"
    if not generator_path.is_file():
        raise FileNotFoundError(
            f"Cannot resume step {step}; missing matching generator {generator_path}"
        )
    generator.load_state_dict(
        torch.load(generator_path, map_location=device, weights_only=True)["generator"]
    )
    mpd.load_state_dict(state["mpd"])
    msd.load_state_dict(state["msd"])
    optim_g.load_state_dict(state["optim_g"])
    optim_d.load_state_dict(state["optim_d"])
    return step, epoch, generator_path


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def waveform_to_mel(waveform, config):
    return compute_mel_tensor(
        waveform.squeeze(1),
        sr=int(config.sampling_rate),
        n_fft=int(config.n_fft),
        hop_length=int(config.hop_size),
        win_length=int(config.win_size),
        n_mels=int(config.num_mels),
        fmax=config.fmax_for_loss,
    )


@torch.no_grad()
def validate(generator, loader, config, device, max_items=None):
    generator.eval()
    total = 0.0
    count = 0
    for mel, audio, _ in loader:
        mel = mel.to(device)
        audio = audio.to(device)
        prediction = generator(mel)
        prediction_mel = waveform_to_mel(prediction, config)
        target_mel = compute_mel_tensor(
            audio,
            sr=int(config.sampling_rate),
            n_fft=int(config.n_fft),
            hop_length=int(config.hop_size),
            win_length=int(config.win_size),
            n_mels=int(config.num_mels),
            fmax=config.fmax_for_loss,
        )
        total += F.l1_loss(prediction_mel, target_mel).item()
        count += 1
        if max_items is not None and count >= max_items:
            break
    generator.train()
    return total / count


def train(args):
    device = choose_device(args.device)
    seed_everything(args.seed)
    model_dir = Path(args.pretrained_dir).resolve()
    config = load_config(model_dir)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.num_workers is not None:
        config.num_workers = args.num_workers

    index_path = Path(args.dataset_dir).resolve() / "index.json"
    if args.prepare or not index_path.is_file():
        prepared = prepare_hifigan_dataset(
            args.manifest,
            args.splits,
            args.dataset_dir,
            sample_rate=int(config.sampling_rate),
            n_fft=int(config.n_fft),
            hop_size=int(config.hop_size),
            win_size=int(config.win_size),
            num_mels=int(config.num_mels),
            fmax=int(config.fmax),
        )
        print(f"Prepared dataset: {prepared['split_counts']} at {index_path}")
    if args.prepare_only:
        return

    trainset = HiFiGANCleanDataset(
        index_path, "train", config.segment_size, random_crop=True, seed=args.seed
    )
    valset = HiFiGANCleanDataset(
        index_path, "val", config.segment_size, random_crop=False, seed=args.seed
    )
    train_loader = DataLoader(
        trainset,
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

    generator, pretrained_path = load_pretrained_generator(model_dir, config, device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    generator.train()
    mpd.train()
    msd.train()
    optim_g = torch.optim.AdamW(
        generator.parameters(), config.learning_rate, betas=(config.adam_b1, config.adam_b2)
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        config.learning_rate,
        betas=(config.adam_b1, config.adam_b2),
    )

    output_dir = Path(args.checkpoint_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep each checkpoint directory directly loadable by src/inference/run.py.
    shutil.copy2(model_dir / "config.json", output_dir / "config.json")
    step = 0
    epoch = 0
    if args.resume:
        step, epoch, generator_path = resume_checkpoint(
            output_dir, generator, mpd, msd, optim_g, optim_d, device
        )
        pretrained_path = generator_path

    print(f"Device: {device}")
    print(f"Starting generator: {pretrained_path}")
    print(
        f"Data: train={len(trainset)}, val={len(valset)}, "
        f"batch={config.batch_size}, segment={config.segment_size} samples"
    )
    started = time.perf_counter()
    recent_times = []
    while step < args.max_steps:
        for mel, audio, _ in train_loader:
            batch_started = time.perf_counter()
            mel = mel.to(device)
            real = audio.unsqueeze(1).to(device)
            generated = generator(mel)
            if generated.shape != real.shape:
                raise RuntimeError(
                    f"Generator/target shape mismatch: {tuple(generated.shape)} vs {tuple(real.shape)}"
                )

            set_requires_grad(mpd, True)
            set_requires_grad(msd, True)
            optim_d.zero_grad(set_to_none=True)
            mpd_real, mpd_fake, _, _ = mpd(real, generated.detach())
            msd_real, msd_fake, _, _ = msd(real, generated.detach())
            loss_disc_mpd, _, _ = discriminator_loss(mpd_real, mpd_fake)
            loss_disc_msd, _, _ = discriminator_loss(msd_real, msd_fake)
            loss_disc = loss_disc_mpd + loss_disc_msd
            loss_disc.backward()
            optim_d.step()

            set_requires_grad(mpd, False)
            set_requires_grad(msd, False)
            optim_g.zero_grad(set_to_none=True)
            generated_mel = waveform_to_mel(generated, config)
            target_mel = compute_mel_tensor(
                real.squeeze(1),
                sr=int(config.sampling_rate),
                n_fft=int(config.n_fft),
                hop_length=int(config.hop_size),
                win_length=int(config.win_size),
                n_mels=int(config.num_mels),
                fmax=config.fmax_for_loss,
            )
            loss_mel_raw = F.l1_loss(target_mel, generated_mel)
            _, mpd_fake, fmap_mpd_real, fmap_mpd_fake = mpd(real, generated)
            _, msd_fake, fmap_msd_real, fmap_msd_fake = msd(real, generated)
            loss_fm = feature_loss(fmap_mpd_real, fmap_mpd_fake) + feature_loss(
                fmap_msd_real, fmap_msd_fake
            )
            loss_gen_mpd, _ = generator_loss(mpd_fake)
            loss_gen_msd, _ = generator_loss(msd_fake)
            loss_gen = loss_gen_mpd + loss_gen_msd + loss_fm + 45 * loss_mel_raw
            loss_gen.backward()
            optim_g.step()
            set_requires_grad(mpd, True)
            set_requires_grad(msd, True)

            step += 1
            elapsed = time.perf_counter() - batch_started
            recent_times.append(elapsed)
            if step == 1 or step % args.log_interval == 0:
                print(
                    f"step={step} gen={loss_gen.item():.4f} disc={loss_disc.item():.4f} "
                    f"mel={loss_mel_raw.item():.4f} sec/step={elapsed:.3f}",
                    flush=True,
                )
            if args.checkpoint_interval and step % args.checkpoint_interval == 0:
                paths = save_checkpoint(
                    output_dir, step, epoch, generator, mpd, msd, optim_g, optim_d
                )
                print(f"Saved checkpoints: {paths[0].name}, {paths[1].name}")
            if args.validation_interval and step % args.validation_interval == 0:
                val_error = validate(
                    generator, val_loader, config, device, max_items=args.max_val_items
                )
                print(f"step={step} val_mel={val_error:.4f}", flush=True)
            if step >= args.max_steps:
                break
        epoch += 1

    if args.save_final and (not args.checkpoint_interval or step % args.checkpoint_interval):
        paths = save_checkpoint(output_dir, step, epoch, generator, mpd, msd, optim_g, optim_d)
        print(f"Saved final checkpoints: {paths[0].name}, {paths[1].name}")
    total_time = time.perf_counter() - started
    mean_step = sum(recent_times) / len(recent_times)
    print(
        f"Finished {step} steps in {total_time:.1f}s ({mean_step:.3f}s/step); "
        f"estimated 1,000 steps={mean_step * 1000 / 60:.1f}min"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=os.fspath(DEFAULT_MANIFEST))
    parser.add_argument("--splits", default=os.fspath(DEFAULT_SPLITS))
    parser.add_argument("--dataset-dir", default=os.fspath(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pretrained-dir", default=os.fspath(DEFAULT_PRETRAINED_DIR))
    parser.add_argument("--checkpoint-dir", default=os.fspath(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--prepare", action="store_true", help="Rebuild cached mel/waveform pairs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--validation-interval", type=int, default=500)
    parser.add_argument("--max-val-items", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume", action="store_true", help="Resume from do_latest")
    parser.add_argument("--save-final", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
