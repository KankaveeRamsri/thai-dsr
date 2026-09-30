"""Train the configured wav2vec2-to-mel Bi-LSTM mapper."""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.mapper import build_mapper_from_config
from src.training.dataset import DysarthricDataset, collate_fn
from src.training.grad_clip import clip
from src.training.splits import SPLIT_NAMES, load_or_create_splits, row_indices_for_split
from src.utils.config import (
    DEFAULT_DATA_CONFIG_PATH,
    DEFAULT_MODEL_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    REPO_ROOT,
    load_yaml_config,
)


COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def project_path(path):
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference="auto"):
    """Auto-detect the best available device unless one is requested."""
    if preference not in (None, "auto"):
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_l1_loss(pred, target, lengths):
    """L1 mel loss that excludes padded timesteps."""
    max_len = pred.size(1)
    mel_dim = pred.size(-1)
    mask = (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]).float()
    mask = mask.unsqueeze(-1)
    diff = (pred - target).abs() * mask
    return diff.sum() / (mask.sum() * mel_dim).clamp(min=1)


def run_epoch(model, loader, device, reconstruction_weight, optimizer=None, desc="epoch",
              extras=None):
    """Run one pass over a loader; train only when an optimizer is supplied.

    ``extras`` (a StepExtras) supplies encoder features, gradient clipping and
    per-step checkpoint/eval/max-steps hooks; without it the loop is unchanged.
    """
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(is_train):
        for batch in progress_bar:
            if extras is None:
                embeddings, mels, lengths = batch
                embeddings = embeddings.to(device)
            else:
                embeddings = extras.embed(batch)
                _, mels, lengths = batch
            mels = mels.to(device)
            lengths = lengths.to(device)

            predictions = model(embeddings, lengths)
            loss = reconstruction_weight * masked_l1_loss(predictions, mels, lengths)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if extras is not None:
                    extras.clip_gradients()
                optimizer.step()

            batch_size = embeddings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if is_train and extras is not None and extras.after_step(loss.item()):
                break

    if total_samples == 0:
        raise ValueError("Cannot run an epoch with an empty data loader")
    return total_loss / total_samples


def build_optimizer(model, config):
    name = str(config["optimizer"]).lower()
    kwargs = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, name, num_epochs):
    name = None if name is None else str(name).lower()
    if name in (None, "none", "null"):
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(num_epochs, 1)
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(num_epochs // 3, 1), gamma=0.5
        )
    raise ValueError(f"Unsupported lr_scheduler: {name}")


def step_scheduler(scheduler, scheduler_name, val_loss):
    if scheduler is None:
        return
    if str(scheduler_name).lower() == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def plot_loss_curve(train_losses, val_losses, output_path):
    epochs = list(range(1, len(train_losses) + 1))
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(epochs, train_losses, color=COLOR_TRAIN, linewidth=2, label="Train loss")
    ax.plot(epochs, val_losses, color=COLOR_VAL, linewidth=2, label="Val loss")
    best_epoch = int(np.argmin(val_losses)) + 1
    ax.scatter(
        [best_epoch], [val_losses[best_epoch - 1]],
        color=COLOR_VAL, s=50, zorder=5, edgecolor=INK, linewidth=0.8,
    )
    ax.set_title("Mapper training: L1 loss on mel-spectrogram", loc="left", fontsize=12, color=INK)
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("L1 loss", color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def checkpoint_payload(
    epoch, model, optimizer, scheduler, val_loss, best_val_loss,
    train_config, data_config, model_config, train_losses, val_losses, extra=None,
):
    return {**(extra or {}),
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "train_config": train_config,
        "data_config": data_config,
        "model_config": model_config,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def print_effective_config(values):
    print("\nEffective configuration:")
    for key, value in values.items():
        print(f"  {key}: {value}")


class StepExtras:
    """Per-step hooks for the optional features; inactive options cost nothing.

    - encoder (PartialWav2Vec2 or None): features from partially unfrozen wav2vec2
    - gradient clipping (rescale, logged to gradients.jsonl) per parameter group
    - eval every N steps: W5 32-utterance mel L1 + 8 fixed W5 clips STOI/PESQ,
      appended to eval_log.jsonl (same format as the mel-adversarial trainer)
    - checkpoints/step_N.pt + checkpoints/latest.pt every N steps
    - stop after --max-steps optimizer steps
    """

    def __init__(self, *, mapper, encoder, dataset, device, args, out_dir, payload):
        self.mapper, self.encoder, self.dataset, self.device = mapper, encoder, dataset, device
        self.args, self.out_dir, self.payload = args, out_dir, payload
        self.step, self.done, self.nonfinite = 0, False, False
        self.eval_cache, self.voc = {}, None
        self.gradients = out_dir / "gradients.jsonl"

    def embed(self, batch):
        if self.encoder is None:
            return batch[0].to(self.device)
        keys, _, lengths = batch
        return self.encoder.batch_features(keys, lengths)

    def clip_gradients(self):
        common = dict(diagnostics=self.gradients, step=self.step + 1)
        if self.args.mapper_clip_norm > 0:
            clip(self.mapper.parameters(), component="mapper",
                 max_norm=self.args.mapper_clip_norm, **common)
        if self.encoder is not None and self.args.wav2vec_clip_norm > 0:
            clip(self.encoder.trainable_parameters(), component="wav2vec2",
                 max_norm=self.args.wav2vec_clip_norm, **common)

    def after_step(self, loss):
        """Return True once --max-steps optimizer steps have run."""
        self.step += 1
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {self.step}")
        if self.args.checkpoint_every and self.step % self.args.checkpoint_every == 0:
            self.save_periodic()
        if self.args.eval_every and self.step % self.args.eval_every == 0:
            self.evaluate()
        self.done = bool(self.args.max_steps) and self.step >= self.args.max_steps
        return self.done

    def save_periodic(self):
        folder = self.out_dir / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"step_{self.step:06d}.pt"
        torch.save(self.payload(), path)
        temporary = folder / "latest.pt.tmp"
        temporary.unlink(missing_ok=True)
        try:
            os.link(path, temporary)  # no second copy on disk (Drive/FUSE may refuse)
        except OSError:
            temporary.write_bytes(path.read_bytes())
        os.replace(temporary, folder / "latest.pt")

    def _eval_data(self, row):
        from src.inference.run import compute_mel_frame_count

        uid = row["utterance_id"]
        if uid not in self.eval_cache:
            mel = torch.from_numpy(self.dataset._extract_clean_mel(row["clean_path"])).float()
            frames = compute_mel_frame_count(REPO_ROOT / row["distorted_path"])
            self.eval_cache[uid] = (mel, frames)
        mel, frames = self.eval_cache[uid]
        if self.encoder is None:
            features = torch.from_numpy(self.dataset._load_distorted_embedding(uid, "severe")).float()
        else:
            with torch.no_grad():
                features = self.encoder(uid, REPO_ROOT / row["distorted_path"]).float().cpu()
        return features, mel, frames

    def evaluate(self):
        from src.evaluation.mapper_eval import audio_metrics, load_vocoder, mel_l1, w5_eval_rows

        if self.voc is None:
            self.voc = load_vocoder()
            self.w5_val, self.w5_audio = w5_eval_rows()
        was_training = self.mapper.training
        self.mapper.eval()
        mel = mel_l1(self.mapper, self.w5_val, self._eval_data)
        audio = audio_metrics(self.mapper, self.voc, self.w5_audio, self._eval_data,
                              self.out_dir / "eval" / f"step_{self.step:06d}")
        self.mapper.train(was_training)
        entry = dict(step=self.step, mel_l1=mel, stoi=audio["stoi"], pesq=audio["pesq"])
        with (self.out_dir / "eval_log.jsonl").open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print("EVAL", json.dumps(entry), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Thai-DSR mapper.")
    parser.add_argument("--train-config", default=os.fspath(DEFAULT_TRAIN_CONFIG_PATH))
    parser.add_argument("--data-config", default=os.fspath(DEFAULT_DATA_CONFIG_PATH))
    parser.add_argument("--model-config", default=os.fspath(DEFAULT_MODEL_CONFIG_PATH))
    parser.add_argument("--num-epochs", type=int, default=None, help="Temporary run override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Temporary loader override.")
    parser.add_argument("--checkpoint-dir", default=None, help="Temporary output override.")
    parser.add_argument("--log-dir", default=None, help="Temporary output override.")
    parser.add_argument("--layer", type=int, default=None, help="Override the wav2vec2 layer.")
    parser.add_argument("--resume-checkpoint", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--best-checkpoint", default=None, help="Best-checkpoint output path.")
    parser.add_argument("--loss-curve", default=None, help="Loss-curve output path.")
    parser.add_argument(
        "--early-stop-patience", type=int, default=None,
        help="Stop after this many epochs without a new best val loss.",
    )
    parser.add_argument("--manifest", default=None, help="Override the config manifest_path.")
    parser.add_argument("--splits", default=None, help="Override the config splits_path.")
    parser.add_argument(
        "--init-checkpoint", default=None,
        help="Start the mapper (and unfrozen wav2vec2 blocks, if saved) from this checkpoint; "
             "fresh optimizer. Unlike --resume-checkpoint, epochs restart at 1.",
    )
    parser.add_argument(
        "--unfreeze-top-layers", type=int, default=0,
        help="Train the N wav2vec2 transformer blocks directly below the selected layer "
             "(0 = fully frozen, cached embeddings).",
    )
    parser.add_argument("--wav2vec-lr", type=float, default=1e-5,
                        help="Learning rate for the unfrozen wav2vec2 blocks.")
    parser.add_argument("--wav2vec-clip-norm", type=float, default=1.0,
                        help="Gradient-norm clip for the unfrozen wav2vec2 blocks (0 = off).")
    parser.add_argument("--mapper-clip-norm", type=float, default=0.0,
                        help="Gradient-norm clip for the mapper (0 = off, the original recipe).")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Every N steps (and at step 0) log W5 mel L1 + 8-clip STOI/PESQ "
                             "to eval_log.jsonl (0 = off).")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Every N steps save checkpoints/step_N.pt and checkpoints/latest.pt (0 = off).")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N optimizer steps (0 = no limit).")
    return parser.parse_args()


def main():
    args = parse_args()
    train_config = load_yaml_config(args.train_config)
    data_config = load_yaml_config(args.data_config)
    model_config = load_yaml_config(args.model_config)
    if args.layer is not None:
        if not 0 <= args.layer <= 24:
            raise ValueError(f"layer must be between 0 and 24, got {args.layer}")
        train_config["data"]["layer"] = args.layer

    runtime = train_config["runtime"]
    optim_config = train_config["optim"]
    logging_config = train_config["logging"]
    seed = int(runtime["seed"])
    set_seed(seed)
    device = get_device(runtime["device"])

    num_epochs = (
        int(optim_config["num_epochs"])
        if args.num_epochs is None
        else args.num_epochs
    )
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {num_epochs}")
    checkpoint_dir = project_path(args.checkpoint_dir or logging_config["checkpoint_dir"])
    log_dir = project_path(args.log_dir or logging_config["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    paths = data_config["paths"]
    train_data_config = train_config["data"]
    manifest_path = args.manifest or train_data_config.get("manifest_path", paths["manifest_path"])
    splits_path = args.splits or train_data_config.get("splits_path", paths["splits_path"])
    severity_value = train_data_config["severity"]
    severity = None if str(severity_value).lower() == "all" else severity_value
    dataset = DysarthricDataset(
        manifest_path=project_path(manifest_path),
        embeddings_dir=project_path(paths["embeddings_dir"]) / "distorted",
        severity=severity,
        train_config_path=args.train_config,
        layer=train_data_config["layer"],
    )
    if len(dataset) == 0:
        raise ValueError(f"No dataset rows found for severity={severity_value!r}")

    split_config = data_config["split"]
    ratios = {
        "train": float(split_config["train_ratio"]),
        "val": float(split_config["val_ratio"]),
        "test": float(split_config["test_ratio"]),
    }
    utterance_ids = dataset.manifest["utterance_id"].astype(str).unique().tolist()
    splits, wrote_splits = load_or_create_splits(
        utterance_ids=utterance_ids,
        ratios=ratios,
        seed=int(split_config["seed"]),
        output_path=project_path(splits_path),
    )
    for name in ("unfreeze_top_layers", "eval_every", "checkpoint_every", "max_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.init_checkpoint and (args.resume_checkpoint or runtime["resume_checkpoint"]):
        raise ValueError("Use either --init-checkpoint or --resume-checkpoint, not both")
    encoder = None
    loader_dataset, loader_collate = dataset, collate_fn
    if args.unfreeze_top_layers:
        from src.training.partial_encoder import (
            EncoderInputDataset, PartialWav2Vec2, collate_encoder_inputs,
        )

        encoder = PartialWav2Vec2(dataset.layer, args.unfreeze_top_layers, device)
        loader_dataset, loader_collate = EncoderInputDataset(dataset), collate_encoder_inputs
    subsets = {
        name: Subset(loader_dataset, row_indices_for_split(dataset.manifest, splits[name]))
        for name in SPLIT_NAMES
    }

    num_workers = (
        int(runtime["num_workers"])
        if args.num_workers is None
        else args.num_workers
    )
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    loader_options = {
        "batch_size": int(optim_config["batch_size"]),
        "num_workers": num_workers,
        "collate_fn": loader_collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        subsets["train"], shuffle=True, generator=generator, **loader_options
    )
    val_loader = DataLoader(subsets["val"], shuffle=False, **loader_options)
    test_loader = DataLoader(subsets["test"], shuffle=False, **loader_options)

    mapper = build_mapper_from_config(model_config).to(device)
    if args.init_checkpoint:
        initial = torch.load(project_path(args.init_checkpoint), map_location=device, weights_only=True)
        mapper.load_state_dict(initial["model_state_dict"])
        if encoder is not None and "wav2vec2_top_layers_state_dict" in initial:
            if initial.get("unfreeze_top_layers") != args.unfreeze_top_layers:
                raise ValueError("--init-checkpoint was saved with a different --unfreeze-top-layers")
            encoder.load_top_state_dict(initial["wav2vec2_top_layers_state_dict"])
        del initial
    optimizer = build_optimizer(mapper, optim_config)
    if encoder is not None:
        mapper_parameters = sum(p.numel() for p in mapper.parameters())
        encoder_parameters = sum(p.numel() for p in encoder.trainable_parameters())
        print(f"Unfrozen wav2vec2 blocks: {encoder_parameters:,} parameters "
              f"(mapper {mapper_parameters:,})")
        optimizer.add_param_group({"params": encoder.trainable_parameters(), "lr": args.wav2vec_lr})
    scheduler_name = optim_config["lr_scheduler"]
    scheduler = build_scheduler(optimizer, scheduler_name, num_epochs)
    reconstruction_weight = float(train_config["loss"]["reconstruction_weight"])
    log_interval = int(logging_config["log_interval"])
    save_interval = int(logging_config["save_every_n_epochs"])

    print_effective_config({
        "device": device,
        "severity": severity_value,
        "manifest_path": manifest_path,
        "splits_path": splits_path,
        "wav2vec2_layer": dataset.layer,
        "mapper_architecture": model_config["mapper"]["architecture"],
        "split_ratios": ratios,
        "split_seed": split_config["seed"],
        "batch_size": loader_options["batch_size"],
        "learning_rate": optim_config["learning_rate"],
        "weight_decay": optim_config["weight_decay"],
        "optimizer": optim_config["optimizer"],
        "lr_scheduler": scheduler_name,
        "reconstruction_weight": reconstruction_weight,
        "num_epochs": num_epochs,
        "num_workers": num_workers,
        "log_interval": log_interval,
        "save_every_n_epochs": save_interval,
        "unfreeze_top_layers": args.unfreeze_top_layers,
        "wav2vec_lr": args.wav2vec_lr if encoder is not None else None,
        "wav2vec_clip_norm": args.wav2vec_clip_norm if encoder is not None else None,
        "mapper_clip_norm": args.mapper_clip_norm,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "max_steps": args.max_steps,
        "resume_checkpoint": args.resume_checkpoint or runtime["resume_checkpoint"],
        "checkpoint_dir": checkpoint_dir,
        "log_dir": log_dir,
    })
    print(f"\nSplit assignments: {'wrote' if wrote_splits else 'reused'} {splits_path}")
    for name in SPLIT_NAMES:
        print(
            f"  {name}: {len(splits[name])} utterance(s), "
            f"{len(subsets[name])} row(s)"
        )
    mapper.count_parameters()

    best_path = (
        project_path(args.best_checkpoint)
        if args.best_checkpoint
        else checkpoint_dir / "best_mapper.pt"
    )
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    start_epoch = 1
    best_epoch = 0
    resume_checkpoint = args.resume_checkpoint or runtime["resume_checkpoint"]
    if resume_checkpoint:
        resume_path = project_path(resume_checkpoint)
        checkpoint = load_checkpoint(resume_path, mapper, optimizer, scheduler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint["val_loss"]))
        best_epoch = start_epoch - 1  # patience restarts from the resume point
        train_losses = list(checkpoint.get("train_losses", []))
        val_losses = list(checkpoint.get("val_losses", []))
        if encoder is not None:
            if checkpoint.get("unfreeze_top_layers") != args.unfreeze_top_layers:
                raise ValueError("--resume-checkpoint was saved with a different --unfreeze-top-layers")
            encoder.load_top_state_dict(checkpoint["wav2vec2_top_layers_state_dict"])
        resumed_step = int(checkpoint.get("global_step", 0))
        if not best_path.exists():
            torch.save(checkpoint, best_path)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    extras = None
    if (encoder is not None or args.mapper_clip_norm > 0 or args.eval_every
            or args.checkpoint_every or args.max_steps):
        def current_payload():
            return checkpoint_payload(
                epoch, mapper, optimizer, scheduler, val_losses[-1] if val_losses else None,
                best_val_loss, train_config, data_config, model_config, train_losses, val_losses,
                extra=payload_extra(),
            )

        extras = StepExtras(mapper=mapper, encoder=encoder, dataset=dataset, device=device,
                            args=args, out_dir=checkpoint_dir, payload=current_payload)
        if resume_checkpoint:
            extras.step = resumed_step

    def payload_extra():
        """Extra checkpoint keys; None (unchanged payload) when no new option is active."""
        if extras is None:
            return None
        extra = {"global_step": extras.step}
        if encoder is not None:
            extra.update(unfreeze_top_layers=args.unfreeze_top_layers,
                         wav2vec2_top_layers_state_dict=encoder.top_state_dict())
        return extra

    if extras is not None and args.eval_every:
        extras.evaluate()

    epoch_bar = tqdm(range(start_epoch, num_epochs + 1), desc="Training")
    for epoch in epoch_bar:
        train_loss = run_epoch(
            mapper, train_loader, device, reconstruction_weight,
            optimizer=optimizer, desc=f"Epoch {epoch}/{num_epochs} [train]", extras=extras,
        )
        val_loss = run_epoch(
            mapper, val_loader, device, reconstruction_weight,
            desc=f"Epoch {epoch}/{num_epochs} [val]", extras=extras,
        )
        step_scheduler(scheduler, scheduler_name, val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        best_so_far = min(best_val_loss, val_loss)
        epoch_bar.set_description(
            f"Epoch {epoch}/{num_epochs} | train={train_loss:.3f} "
            f"val={val_loss:.3f} best={best_so_far:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                checkpoint_payload(
                    epoch, mapper, optimizer, scheduler, val_loss, best_val_loss,
                    train_config, data_config, model_config, train_losses, val_losses,
                    extra=payload_extra(),
                ),
                best_path,
            )

        if save_interval > 0 and epoch % save_interval == 0:
            prefix = best_path.stem if args.best_checkpoint else "mapper"
            periodic_path = checkpoint_dir / f"{prefix}_epoch_{epoch:03d}.pt"
            torch.save(
                checkpoint_payload(
                    epoch, mapper, optimizer, scheduler, val_loss, best_val_loss,
                    train_config, data_config, model_config, train_losses, val_losses,
                    extra=payload_extra(),
                ),
                periodic_path,
            )

        if epoch == start_epoch or epoch % max(log_interval, 1) == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | best_val_loss={best_val_loss:.4f}"
            )

        if args.early_stop_patience and epoch - best_epoch >= args.early_stop_patience:
            print(
                f"Early stop at epoch {epoch}: no val improvement since epoch {best_epoch}"
            )
            break

        if extras is not None and extras.done:
            print(f"Stopped after --max-steps {args.max_steps} optimizer steps")
            break

    if not train_losses:
        raise ValueError(
            f"No epochs ran: resume epoch {start_epoch} exceeds configured num_epochs {num_epochs}"
        )
    loss_curve_path = (
        project_path(args.loss_curve)
        if args.loss_curve
        else log_dir / logging_config.get("loss_curve_filename", "loss_curve.png")
    )
    plot_loss_curve(train_losses, val_losses, loss_curve_path)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    mapper.load_state_dict(best_checkpoint["model_state_dict"])
    if encoder is not None:
        encoder.load_top_state_dict(best_checkpoint["wav2vec2_top_layers_state_dict"])
    test_loss = run_epoch(mapper, test_loader, device, reconstruction_weight, extras=extras)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Held-out test loss: {test_loss:.4f}")
    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
