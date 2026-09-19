"""Train the configured wav2vec2-to-mel Bi-LSTM mapper."""

import argparse
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


def run_epoch(model, loader, device, reconstruction_weight, optimizer=None, desc="epoch"):
    """Run one pass over a loader; train only when an optimizer is supplied."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(is_train):
        for embeddings, mels, lengths in progress_bar:
            embeddings = embeddings.to(device)
            mels = mels.to(device)
            lengths = lengths.to(device)

            predictions = model(embeddings, lengths)
            loss = reconstruction_weight * masked_l1_loss(predictions, mels, lengths)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = embeddings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

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
    train_config, data_config, model_config, train_losses, val_losses,
):
    return {
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
    manifest_path = train_data_config.get("manifest_path", paths["manifest_path"])
    splits_path = train_data_config.get("splits_path", paths["splits_path"])
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
    subsets = {
        name: Subset(dataset, row_indices_for_split(dataset.manifest, splits[name]))
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
        "collate_fn": collate_fn,
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
    optimizer = build_optimizer(mapper, optim_config)
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
    resume_checkpoint = args.resume_checkpoint or runtime["resume_checkpoint"]
    if resume_checkpoint:
        resume_path = project_path(resume_checkpoint)
        checkpoint = load_checkpoint(resume_path, mapper, optimizer, scheduler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", checkpoint["val_loss"]))
        train_losses = list(checkpoint.get("train_losses", []))
        val_losses = list(checkpoint.get("val_losses", []))
        if not best_path.exists():
            torch.save(checkpoint, best_path)
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    epoch_bar = tqdm(range(start_epoch, num_epochs + 1), desc="Training")
    for epoch in epoch_bar:
        train_loss = run_epoch(
            mapper, train_loader, device, reconstruction_weight,
            optimizer=optimizer, desc=f"Epoch {epoch}/{num_epochs} [train]",
        )
        val_loss = run_epoch(
            mapper, val_loader, device, reconstruction_weight,
            desc=f"Epoch {epoch}/{num_epochs} [val]",
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
            torch.save(
                checkpoint_payload(
                    epoch, mapper, optimizer, scheduler, val_loss, best_val_loss,
                    train_config, data_config, model_config, train_losses, val_losses,
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
                ),
                periodic_path,
            )

        if epoch == start_epoch or epoch % max(log_interval, 1) == 0 or epoch == num_epochs:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | best_val_loss={best_val_loss:.4f}"
            )

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
    test_loss = run_epoch(mapper, test_loader, device, reconstruction_weight)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Held-out test loss: {test_loss:.4f}")
    print(f"Best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()
