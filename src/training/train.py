# train.py
#
# Purpose:
#   Main training entry point that ties together dataset.py, the models in
#   src/models/, and the hyperparameters in configs/train.yaml to fit the
#   dysarthric-to-clean reconstruction model.
#
# Expected responsibilities:
#   - Parse configs/{data,model,train}.yaml
#   - Build DataLoaders (dataset.py), model (encoder/mapper/vocoder)
#   - Run the training loop: forward pass, loss, backward, optimizer step
#   - Log metrics/checkpoints to results/logs/ and results/checkpoints/
#   - Support resuming from a checkpoint

import os
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

from src.models.mapper import MapperModel
from src.training.dataset import DysarthricDataset, collate_fn

CONFIG_PATH = os.path.join("configs", "train.yaml")
NUM_VAL_SAMPLES = 2
PRINT_EVERY_N_EPOCHS = 10

# Loss-curve plot palette (project dataviz reference: categorical slots 1/2).
COLOR_TRAIN = "#2a78d6"  # blue
COLOR_VAL = "#eb6834"  # orange
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference="auto"):
    """Auto-detect the best available device unless a specific one is requested."""
    if preference not in (None, "auto"):
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_l1_loss(pred, target, lengths):
    """L1 loss on the mel-spectrogram, ignoring padded timesteps.

    collate_fn pads variable-length sequences to a common length within a
    batch; `lengths` marks how much of each sample is real so padding never
    pollutes the loss.
    """
    max_len = pred.size(1)
    mel_dim = pred.size(-1)
    mask = (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]).float()
    mask = mask.unsqueeze(-1)  # (B, T, 1), broadcasts over the mel dimension

    diff = (pred - target).abs() * mask
    denom = mask.sum() * mel_dim
    return diff.sum() / denom.clamp(min=1)


def run_epoch(model, loader, device, optimizer=None):
    """Run one pass over `loader`; trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for embeddings, mels, lengths in loader:
            embeddings = embeddings.to(device)
            mels = mels.to(device)
            lengths = lengths.to(device)

            preds = model(embeddings, lengths)
            loss = masked_l1_loss(preds, mels, lengths)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = embeddings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved loss curve to: {output_path}")


def main():
    config = load_config()

    seed = config["runtime"]["seed"]
    set_seed(seed)

    device = get_device(config["runtime"].get("device", "auto"))
    print(f"Using device: {device}")

    severity = config["data"]["severity"]
    layer = config["data"]["layer"]
    batch_size = config["optim"]["batch_size"]
    num_epochs = config["optim"]["num_epochs"]
    learning_rate = config["optim"]["learning_rate"]
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    log_dir = config["logging"]["log_dir"]

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = DysarthricDataset(severity=severity, layer=layer)
    print(f"Loaded {len(dataset)} '{severity}' samples (wav2vec2 layer {layer}).")

    num_val = NUM_VAL_SAMPLES
    num_train = len(dataset) - num_val
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [num_train, num_val], generator=split_generator)
    print(f"Split: {num_train} train / {num_val} val")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MapperModel().to(device)
    model.count_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    checkpoint_path = os.path.join(checkpoint_dir, "best_mapper.pt")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )

        if epoch == 1 or epoch % PRINT_EVERY_N_EPOCHS == 0:
            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"best_val_loss={best_val_loss:.4f}"
            )

    plot_loss_curve(train_losses, val_losses, os.path.join(log_dir, "loss_curve.png"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
