"""Helpers for loading project configuration from one well-known location."""

from pathlib import Path

import os

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "train.yaml"
DEFAULT_DATA_CONFIG_PATH = REPO_ROOT / "configs" / "data.yaml"
DEFAULT_MODEL_CONFIG_PATH = REPO_ROOT / "configs" / "model.yaml"


def load_yaml_config(path):
    """Load a YAML mapping and fail clearly for missing or malformed configs."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    return config


def cv_data_root(data_config_path=DEFAULT_DATA_CONFIG_PATH):
    """Root of the derived Common Voice audio set: $DATA_ROOT, else data.yaml."""
    value = os.environ.get("DATA_ROOT")
    if not value:
        value = load_yaml_config(data_config_path)["paths"]["cv27_data_root"]
    return Path(value).expanduser()


def resolve_data_path(path, root=None):
    """Resolve a path stored relative to the data root; absolute paths are kept."""
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (cv_data_root() if root is None else Path(root)) / value


def get_selected_layer(train_config_path=DEFAULT_TRAIN_CONFIG_PATH):
    """Return the sole configured wav2vec2 layer used by training/inference."""
    config = load_yaml_config(train_config_path)
    try:
        layer = int(config["data"]["layer"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Expected integer data.layer in training config: {train_config_path}"
        ) from exc
    if not 0 <= layer <= 24:
        raise ValueError(f"data.layer must be between 0 and 24, got {layer}")
    return layer
