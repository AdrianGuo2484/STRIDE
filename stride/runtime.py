from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)


def save_json(payload: object, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def save_checkpoint(payload: dict[str, object], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)


def class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"training split is missing classes; class counts={counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def move_longitudinal_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    result = dict(batch)
    for key in ("x_t", "x_next", "lesion_t", "lesion_next", "modality_t", "modality_next", "delta_weeks"):
        value = result[key]
        assert isinstance(value, torch.Tensor)
        result[key] = value.to(device=device, dtype=torch.float32, non_blocking=True)
    label = result["label"]
    assert isinstance(label, torch.Tensor)
    result["label"] = label.to(device=device, dtype=torch.long, non_blocking=True)
    return result
