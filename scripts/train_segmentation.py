from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from stride.data import SegmentationDataset
from stride.losses import segmentation_loss
from stride.metrics import dice_score, hd95
from stride.model import AWHTSegmenter
from stride.runtime import load_yaml, resolve_device, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the independent AWHT BraTS lesion segmenter.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def run_epoch(
    model: AWHTSegmenter,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
    spacing: tuple[float, float, float],
    region_names: list[str],
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_bce = 0.0
    total_dice_loss = 0.0
    dice_values: dict[str, list[float]] = {name: [] for name in region_names}
    hd95_values: dict[str, list[float]] = {name: [] for name in region_names}
    count = 0
    for batch in loader:
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["mask"].to(device=device, dtype=torch.float32, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(images)
                logits = output["logits"]
                assert isinstance(logits, torch.Tensor)
                loss, components = segmentation_loss(logits, targets)
            if training:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_bce += float(components["bce"].detach()) * batch_size
        total_dice_loss += float(components["dice_loss"].detach()) * batch_size
        count += batch_size
        predictions = (torch.sigmoid(logits.detach()) >= 0.5).cpu().numpy()
        truth = targets.detach().cpu().numpy() >= 0.5
        for index in range(batch_size):
            for region_index, region_name in enumerate(region_names):
                dice_values[region_name].append(
                    dice_score(predictions[index, region_index], truth[index, region_index])
                )
                if not training:
                    hd95_values[region_name].append(
                        hd95(predictions[index, region_index], truth[index, region_index], spacing)
                    )
    result = {
        "loss": total_loss / max(count, 1),
        "bce": total_bce / max(count, 1),
        "dice_loss": total_dice_loss / max(count, 1),
    }
    for name in region_names:
        finite_hd95 = [value for value in hd95_values[name] if math.isfinite(value)]
        result[f"dice_{name.lower()}"] = float(np.mean(dice_values[name])) if dice_values[name] else math.nan
        result[f"hd95_{name.lower()}_mm"] = float(np.mean(finite_hd95)) if finite_hd95 else math.nan
    result["mean_dice"] = float(np.mean([result[f"dice_{name.lower()}"] for name in region_names]))
    return result


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "auto")))
    data_config = config["data"]
    train_config = config["train"]
    manifest_path = repository_path(data_config["manifest"])
    frame = pd.read_csv(manifest_path)
    n_splits = int(data_config.get("n_splits", 5))
    if not 0 <= args.fold < n_splits:
        raise ValueError(f"fold must be in [0,{n_splits - 1}]")
    group_column = str(data_config.get("patient_column", "patient_id"))
    splitter = GroupKFold(n_splits=n_splits)
    partitions = list(splitter.split(frame, groups=frame[group_column].astype(str)))
    train_indices, validation_indices = partitions[args.fold]
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    validation_frame = frame.iloc[validation_indices].reset_index(drop=True)
    if set(train_frame[group_column]) & set(validation_frame[group_column]):
        raise RuntimeError("patient leakage detected in segmentation split")

    dataset_kwargs = {
        "manifest_dir": manifest_path.parent,
        "modality_columns": data_config["modality_columns"],
        "mask_column": str(data_config["mask_column"]),
        "region_labels": data_config["region_labels"],
        "target_shape": data_config.get("target_shape", [128, 128, 128]),
        "intensity_scale_range": data_config.get("intensity_scale_range", [0.9, 1.1]),
        "intensity_shift_range": data_config.get("intensity_shift_range", [-0.1, 0.1]),
        "intensity_noise_std": float(data_config.get("intensity_noise_std", 0.03)),
    }
    train_dataset = SegmentationDataset(train_frame, augment=bool(data_config.get("augment", True)), **dataset_kwargs)
    validation_dataset = SegmentationDataset(validation_frame, augment=False, **dataset_kwargs)
    workers = int(train_config.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(train_config["batch_size"]),
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_kwargs)
    model = AWHTSegmenter(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = repository_path(train_config["output_dir"]) / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    start_epoch = 1
    best_dice = -math.inf
    history: list[dict[str, object]] = []
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if "scaler_state" in payload:
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_dice = float(payload.get("best_dice", -math.inf))
        history = list(payload.get("history", []))
    grad_clip_value = float(train_config.get("grad_clip", 0.0))
    grad_clip = grad_clip_value if grad_clip_value > 0 else None
    spacing = tuple(float(value) for value in data_config.get("target_spacing", [1.0, 1.0, 1.0]))
    region_names = [str(value) for value in data_config.get("region_names", ["WT", "TC", "ET"])]
    print(f"device={device} fold={args.fold} train={len(train_frame)} val={len(validation_frame)}")
    for epoch in range(start_epoch, int(train_config["epochs"]) + 1):
        train_metrics = run_epoch(
            model, train_loader, device=device, optimizer=optimizer, grad_clip=grad_clip,
            spacing=spacing, region_names=region_names, scaler=scaler, amp_enabled=amp_enabled
        )
        validation_metrics = run_epoch(
            model, validation_loader, device=device, optimizer=None, grad_clip=None,
            spacing=spacing, region_names=region_names, scaler=scaler, amp_enabled=amp_enabled
        )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        save_json(history, output_dir / "history.json")
        checkpoint = {
            "epoch": epoch,
            "fold": args.fold,
            "best_dice": max(best_dice, validation_metrics["mean_dice"]),
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
        }
        if validation_metrics["mean_dice"] > best_dice:
            best_dice = validation_metrics["mean_dice"]
            checkpoint["best_dice"] = best_dice
            save_checkpoint(checkpoint, best_path)
            save_json(validation_metrics, output_dir / "best_validation_metrics.json")
        save_checkpoint(checkpoint, last_path)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_mean_dice={validation_metrics['mean_dice']:.4f} "
            f"val_wt_dice={validation_metrics['dice_wt']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
