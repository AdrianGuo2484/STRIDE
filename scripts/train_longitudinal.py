from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from stride.data import LongitudinalPairDataset
from stride.losses import longitudinal_loss
from stride.metrics import classification_metrics
from stride.model import STRIDE, load_longitudinal_pretraining
from stride.runtime import class_weights, load_yaml, move_longitudinal_batch, resolve_device, save_checkpoint, save_json, set_seed

BURDENKO_N_SPLITS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train STRIDE on LUMIERE or Burdenko.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=None, help="Burdenko held-out test fold in [0, 9].")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt in the output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs, useful for smoke runs.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override train.output_dir.")
    parser.add_argument("--manifest", type=Path, default=None, help="Override data.manifest.")
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help="Override the LUMIERE checkpoint used for Burdenko fine-tuning.",
    )
    return parser.parse_args()


def repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def split_lumiere(
    frame: pd.DataFrame,
    split_column: str,
    patient_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    required = {split_column, patient_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"LUMIERE manifest is missing columns: {sorted(missing)}")
    split = frame[split_column].astype(str).str.lower()
    train = frame[split == "train"].reset_index(drop=True)
    validation = frame[split.isin(["val", "validation"])].reset_index(drop=True)
    test = frame[split == "test"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise ValueError("LUMIERE manifest must contain non-empty train and val partitions")
    partitions = [train, validation] if test.empty else [train, validation, test]
    patient_sets = [set(part[patient_column].astype(str)) for part in partitions]
    for left in range(len(patient_sets)):
        for right in range(left + 1, len(patient_sets)):
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise RuntimeError(
                    "patient leakage detected in predefined LUMIERE split; "
                    f"partitions {left} and {right} share {len(overlap)} patients"
                )
    return train, validation, None if test.empty else test


def validate_modality_contract(
    data_config: dict[str, object],
    model_config: dict[str, object],
) -> tuple[str, ...]:
    columns = data_config.get("modality_columns")
    names = data_config.get("modality_names")
    if not isinstance(columns, list) or not isinstance(names, list):
        raise TypeError("data.modality_columns and data.modality_names must be lists")
    normalized_names = tuple(str(name).strip().lower() for name in names)
    expected = int(model_config["num_modalities"])
    if len(columns) != expected or len(normalized_names) != expected:
        raise ValueError(
            "modality contract does not match model.num_modalities: "
            f"columns={len(columns)} names={len(normalized_names)} model={expected}"
        )
    if len(set(normalized_names)) != expected:
        raise ValueError(f"data.modality_names must be unique, got {normalized_names}")
    return normalized_names


def validate_pretraining_contract(
    payload: dict[str, object],
    downstream_modalities: tuple[str, ...],
) -> None:
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise KeyError("pretraining checkpoint does not contain its resolved config")
    if str(checkpoint_config.get("task", "")).lower() != "lumiere":
        raise ValueError("pretraining checkpoint must come from the LUMIERE task")
    checkpoint_data = checkpoint_config.get("data")
    checkpoint_model = checkpoint_config.get("model")
    if not isinstance(checkpoint_data, dict) or not isinstance(checkpoint_model, dict):
        raise TypeError("pretraining checkpoint has an invalid data/model config")
    pretrained_modalities = validate_modality_contract(checkpoint_data, checkpoint_model)
    if pretrained_modalities != downstream_modalities:
        raise ValueError(
            "MRI modality order differs between pretraining and fine-tuning: "
            f"pretrained={pretrained_modalities}, downstream={downstream_modalities}"
        )


def encode_class_labels(
    frame: pd.DataFrame,
    *,
    label_column: str,
    class_names: list[str],
) -> pd.DataFrame:
    if label_column not in frame:
        raise KeyError(f"manifest is missing label column: {label_column}")
    name_to_index = {name.strip().casefold(): index for index, name in enumerate(class_names)}
    encoded: list[int] = []
    invalid: list[str] = []
    for value in frame[label_column]:
        label: int | None = None
        if not pd.isna(value):
            try:
                numeric = float(value)
                if numeric.is_integer():
                    label = int(numeric)
            except (TypeError, ValueError):
                label = name_to_index.get(str(value).strip().casefold())
        if label is None or not 0 <= label < len(class_names):
            invalid.append(str(value))
        else:
            encoded.append(label)
    if invalid:
        examples = sorted(set(invalid))[:5]
        raise ValueError(
            f"{label_column} contains labels outside {class_names}: examples={examples}"
        )
    result = frame.copy()
    result[label_column] = np.asarray(encoded, dtype=np.int64)
    return result


def split_burdenko(
    frame: pd.DataFrame,
    *,
    label_column: str,
    patient_column: str,
    n_splits: int,
    seed: int,
    test_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    required = {label_column, patient_column}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Burdenko manifest is missing columns: {sorted(missing)}")
    if not 0 <= test_fold < n_splits:
        raise ValueError(f"fold must be in [0,{n_splits - 1}], got {test_fold}")
    patient_labels = frame[[patient_column, label_column]].copy()
    patient_labels[patient_column] = patient_labels[patient_column].astype(str)
    inconsistent = patient_labels.groupby(patient_column)[label_column].nunique()
    if (inconsistent > 1).any():
        examples = inconsistent[inconsistent > 1].index.tolist()[:5]
        raise ValueError(f"patients have inconsistent Burdenko labels: {examples}")
    patient_labels = patient_labels.drop_duplicates(patient_column).reset_index(drop=True)
    patient_class_counts = patient_labels[label_column].value_counts().sort_index()
    if (patient_class_counts < n_splits).any():
        raise ValueError(
            f"each class needs at least {n_splits} patients for stratified CV; "
            f"patient counts={patient_class_counts.to_dict()}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    patient_to_fold: dict[str, int] = {}
    for fold, (_, held_out) in enumerate(
        splitter.split(patient_labels, patient_labels[label_column].astype(int))
    ):
        for patient_id in patient_labels.iloc[held_out][patient_column]:
            patient_to_fold[str(patient_id)] = fold
    fold_assignment = frame[patient_column].astype(str).map(patient_to_fold).to_numpy(dtype=np.int64)
    validation_fold = (test_fold + 1) % n_splits
    train = frame[(fold_assignment != test_fold) & (fold_assignment != validation_fold)].reset_index(drop=True)
    validation = frame[fold_assignment == validation_fold].reset_index(drop=True)
    test = frame[fold_assignment == test_fold].reset_index(drop=True)
    expected_classes = set(patient_labels[label_column].astype(int))
    for partition_name, partition in (("train", train), ("validation", validation), ("test", test)):
        missing_classes = expected_classes.difference(partition[label_column].astype(int))
        if missing_classes:
            raise RuntimeError(f"{partition_name} partition is missing classes: {sorted(missing_classes)}")
    patient_sets = [set(part[patient_column].astype(str)) for part in (train, validation, test)]
    if patient_sets[0] & patient_sets[1] or patient_sets[0] & patient_sets[2] or patient_sets[1] & patient_sets[2]:
        raise RuntimeError("patient leakage detected across train/validation/test")
    return train, validation, test, validation_fold


def make_dataset(frame: pd.DataFrame, config: dict[str, object], manifest_path: Path, *, augment: bool) -> LongitudinalPairDataset:
    return LongitudinalPairDataset(
        frame,
        manifest_dir=manifest_path.parent,
        modality_columns=config["modality_columns"],
        lesion_prior_columns=config["lesion_prior_columns"],
        label_column=str(config["label_column"]),
        target_shape=config.get("target_shape", [128, 128, 128]),
        allow_missing_modalities=bool(config.get("allow_missing_modalities", False)),
        required_modalities=config.get("required_modalities"),
        probability_mode=str(config.get("probability_mode", "foreground_max")),
        probability_channel=int(config.get("probability_channel", 0)),
        augment=augment,
        intensity_scale_range=config.get("intensity_scale_range", [0.9, 1.1]),
        intensity_shift_range=config.get("intensity_shift_range", [-0.1, 0.1]),
        intensity_noise_std=float(config.get("intensity_noise_std", 0.03)),
    )


def make_loader(
    dataset: LongitudinalPairDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        generator=generator,
    )


def run_epoch(
    model: STRIDE,
    loader: DataLoader,
    *,
    task: str,
    num_classes: int,
    class_names: list[str],
    device: torch.device,
    weights: torch.Tensor,
    transition_weight: float,
    direction_weight: float,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    training = optimizer is not None
    model.train(training)
    if training:
        model.enforce_frozen_module_eval()
    totals = {"loss": 0.0, "classification": 0.0, "transition": 0.0, "direction": 0.0}
    count = 0
    all_targets: list[int] = []
    all_probabilities: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    for raw_batch in loader:
        batch = move_longitudinal_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(
                    batch["x_t"],
                    batch["x_next"],
                    batch["lesion_t"],
                    batch["lesion_next"],
                    batch["modality_t"],
                    batch["modality_next"],
                    batch["delta_weeks"],
                    task=task,
                )
                loss, components = longitudinal_loss(
                    output,
                    batch["label"],
                    class_weights=weights,
                    transition_weight=transition_weight,
                    direction_weight=direction_weight,
                )
            if training:
                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        logits = output["logits"]
        assert isinstance(logits, torch.Tensor)
        probabilities = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        targets = batch["label"].detach().cpu().numpy()
        batch_size = len(targets)
        totals["loss"] += float(loss.detach()) * batch_size
        for name, value in components.items():
            totals[name] += float(value.detach()) * batch_size
        count += batch_size
        all_targets.extend(targets.tolist())
        all_probabilities.extend(probabilities)
        for index in range(batch_size):
            row: dict[str, object] = {
                "pair_id": raw_batch["pair_id"][index],
                "patient_id": raw_batch["patient_id"][index],
                "target": int(targets[index]),
                "prediction": int(probabilities[index].argmax()),
            }
            row["target_name"] = class_names[int(targets[index])]
            row["prediction_name"] = class_names[int(probabilities[index].argmax())]
            row.update(
                {
                    f"probability_{class_name}": float(probabilities[index, class_index])
                    for class_index, class_name in enumerate(class_names)
                }
            )
            rows.append(row)
    if count == 0:
        raise RuntimeError("empty data loader")
    probability_array = np.asarray(all_probabilities, dtype=np.float64)
    metrics = classification_metrics(
        all_targets,
        probability_array,
        num_classes=num_classes,
        class_names=class_names,
    )
    metrics.update({name: value / count for name, value in totals.items()})
    return metrics, pd.DataFrame(rows)


def metric_value(metrics: dict[str, object], name: str) -> float:
    value = metrics.get(name, math.nan)
    return float(value) if value is not None else math.nan


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    task = str(config["task"]).lower()
    if task not in {"lumiere", "burdenko"}:
        raise ValueError("config task must be lumiere or burdenko")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    data_config = config["data"]
    train_config = config["train"]
    model_config = config["model"]
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        train_config["epochs"] = args.epochs
    if args.output_dir is not None:
        train_config["output_dir"] = str(args.output_dir)
    if args.manifest is not None:
        data_config["manifest"] = str(args.manifest)
    if args.pretrained_checkpoint is not None:
        train_config["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
    modality_names = validate_modality_contract(data_config, model_config)
    manifest_path = repository_path(data_config["manifest"])
    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError(f"empty manifest: {manifest_path}")
    num_classes = int(
        model_config["num_lumiere_classes" if task == "lumiere" else "num_burdenko_classes"]
    )
    class_names = [str(value) for value in data_config.get("class_names", [])]
    if len(class_names) != num_classes:
        raise ValueError(
            f"data.class_names must contain {num_classes} entries for task={task}; "
            f"got {class_names}"
        )
    frame = encode_class_labels(
        frame,
        label_column=str(data_config["label_column"]),
        class_names=class_names,
    )

    if task == "lumiere":
        train_frame, validation_frame, test_frame = split_lumiere(
            frame,
            str(data_config.get("split_column", "split")),
            str(data_config.get("patient_column", "patient_id")),
        )
        fold = None
        validation_fold = None
    else:
        configured_splits = int(data_config.get("n_splits", BURDENKO_N_SPLITS))
        if configured_splits != BURDENKO_N_SPLITS:
            raise ValueError(
                "the Burdenko protocol requires exactly 10 folds; "
                f"got data.n_splits={configured_splits}"
            )
        fold = int(args.fold if args.fold is not None else train_config.get("fold", 0))
        train_frame, validation_frame, test_frame, validation_fold = split_burdenko(
            frame,
            label_column=str(data_config["label_column"]),
            patient_column=str(data_config.get("patient_column", "patient_id")),
            n_splits=BURDENKO_N_SPLITS,
            seed=int(data_config.get("split_seed", seed)),
            test_fold=fold,
        )

    output_root = repository_path(train_config["output_dir"])
    output_dir = output_root if fold is None else output_root / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")
    train_dataset = make_dataset(train_frame, data_config, manifest_path, augment=bool(data_config.get("augment", True)))
    validation_dataset = make_dataset(validation_frame, data_config, manifest_path, augment=False)
    test_dataset = None if test_frame is None else make_dataset(test_frame, data_config, manifest_path, augment=False)
    batch_size = int(train_config["batch_size"])
    workers = int(train_config.get("num_workers", 0))
    train_loader = make_loader(train_dataset, batch_size=batch_size, workers=workers, shuffle=True, seed=seed)
    validation_loader = make_loader(validation_dataset, batch_size=batch_size, workers=workers, shuffle=False, seed=seed)
    test_loader = None if test_dataset is None else make_loader(test_dataset, batch_size=batch_size, workers=workers, shuffle=False, seed=seed)

    model = STRIDE(**model_config).to(device)
    pretrained_path_value = str(train_config.get("pretrained_checkpoint", "")).strip()
    pretraining_loaded = False
    if task == "burdenko" and pretrained_path_value:
        pretrained_path = repository_path(pretrained_path_value)
        payload = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        validate_pretraining_contract(payload, modality_names)
        missing, unexpected = load_longitudinal_pretraining(model, payload)
        expected_missing = {
            "lumiere_head.weight",
            "lumiere_head.bias",
            "burdenko_head.weight",
            "burdenko_head.bias",
        }
        if set(missing) != expected_missing or unexpected:
            raise RuntimeError(
                "incomplete longitudinal pretraining transfer: "
                f"missing={missing}, unexpected={unexpected}"
            )
        print(f"loaded longitudinal pretraining: {pretrained_path}")
        print(f"transfer missing={missing} unexpected={unexpected}")
        pretraining_loaded = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    amp_enabled = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    best_value = -math.inf
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    history: list[dict[str, object]] = []
    if args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if "scaler_state" in payload:
            scaler.load_state_dict(payload["scaler_state"])
        start_epoch = int(payload["epoch"]) + 1
        best_value = float(payload.get("best_value", -math.inf))
        history = list(payload.get("history", []))
        print(f"resumed from {last_path} at epoch {start_epoch}")

    weights = class_weights(train_frame[str(data_config["label_column"])].to_numpy(), num_classes, device)
    selection_metric = str(train_config["selection_metric"])
    transition_weight = float(train_config["transition_weight"])
    direction_weight = float(train_config["direction_weight"])
    grad_clip_value = float(train_config.get("grad_clip", 0.0))
    grad_clip = grad_clip_value if grad_clip_value > 0 else None
    freeze_epochs = (
        int(train_config.get("freeze_transferred_epochs", 0))
        if task == "burdenko" and pretraining_loaded
        else 0
    )
    checkpoint_warmup_epochs = int(train_config.get("checkpoint_warmup_epochs", 0))
    exclude_perfect_auc = bool(train_config.get("exclude_perfect_validation_auc", False))
    print(
        f"task={task} device={device} train={len(train_frame)} val={len(validation_frame)} "
        f"test={0 if test_frame is None else len(test_frame)} fold={fold} val_fold={validation_fold}"
    )
    for epoch in range(start_epoch, int(train_config["epochs"]) + 1):
        model.set_transferred_modules_trainable(not (freeze_epochs > 0 and epoch <= freeze_epochs))
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            task=task,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            weights=weights,
            transition_weight=transition_weight,
            direction_weight=direction_weight,
            optimizer=optimizer,
            grad_clip=grad_clip,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        validation_metrics, validation_predictions = run_epoch(
            model,
            validation_loader,
            task=task,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            weights=weights,
            transition_weight=transition_weight,
            direction_weight=direction_weight,
            optimizer=None,
            grad_clip=None,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(record)
        save_json(history, output_dir / "history.json")
        current = metric_value(validation_metrics, selection_metric)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} {selection_metric}={current:.4f}"
        )
        checkpoint = {
            "epoch": epoch,
            "task": task,
            "fold": fold,
            "validation_fold": validation_fold,
            "best_value": best_value,
            "config": config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "history": history,
        }
        checkpoint_eligible = epoch > checkpoint_warmup_epochs and math.isfinite(current)
        if exclude_perfect_auc and math.isclose(current, 1.0, rel_tol=0.0, abs_tol=1e-12):
            checkpoint_eligible = False
        if checkpoint_eligible and current > best_value:
            best_value = current
            checkpoint["best_value"] = best_value
            save_checkpoint(checkpoint, best_path)
            validation_predictions.to_csv(output_dir / "best_validation_predictions.csv", index=False)
        save_checkpoint(checkpoint, last_path)

    if not best_path.exists():
        raise RuntimeError(f"no finite validation {selection_metric}; no best checkpoint was produced")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    final_loader = test_loader if test_loader is not None else validation_loader
    final_name = "test" if test_loader is not None else "validation"
    final_metrics, final_predictions = run_epoch(
        model,
        final_loader,
        task=task,
        num_classes=num_classes,
        class_names=class_names,
        device=device,
        weights=weights,
        transition_weight=transition_weight,
        direction_weight=direction_weight,
        optimizer=None,
        grad_clip=None,
        scaler=scaler,
        amp_enabled=amp_enabled,
    )
    final_metrics["best_epoch"] = int(best["epoch"])
    final_metrics["fold"] = fold
    final_metrics["validation_fold"] = validation_fold
    save_json(final_metrics, output_dir / f"{final_name}_metrics.json")
    final_predictions.to_csv(output_dir / f"{final_name}_predictions.csv", index=False)
    print(f"saved {final_name} results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
