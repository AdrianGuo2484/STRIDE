from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value) and str(value).strip() != ""


def _resolve_path(value: object, base_dir: Path) -> Path | None:
    if not _present(value):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base_dir / path


def _load_nifti(path: Path) -> np.ndarray:
    data = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D NIfTI volume, got {data.shape} from {path}")
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)


def _center_crop_or_pad(volume: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got {volume.shape}")
    result = np.zeros(shape, dtype=volume.dtype)
    source_slices = []
    target_slices = []
    for source_size, target_size in zip(volume.shape, shape):
        copy_size = min(source_size, target_size)
        source_start = (source_size - copy_size) // 2
        target_start = (target_size - copy_size) // 2
        source_slices.append(slice(source_start, source_start + copy_size))
        target_slices.append(slice(target_start, target_start + copy_size))
    result[tuple(target_slices)] = volume[tuple(source_slices)]
    return result


def _normalize_mri(volume: np.ndarray) -> np.ndarray:
    foreground = np.abs(volume) > 1e-6
    if not np.any(foreground):
        return np.zeros_like(volume, dtype=np.float32)
    values = volume[foreground]
    std = max(float(values.std()), 1e-6)
    normalized = (volume - float(values.mean())) / std
    normalized[~foreground] = 0.0
    return np.clip(normalized, -5.0, 5.0).astype(np.float32)


def load_mri(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return _normalize_mri(_center_crop_or_pad(_load_nifti(path), shape))


def load_lesion_prior(
    path: Path,
    shape: tuple[int, int, int],
    probability_mode: str = "foreground_max",
    probability_channel: int = 0,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        archive = np.load(path)
        if "abnormal_probability" in archive.files:
            prior = np.asarray(archive["abnormal_probability"], dtype=np.float32)
            return np.clip(_center_crop_or_pad(np.nan_to_num(prior), shape), 0.0, 1.0).astype(np.float32)
        key = "probabilities" if "probabilities" in archive.files else archive.files[0]
        probabilities = np.asarray(archive[key], dtype=np.float32)
        if probabilities.ndim == 3:
            prior = probabilities
        elif probabilities.ndim == 4:
            if probabilities.shape[0] <= 8:
                channels = probabilities
            elif probabilities.shape[-1] <= 8:
                channels = np.moveaxis(probabilities, -1, 0)
            else:
                raise ValueError(f"cannot infer probability channel axis for {path}: {probabilities.shape}")
            if channels.shape[0] == 1:
                prior = channels[0]
            elif probability_mode == "foreground_max":
                prior = channels.max(axis=0)
            elif probability_mode == "background_complement":
                prior = 1.0 - channels[int(probability_channel)]
            elif probability_mode == "channel":
                prior = channels[int(probability_channel)]
            else:
                raise ValueError(
                    "probability_mode must be foreground_max, background_complement, or channel; "
                    f"got {probability_mode!r}"
                )
        else:
            raise ValueError(f"expected a 3D/4D probability array, got {probabilities.shape} from {path}")
        prior = _center_crop_or_pad(np.nan_to_num(prior), shape)
    else:
        # Reference or hard predicted labels are converted to a binary abnormal mask.
        prior = _center_crop_or_pad((_load_nifti(path) > 0).astype(np.float32), shape)
    return np.clip(prior, 0.0, 1.0).astype(np.float32)


def load_region_targets(
    path: Path,
    shape: tuple[int, int, int],
    region_labels: Sequence[Sequence[int]],
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    labels = _center_crop_or_pad(_load_nifti(path), shape)
    regions = [np.isin(labels, np.asarray(values)).astype(np.float32) for values in region_labels]
    return np.stack(regions)


def _column_options(specification: object, prefix: str) -> list[str]:
    values = specification if isinstance(specification, list) else [specification]
    return [str(value).format(prefix=prefix) for value in values]


def _first_path(row: pd.Series, columns: Sequence[str], base_dir: Path) -> Path | None:
    for column in columns:
        if column in row and _present(row[column]):
            return _resolve_path(row[column], base_dir)
    return None


def _augment_mri_intensity(
    images: torch.Tensor,
    presence: torch.Tensor,
    *,
    scale_range: tuple[float, float],
    shift_range: tuple[float, float],
    noise_std: float,
) -> torch.Tensor:
    result = images.clone()
    for channel in range(result.shape[0]):
        if float(presence[channel]) == 0.0:
            continue
        foreground = result[channel].abs() > 1e-6
        if not torch.any(foreground):
            continue
        scale = torch.empty(()).uniform_(*scale_range)
        shift = torch.empty(()).uniform_(*shift_range)
        noise = torch.randn_like(result[channel]) * noise_std
        transformed = (result[channel] * scale) + shift + noise
        result[channel] = torch.where(foreground, transformed, result[channel])
    return result


class LongitudinalPairDataset(Dataset):
    """Loads paired MRI, lesion priors, modality indicators, interval, and label."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        manifest_dir: str | Path,
        modality_columns: Sequence[object],
        lesion_prior_columns: Sequence[str],
        label_column: str,
        target_shape: Sequence[int] = (128, 128, 128),
        allow_missing_modalities: bool = False,
        required_modalities: Sequence[bool] | None = None,
        probability_mode: str = "foreground_max",
        probability_channel: int = 0,
        augment: bool = False,
        intensity_scale_range: Sequence[float] = (0.9, 1.1),
        intensity_shift_range: Sequence[float] = (-0.1, 0.1),
        intensity_noise_std: float = 0.03,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.manifest_dir = Path(manifest_dir)
        self.modality_columns = list(modality_columns)
        self.lesion_prior_columns = list(lesion_prior_columns)
        self.label_column = str(label_column)
        self.target_shape = tuple(int(value) for value in target_shape)
        self.allow_missing_modalities = bool(allow_missing_modalities)
        self.required_modalities = (
            [not self.allow_missing_modalities] * len(self.modality_columns)
            if required_modalities is None
            else [bool(value) for value in required_modalities]
        )
        if len(self.required_modalities) != len(self.modality_columns):
            raise ValueError("required_modalities must match the modality column count")
        self.probability_mode = str(probability_mode)
        self.probability_channel = int(probability_channel)
        self.augment = bool(augment)
        self.intensity_scale_range = tuple(float(value) for value in intensity_scale_range)
        self.intensity_shift_range = tuple(float(value) for value in intensity_shift_range)
        self.intensity_noise_std = float(intensity_noise_std)
        if len(self.intensity_scale_range) != 2 or len(self.intensity_shift_range) != 2:
            raise ValueError("intensity augmentation ranges must contain two values")
        required = {"patient_id", "pair_id", "delta_weeks", self.label_column}
        missing = required.difference(self.frame.columns)
        if missing:
            raise KeyError(f"manifest is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def _load_visit(self, row: pd.Series, prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        volumes: list[np.ndarray] = []
        presence: list[float] = []
        for modality_index, specification in enumerate(self.modality_columns):
            path = _first_path(row, _column_options(specification, prefix), self.manifest_dir)
            if path is None:
                if self.required_modalities[modality_index] or not self.allow_missing_modalities:
                    raise FileNotFoundError(
                        f"missing required modality for pair={row['pair_id']} prefix={prefix}: {specification}"
                    )
                volumes.append(np.zeros(self.target_shape, dtype=np.float32))
                presence.append(0.0)
            else:
                volumes.append(load_mri(path, self.target_shape))
                presence.append(1.0)
        prior_path = _first_path(
            row,
            _column_options(self.lesion_prior_columns, prefix),
            self.manifest_dir,
        )
        if prior_path is None:
            raise FileNotFoundError(f"missing lesion prior for pair={row['pair_id']} prefix={prefix}")
        prior = load_lesion_prior(
            prior_path,
            self.target_shape,
            probability_mode=self.probability_mode,
            probability_channel=self.probability_channel,
        )
        return (
            torch.from_numpy(np.stack(volumes)).float(),
            torch.from_numpy(prior[None]).float(),
            torch.tensor(presence, dtype=torch.float32),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        x_t, lesion_t, modality_t = self._load_visit(row, "t")
        x_next, lesion_next, modality_next = self._load_visit(row, "next")
        if self.augment:
            for dimension in (1, 2, 3):
                if torch.rand(()) < 0.5:
                    x_t = torch.flip(x_t, dims=(dimension,))
                    x_next = torch.flip(x_next, dims=(dimension,))
                    lesion_t = torch.flip(lesion_t, dims=(dimension,))
                    lesion_next = torch.flip(lesion_next, dims=(dimension,))
            x_t = _augment_mri_intensity(
                x_t,
                modality_t,
                scale_range=self.intensity_scale_range,
                shift_range=self.intensity_shift_range,
                noise_std=self.intensity_noise_std,
            )
            x_next = _augment_mri_intensity(
                x_next,
                modality_next,
                scale_range=self.intensity_scale_range,
                shift_range=self.intensity_shift_range,
                noise_std=self.intensity_noise_std,
            )
        return {
            "x_t": x_t.contiguous(),
            "x_next": x_next.contiguous(),
            "lesion_t": lesion_t.contiguous(),
            "lesion_next": lesion_next.contiguous(),
            "modality_t": modality_t,
            "modality_next": modality_next,
            "delta_weeks": torch.tensor(float(row["delta_weeks"]), dtype=torch.float32),
            "label": torch.tensor(int(row[self.label_column]), dtype=torch.long),
            "pair_id": str(row["pair_id"]),
            "patient_id": str(row["patient_id"]),
        }


class SegmentationDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        manifest_dir: str | Path,
        modality_columns: Sequence[str],
        mask_column: str,
        region_labels: Sequence[Sequence[int]],
        target_shape: Sequence[int] = (128, 128, 128),
        augment: bool = False,
        intensity_scale_range: Sequence[float] = (0.9, 1.1),
        intensity_shift_range: Sequence[float] = (-0.1, 0.1),
        intensity_noise_std: float = 0.03,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.manifest_dir = Path(manifest_dir)
        self.modality_columns = list(modality_columns)
        self.mask_column = str(mask_column)
        self.region_labels = [list(map(int, region)) for region in region_labels]
        self.target_shape = tuple(int(value) for value in target_shape)
        self.augment = bool(augment)
        self.intensity_scale_range = tuple(float(value) for value in intensity_scale_range)
        self.intensity_shift_range = tuple(float(value) for value in intensity_shift_range)
        self.intensity_noise_std = float(intensity_noise_std)
        required = set(self.modality_columns) | {self.mask_column}
        missing = required.difference(self.frame.columns)
        if missing:
            raise KeyError(f"segmentation manifest is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        images = []
        for column in self.modality_columns:
            path = _resolve_path(row[column], self.manifest_dir)
            if path is None:
                raise FileNotFoundError(f"missing {column} for row {index}")
            images.append(load_mri(path, self.target_shape))
        mask_path = _resolve_path(row[self.mask_column], self.manifest_dir)
        if mask_path is None:
            raise FileNotFoundError(f"missing {self.mask_column} for row {index}")
        mask = load_region_targets(mask_path, self.target_shape, self.region_labels)
        image_tensor = torch.from_numpy(np.stack(images)).float()
        mask_tensor = torch.from_numpy(mask).float()
        if self.augment:
            for dimension in (1, 2, 3):
                if torch.rand(()) < 0.5:
                    image_tensor = torch.flip(image_tensor, dims=(dimension,))
                    mask_tensor = torch.flip(mask_tensor, dims=(dimension,))
            image_tensor = _augment_mri_intensity(
                image_tensor,
                torch.ones(image_tensor.shape[0]),
                scale_range=self.intensity_scale_range,
                shift_range=self.intensity_shift_range,
                noise_std=self.intensity_noise_std,
            )
        case_id = str(row.get("case_id", index))
        return {"image": image_tensor.contiguous(), "mask": mask_tensor.contiguous(), "case_id": case_id}
