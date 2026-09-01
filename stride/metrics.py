from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    targets: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, object]:
    y_true = np.asarray(targets, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != num_classes:
        raise ValueError(f"probabilities must have shape [N,{num_classes}], got {probs.shape}")
    if y_true.ndim != 1 or len(y_true) != len(probs):
        raise ValueError(f"targets must have shape [{len(probs)}], got {y_true.shape}")
    if np.any((y_true < 0) | (y_true >= num_classes)):
        raise ValueError(f"targets must be in [0,{num_classes - 1}], got {np.unique(y_true)}")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities contain non-finite values")
    predictions = probs.argmax(axis=1)
    names = [str(index) for index in range(num_classes)] if class_names is None else list(class_names)
    if len(names) != num_classes:
        raise ValueError(f"expected {num_classes} class names, got {names}")
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=np.arange(num_classes)).tolist(),
    }
    try:
        if num_classes == 2:
            result["auc"] = float(roc_auc_score(y_true, probs[:, 1])) if np.unique(y_true).size == 2 else math.nan
            result["precision"] = float(precision_score(y_true, predictions, zero_division=0))
            result["sensitivity"] = float(recall_score(y_true, predictions, zero_division=0))
            matrix = np.asarray(result["confusion_matrix"], dtype=np.int64)
            tn, fp = int(matrix[0, 0]), int(matrix[0, 1])
            result["specificity"] = float(tn / max(tn + fp, 1))
            result["f1"] = float(f1_score(y_true, predictions, zero_division=0))
        else:
            one_hot = np.eye(num_classes, dtype=np.float64)[y_true]
            valid = one_hot.sum(axis=0) > 0
            per_class_auc = []
            for class_index in range(num_classes):
                if not valid[class_index] or one_hot[:, class_index].sum() == len(y_true):
                    per_class_auc.append(math.nan)
                else:
                    per_class_auc.append(float(roc_auc_score(one_hot[:, class_index], probs[:, class_index])))
            result["per_class_auc"] = per_class_auc
            result["per_class_auc_by_class"] = dict(zip(names, per_class_auc))
            result["macro_auc"] = float(np.nanmean(per_class_auc)) if np.any(np.isfinite(per_class_auc)) else math.nan
    except ValueError:
        result["auc" if num_classes == 2 else "macro_auc"] = math.nan
    return result


def dice_score(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    return float((2.0 * np.logical_and(pred, truth).sum() + eps) / (pred.sum() + truth.sum() + eps))


def hd95(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> float:
    pred = np.asarray(prediction, dtype=bool)
    truth = np.asarray(target, dtype=bool)
    if not pred.any() and not truth.any():
        return 0.0
    if not pred.any() or not truth.any():
        return math.inf
    pred_surface = np.logical_xor(pred, binary_erosion(pred))
    truth_surface = np.logical_xor(truth, binary_erosion(truth))
    distance_to_truth = distance_transform_edt(~truth_surface, sampling=spacing)[pred_surface]
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)[truth_surface]
    distances = np.concatenate([distance_to_truth, distance_to_pred])
    return float(np.percentile(distances, 95))
