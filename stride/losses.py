from __future__ import annotations

import torch
import torch.nn.functional as F


def longitudinal_loss(
    output: dict[str, object],
    target: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
    transition_weight: float,
    direction_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output["logits"]
    z_pred = output["z_pred"]
    z_next = output["z_next"]
    assert isinstance(logits, torch.Tensor)
    assert isinstance(z_pred, torch.Tensor)
    assert isinstance(z_next, torch.Tensor)
    classification = F.cross_entropy(logits, target, weight=class_weights)
    transition = F.mse_loss(z_pred, z_next)
    direction = (1.0 - F.cosine_similarity(z_pred, z_next, dim=-1)).mean()
    total = classification + transition_weight * transition + direction_weight * direction
    return total, {
        "classification": classification,
        "transition": transition,
        "direction": direction,
    }


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    dimensions = tuple(range(2, probability.ndim))
    intersection = (probability * target).sum(dim=dimensions)
    denominator = probability.sum(dim=dimensions) + target.sum(dim=dimensions)
    return (1.0 - ((2.0 * intersection + eps) / (denominator + eps))).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    dice = dice_loss_from_logits(logits, target)
    return bce + dice, {"bce": bce, "dice_loss": dice}
