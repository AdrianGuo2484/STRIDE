from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from stride.losses import longitudinal_loss, segmentation_loss
from stride.metrics import classification_metrics
from stride.model import AWHTSegmenter, STRIDE, load_longitudinal_pretraining
from scripts.train_longitudinal import BURDENKO_N_SPLITS, encode_class_labels, split_burdenko


def tiny_model(*, burdenko_classes: int = 3) -> STRIDE:
    return STRIDE(
        num_modalities=4,
        stem_channels=2,
        input_dropout=0.0,
        embed_dim=4,
        depths=[2],
        num_heads=[1],
        patch_size=2,
        attention_window_size=2,
        localization_scales=[1, 2, 4],
        localization_hidden_channels=2,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        use_checkpoint=False,
        time_embedding_dim=8,
        time_frequencies=2,
        max_weeks=52.0,
        context_dim=8,
        dynamics_hidden_dim=16,
        fusion_heads=1,
        fusion_depth=1,
        refinement_dim=16,
        num_lumiere_classes=7,
        num_burdenko_classes=burdenko_classes,
    )


def main() -> int:
    torch.manual_seed(7)
    batch_size = 1
    shape = (6, 6, 6)
    inputs = {
        "x_t": torch.randn(batch_size, 4, *shape),
        "x_next": torch.randn(batch_size, 4, *shape),
        "lesion_t": torch.zeros(batch_size, 1, *shape),
        "lesion_next": torch.rand(batch_size, 1, *shape),
        "modality_t": torch.ones(batch_size, 4),
        "modality_next": torch.tensor([[1.0, 1.0, 0.0, 1.0]]),
        "delta_weeks": torch.tensor([4.0]),
    }
    model = tiny_model()
    output = model(**inputs, task="lumiere")
    assert output["logits"].shape == (batch_size, 7)
    for gate_name in ("soft_gate_t", "soft_gate_next"):
        gate = output[gate_name]
        assert isinstance(gate, torch.Tensor)
        assert torch.all((gate >= 0.0) & (gate <= 1.0))
    for window_name in ("adaptive_window_t", "adaptive_window_next"):
        window = output[window_name]
        assert isinstance(window, torch.Tensor) and window.shape == (batch_size, 1, *shape)
        assert torch.all((window >= 0.0) & (window <= 1.0))
    expected_time = torch.log1p(inputs["delta_weeks"] / 52.0)
    assert torch.allclose(output["normalized_delta"].flatten(), expected_time, atol=1e-6)
    expected_transition = output["z_t"] + (
        output["normalized_delta"] * output["transition_gate"] * output["latent_change"]
    )
    assert torch.allclose(output["z_pred"], expected_transition, atol=1e-6)
    loss, _ = longitudinal_loss(
        output,
        torch.tensor([0]),
        class_weights=None,
        transition_weight=0.25,
        direction_weight=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)

    transferred = tiny_model()
    original_burdenko_head = transferred.burdenko_head.weight.detach().clone()
    missing, unexpected = load_longitudinal_pretraining(transferred, {"model_state": model.state_dict()})
    assert not unexpected
    assert set(missing) == {
        "lumiere_head.weight",
        "lumiere_head.bias",
        "burdenko_head.weight",
        "burdenko_head.bias",
    }
    assert torch.equal(transferred.burdenko_head.weight, original_burdenko_head)
    assert torch.equal(
        transferred.input_adapter.net[0].weight,
        model.input_adapter.net[0].weight,
    )
    transferred.set_transferred_modules_trainable(False)
    transferred_modules = transferred.transferred_modules()
    assert all(
        not parameter.requires_grad
        for module in transferred_modules
        for parameter in module.parameters()
    )
    assert all(parameter.requires_grad for parameter in transferred.burdenko_head.parameters())
    transferred.set_transferred_modules_trainable(True)
    assert all(
        parameter.requires_grad
        for module in transferred_modules
        for parameter in module.parameters()
    )
    downstream = transferred(**inputs, task="burdenko")
    assert downstream["logits"].shape == (batch_size, 3)

    labels = encode_class_labels(
        pd.DataFrame({"clinical_label": ["SD", "psp", "TP", 0, 1, 2]}),
        label_column="clinical_label",
        class_names=["SD", "PsP", "TP"],
    )
    assert labels["clinical_label"].tolist() == [0, 1, 2, 0, 1, 2]
    assert BURDENKO_N_SPLITS == 10
    fold_frame = pd.DataFrame(
        [
            {"patient_id": f"patient_{class_index}_{patient_index}", "clinical_label": class_index}
            for class_index in range(3)
            for patient_index in range(10)
        ]
    )
    train, validation, test, validation_fold = split_burdenko(
        fold_frame,
        label_column="clinical_label",
        patient_column="patient_id",
        n_splits=10,
        seed=42,
        test_fold=0,
    )
    assert (len(train), len(validation), len(test), validation_fold) == (24, 3, 3, 1)
    metrics = classification_metrics(
        [0, 1, 2],
        np.eye(3),
        num_classes=3,
        class_names=["SD", "PsP", "TP"],
    )
    assert metrics["macro_auc"] == 1.0

    segmenter = AWHTSegmenter(
        num_modalities=4,
        num_regions=3,
        stem_channels=2,
        input_dropout=0.0,
        embed_dim=4,
        depths=[2],
        num_heads=[1],
        patch_size=2,
        attention_window_size=2,
        localization_scales=[1, 2, 4],
        localization_hidden_channels=2,
        mlp_ratio=2.0,
        dropout=0.0,
        drop_path=0.0,
        use_checkpoint=False,
    )
    segmentation = segmenter(torch.randn(batch_size, 4, *shape))
    logits = segmentation["logits"]
    assert isinstance(logits, torch.Tensor) and logits.shape == (batch_size, 3, *shape)
    segmentation_objective, _ = segmentation_loss(logits, torch.rand_like(logits).round())
    segmentation_objective.backward()
    assert torch.isfinite(segmentation_objective)
    print(
        "PASS: PSN and longitudinal pretraining/transfer/fine-tuning paths completed "
        "forward and backward passes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
