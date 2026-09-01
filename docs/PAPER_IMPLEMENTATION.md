# Paper-to-code mapping

| Manuscript definition | Implementation |
|---|---|
| Eq. (1), paired three-class prediction | `STRIDE.forward` and `burdenko_head` |
| Eqs. (2)-(3), projected residual SoftGate | `MaskConditionedSoftGate` |
| Eq. (4), two-layer input adaptation | `InputAdapter` |
| Eqs. (5)-(7), multi-scale continuous spatial window | `MultiScaleAdaptiveSpatialWindow` |
| Eqs. (8)-(10), hierarchical regular/shifted window encoding | `AWHTEncoder` and `WindowTransformerBlock3D` |
| Eq. (11), log-scaled interval embedding | `TimeEmbedding` |
| Eqs. (12)-(14), adaptive Hadamard pair context | `interaction_gate` and `pair_context` |
| Eqs. (15)-(16), gated latent transition | `GatedLatentTransition` |
| Eqs. (17)-(19), observed-transition fusion | `LongitudinalFusion` |
| Eqs. (20)-(23), weighted classification and transition losses | `longitudinal_loss` |
| BCE plus Dice lesion-prior objective | `segmentation_loss` |
| Independent frozen lesion-prior network | `AWHTSegmenter` and `predict_lesion_priors.py` |
| Patient-level BraTS2024 five-fold validation | `train_segmentation.py` |
| Predefined patient-independent LUMIERE split | `split_lumiere` |
| Patient-level stratified Burdenko 10-fold CV | `split_burdenko` |
| Head-independent LUMIERE transfer | `load_longitudinal_pretraining` |

## Classification contract

The Burdenko output has exactly three logits in manuscript order:

```text
0=SD, 1=PsP, 2=TP
```

The training entry point accepts these names or IDs, validates every label, and
uses normalized inverse class frequencies computed only from the training
partition. Classification metrics use macro-averaged one-vs-rest ROC-AUC,
accuracy, and macro F1.

Burdenko evaluation is fixed to 10 folds. Fold `k` is test, fold
`(k+1) mod 10` is validation, and the remaining eight folds are training folds.
All pairs from a patient remain together. The first 10 fine-tuning epochs and
validation AUC values exactly equal to `1.0` are excluded from checkpoint
selection.

## Architecture contract

SoftGate computes a learned lesion-prior projection, a sigmoid gate from its
element-wise interaction with MRI features, and residual modulation. AWHT then
uses localization scales `{1, 2, 4}` to create a continuous examination-specific
spatial weighting field. Local Transformer attention uses a fixed `8 x 8 x 8`
token window and alternates regular and shifted blocks.

The interval scale is implemented exactly as:

```text
normalized_delta = log(1 + delta_weeks / 52)
```

The pair context includes the earlier and later representations, their
observed difference, and a learned-gate-weighted Hadamard interaction. Seven
projected sources are passed through the longitudinal fusion operator before
classification.

## Optimization contract

```text
L_pre  = L_cls + 0.25 L_tr + 0.10 L_dir
L_down = L_cls + 0.15 L_tr + 0.05 L_dir
```

Training-only augmentation includes axis flips, intensity scaling in
`[0.9, 1.1]`, intensity shifting in `[-0.1, 0.1]`, and Gaussian noise with
standard deviation `0.03`. All stages use AdamW, weight decay `1e-4`, and
gradient clipping at `1.0`.

The lesion-prior network remains parameter-independent: its state dictionary is
never transferred into STRIDE. LUMIERE masks and Burdenko probability maps are
structural inputs to SoftGate, not longitudinal segmentation targets.
