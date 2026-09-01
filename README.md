# STRIDE

Official minimal implementation of **STRIDE: Spatial-Temporal Representation
for Interval-conditioned Disease Evolution in Longitudinal Glioblastoma MRI**.
STRIDE classifies a pair of post-treatment MRI examinations as **stable disease
(SD), pseudoprogression (PsP), or true progression (TP)** while explicitly
modeling lesion location and the time between scans.

> This anonymous release contains source code and experiment configurations
> only. It does not include patient data, private manifests, trained weights,
> generated lesion priors, or experiment logs.

## Method at a glance

```text
BraTS2024 -> train lesion-prior network -> frozen Burdenko lesion priors
LUMIERE -> longitudinal pretraining -> pretrained STRIDE parameters
paired Burdenko MRI + interval + priors -> SD / PsP / TP (10-fold evaluation)
```

For each visit, a learnable residual SoftGate combines MRI features with a
segmentation-derived lesion prior. AWHT then estimates a scan-specific
continuous spatial window from three localization scales `{1, 2, 4}` and
encodes the weighted volume with regular and shifted `8 x 8 x 8` local
attention. For a visit pair, STRIDE combines an adaptive cross-visit
interaction, a log-scaled interval embedding, a gated latent transition, and
observed-transition fusion before three-class prediction.

The BraTS2024 segmentation network is trained independently and remains
parameter-independent from STRIDE. It is used only to create frozen lesion
priors for Burdenko.

## Reported results

The manuscript reports the following patient-level 10-fold Burdenko result:

| Macro ROC-AUC | Accuracy | Macro F1 |
|---:|---:|---:|
| 0.816 | 0.845 | 0.796 |

These are manuscript results, not precomputed outputs bundled with this
anonymous source release.

## Repository

```text
configs/   Paper experiment configurations
docs/      Manifest schema and paper-to-code mapping
scripts/   Training, lesion-prior inference, and verification entry points
stride/    Data loading, models, losses, metrics, and runtime utilities
```

## Setup

Python 3.10 or newer is required. Full-resolution training requires a CUDA GPU;
the verification script can run on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/smoke_test.py
```

The smoke test runs forward and backward passes through the lesion segmenter,
LUMIERE pretraining, checkpoint transfer, and three-class Burdenko paths. It
also checks the time-transition equation, three-class metrics, label encoding,
module freezing, and task-head isolation.

## Data contract

MRI volumes must be de-identified, rigidly registered to MNI152, skull
stripped, and resampled to `1.25 mm` isotropic spacing. The loader applies
foreground z-score normalization and center-crops or center-pads volumes to
`128 x 128 x 128`. Both longitudinal datasets use the same semantic channel
order:

```text
T1CE/cT1, T1, T2, FLAIR
```

Burdenko requires T1CE and FLAIR. T1 and T2 may be missing; zero-filled channels
and modality-presence indicators are supplied to the model. The downstream
manifest label column is `clinical_label` and accepts either names or IDs:

```text
SD=0, PsP=1, TP=2
```

See [docs/MANIFESTS.md](docs/MANIFESTS.md) for the complete private CSV schemas.
Paths may be absolute or relative to the manifest. The `data/` directory and
medical-image formats are ignored by Git.

## Reproduce the experiments

Run all commands from the repository root.

### 1. Train the BraTS2024 lesion-prior network

```bash
for fold in 0 1 2 3 4; do
  python scripts/train_segmentation.py \
    --config configs/brats_psn.yaml \
    --fold "$fold"
done
```

### 2. Generate frozen Burdenko lesion priors

Select a validation checkpoint and run it on every eligible Burdenko visit:

```bash
python scripts/predict_lesion_priors.py \
  --checkpoint runs/brats2024_psn/fold_0/best.pt \
  --manifest data/burdenko_visits.csv \
  --output-dir outputs/burdenko_lesion_priors
```

Merge `lesion_prior_manifest.csv` into `data/burdenko_pairs.csv` as
`t_path_prob_pred_brats2024` and `next_path_prob_pred_brats2024`.

### 3. Pretrain STRIDE on LUMIERE

```bash
python scripts/train_longitudinal.py \
  --config configs/lumiere_pretrain.yaml
```

### 4. Fine-tune and evaluate on Burdenko

```bash
for fold in 0 1 2 3 4 5 6 7 8 9; do
  python scripts/train_longitudinal.py \
    --config configs/burdenko_finetune.yaml \
    --fold "$fold"
done
```

Fold `k` is held out for testing, fold `(k+1) mod 10` is used for validation,
and the remaining eight folds are used for training. All pairs from one patient
stay in the same partition. Burdenko checkpoints are selected by validation macro
one-vs-rest ROC-AUC; the first 10 epochs and exactly perfect validation AUCs are
excluded from checkpoint selection, as specified in the manuscript.

Use `--resume` to continue from `last.pt`. The optional `--epochs`,
`--output-dir`, `--manifest`, and `--pretrained-checkpoint` arguments support
isolated integration runs without editing the paper configurations.

## Paper settings

| Stage | Evaluation split | Epochs | Batch | Learning rate | Selection metric |
|---|---:|---:|---:|---:|---|
| BraTS2024 segmentation | 5-fold | 200 | 2 | `1e-3` | Mean Dice |
| LUMIERE pretraining | Predefined | 300 | 8 | `1e-4` | Macro ROC-AUC |
| Burdenko fine-tuning | 10-fold | 200 | 8 | `8e-5` | Macro ROC-AUC |

All stages use AdamW, weight decay `1e-4`, gradient clipping at `1.0`, and
training-only spatial and intensity augmentation. The longitudinal objectives
are:

```text
L_pre  = L_cls + 0.25 L_transition + 0.10 L_direction
L_down = L_cls + 0.15 L_transition + 0.05 L_direction
```

Class weights are computed from the training partition only. During Burdenko
transfer, pretrained modules are frozen for five epochs and then optimized
jointly with the three-class head.

## Outputs

Each run writes `resolved_config.json`, `history.json`, `best.pt`, and `last.pt`
under its configured output directory. A completed Burdenko fold also writes:

```text
fold_<k>/test_metrics.json
fold_<k>/test_predictions.csv
```

Predictions contain the target and predicted class names plus
`probability_SD`, `probability_PsP`, and `probability_TP`. Metrics include
accuracy, macro F1, macro one-vs-rest ROC-AUC, per-class AUC, and the confusion
matrix.

Implementation details and equation-level traceability are documented in
[docs/PAPER_IMPLEMENTATION.md](docs/PAPER_IMPLEMENTATION.md).
