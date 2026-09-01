from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from stride.data import load_mri
from stride.model import AWHTSegmenter
from stride.runtime import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen PSN abnormal-probability priors.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = payload["config"]
    model = AWHTSegmenter(**config["model"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    frame = pd.read_csv(manifest_path)
    modality_columns = list(config["data"]["modality_columns"])
    modality_names = [str(value).lower() for value in config["data"]["modality_names"]]
    if len(modality_columns) != len(modality_names):
        raise ValueError("checkpoint modality names and columns have different lengths")
    if "case_id" in frame and frame["case_id"].astype(str).duplicated().any():
        raise ValueError("prior-inference manifest contains duplicate case_id values")
    target_shape = tuple(int(value) for value in config["data"].get("target_shape", [128, 128, 128]))
    whole_tumor_channel = int(config["data"].get("whole_tumor_channel", 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for index, row in frame.iterrows():
            case_id = str(row.get("case_id", f"case_{index:05d}"))
            if Path(case_id).name != case_id:
                raise ValueError(f"case_id must not contain path separators: {case_id!r}")
            channels = []
            for column, modality_name in zip(modality_columns, modality_names):
                value = row.get(column)
                if value is None or pd.isna(value) or str(value).strip() == "":
                    if modality_name in {"t1ce", "ct1", "flair"}:
                        raise FileNotFoundError(f"case={case_id} is missing required modality {modality_name}")
                    channels.append(np.zeros(target_shape, dtype=np.float32))
                    continue
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = manifest_path.parent / path
                channels.append(load_mri(path, target_shape))
            image = torch.from_numpy(np.stack(channels))[None].to(device=device, dtype=torch.float32)
            output = model(image)
            logits = output["logits"]
            assert isinstance(logits, torch.Tensor)
            region_probabilities = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float32)
            abnormal_probability = region_probabilities[whole_tumor_channel]
            destination = output_dir / f"{case_id}.npz"
            np.savez_compressed(
                destination,
                abnormal_probability=abnormal_probability,
                region_probabilities=region_probabilities,
            )
            rows.append({"case_id": case_id, "lesion_prior_path": str(destination)})
    pd.DataFrame(rows).to_csv(output_dir / "lesion_prior_manifest.csv", index=False)
    print(f"generated {len(rows)} lesion priors in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
