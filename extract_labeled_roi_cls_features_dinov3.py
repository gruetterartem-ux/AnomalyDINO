import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image


DEFAULT_EXPERIMENT_DIR = Path(
    r"C:\ai\AnomalyDINO\results_CUSTOM\dinov3_vitb16_688\8-shot_preprocess=force_no_mask_no_rotation_bestsearch8_fast20greedy_maxanomap_res688_evaltrain_20260413"
)
DEFAULT_ROI_METADATA_CSV = (
    DEFAULT_EXPERIMENT_DIR
    / "roi_crops_peak_hysteresis_h0.5_l0.2_merge3bridge0.1"
    / "seed=0"
    / "roi_metadata.csv"
)
DEFAULT_LABELS_FILE = Path(r"C:\ai\AnomalyDINO\labeling_tables\dinov3_res688_roi_labels.xlsx")
DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract DINOv3 CLS-token features for labeled ROI crops and write them in the same "
            "table format used by the GroupKFold logistic regression pipeline."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory used only for the default output location.",
    )
    parser.add_argument(
        "--roi-metadata-csv",
        type=Path,
        default=DEFAULT_ROI_METADATA_CSV,
        help="ROI metadata CSV with crop_path entries.",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS_FILE,
        help="Manual labels file (.xlsx or .csv) with bildname + roi_nummer + label.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <experiment-dir>/cls_roi_features_labeled.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id for DINOv3.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0 or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of labeled ROI crops processed per batch.",
    )
    parser.add_argument(
        "--valid-labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit class labels, e.g. --valid-labels 2D 3D.",
    )
    parser.add_argument(
        "--hf-token-env",
        type=str,
        default="HF_TOKEN",
        help="Environment variable that stores a Hugging Face token if needed.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit after label filtering, for debugging.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: List[Dict[str, object]], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    if not rows:
        output_file.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: Dict[str, object], output_file: Path) -> None:
    ensure_dir(output_file.parent)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_table_file(table_file: Path) -> pd.DataFrame:
    suffix = table_file.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(table_file)
    if suffix in {".csv", ".txt", ".tsv"}:
        return pd.read_csv(table_file, sep=None, engine="python")
    raise ValueError(f"Unsupported table format: {table_file}")


def normalize_bildname(value: object) -> str:
    return str(value).strip().replace("\\", "/").split("/")[-1]


def normalize_roi_nummer(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("roi"):
        return text
    if text.isdigit():
        return f"roi{text}"
    return text


def clean_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def import_transformers():
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency `transformers`. Install `transformers` in .venvAnomalyDINO."
        ) from exc
    return AutoImageProcessor, AutoModel


def resolve_device(device_name: str) -> str:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to cpu.")
        return "cpu"
    return device_name


def load_dinov3(model_id: str, device: str, token_env_name: str):
    AutoImageProcessor, AutoModel = import_transformers()
    token = os.getenv(token_env_name)
    try:
        processor = AutoImageProcessor.from_pretrained(model_id, token=token, local_files_only=True)
        model = AutoModel.from_pretrained(model_id, token=token, local_files_only=True)
        print("Loaded DINOv3 from local Hugging Face cache.")
    except Exception:
        processor = AutoImageProcessor.from_pretrained(model_id, token=token)
        model = AutoModel.from_pretrained(model_id, token=token)

    model = model.to(device)
    model.eval()
    return processor, model


def load_roi_table(roi_metadata_csv: Path) -> pd.DataFrame:
    if not roi_metadata_csv.exists():
        raise FileNotFoundError(f"ROI metadata not found: {roi_metadata_csv}")
    table = pd.read_csv(roi_metadata_csv)
    table["bildname"] = table["sample"].map(normalize_bildname)
    table["roi_nummer"] = "roi" + table["roi_index"].astype(int).astype(str)
    table["roi_uid"] = table["sample"].astype(str) + "__roi_" + table["roi_index"].astype(int).map(lambda idx: f"{idx:03d}")
    return table


def load_labels_table(labels_file: Path, valid_labels: List[str] | None) -> pd.DataFrame:
    labels = load_table_file(labels_file)
    if not {"bildname", "roi_nummer", "label"}.issubset(labels.columns):
        raise ValueError("Labels file must contain bildname, roi_nummer, and label columns.")

    labels = labels.copy()
    labels["bildname"] = labels["bildname"].map(normalize_bildname)
    labels["roi_nummer"] = labels["roi_nummer"].map(normalize_roi_nummer)
    labels["label"] = labels["label"].map(clean_label)
    labels = labels[labels["label"] != ""].copy()

    if valid_labels is not None:
        valid_label_set = {label.lower(): label for label in valid_labels}
        labels["label_lower"] = labels["label"].str.lower()
        labels = labels[labels["label_lower"].isin(valid_label_set.keys())].copy()
        labels["label"] = labels["label_lower"].map(valid_label_set)
        labels = labels.drop(columns=["label_lower"])

    if labels.duplicated(["bildname", "roi_nummer"]).any():
        raise ValueError("Labels file contains duplicate bildname + roi_nummer entries.")

    return labels


def prepare_labeled_roi_table(
    roi_table: pd.DataFrame,
    labels_table: pd.DataFrame,
    limit: int | None,
) -> pd.DataFrame:
    merged = roi_table.merge(labels_table, on=["bildname", "roi_nummer"], how="inner", suffixes=("", "_label"))
    if merged.empty:
        raise ValueError("No labeled ROIs matched the ROI metadata.")

    if "Genaues Label" in merged.columns:
        merged = merged.rename(columns={"Genaues Label": "detailed_label"})
    else:
        merged["detailed_label"] = ""

    merged["detailed_label"] = merged["detailed_label"].fillna("").astype(str).str.strip()
    merged["crop_path"] = merged["crop_path"].astype(str)
    missing_crops = [path for path in merged["crop_path"].tolist() if not Path(path).exists()]
    if missing_crops:
        raise FileNotFoundError(f"Missing ROI crop files, first examples: {missing_crops[:5]}")

    merged = merged.sort_values(["bildname", "roi_index"]).reset_index(drop=True)
    if limit is not None:
        merged = merged.iloc[:limit].copy()
    return merged


def load_image_batch(image_paths: List[Path]) -> tuple[List[Image.Image], List[tuple[int, int]]]:
    images: List[Image.Image] = []
    sizes: List[tuple[int, int]] = []
    for image_path in image_paths:
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            sizes.append(image.size)
            images.append(image.copy())
    return images, sizes


def extract_cls_embeddings(image_paths: List[Path], processor, model, device: str) -> tuple[np.ndarray, List[tuple[int, int]]]:
    images, sizes = load_image_batch(image_paths)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings.float().cpu().numpy(), sizes


def default_output_dir(experiment_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (experiment_dir / "cls_roi_features_labeled").resolve()


def main():
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    roi_metadata_csv = args.roi_metadata_csv.resolve()
    labels_file = args.labels_file.resolve()
    output_dir = default_output_dir(experiment_dir, args.output_dir)
    ensure_dir(output_dir)

    valid_labels = None if args.valid_labels is None else [str(label) for label in args.valid_labels]
    roi_table = load_roi_table(roi_metadata_csv)
    labels_table = load_labels_table(labels_file, valid_labels)
    labeled_rois = prepare_labeled_roi_table(roi_table, labels_table, args.limit)

    device = resolve_device(args.device)
    processor, model = load_dinov3(args.model_id, device, args.hf_token_env)

    all_features: List[np.ndarray] = []
    metadata_rows: List[Dict[str, object]] = []

    crop_paths = [Path(path) for path in labeled_rois["crop_path"].tolist()]
    for start in range(0, len(crop_paths), args.batch_size):
        batch_paths = crop_paths[start : start + args.batch_size]
        batch_embeddings, batch_sizes = extract_cls_embeddings(batch_paths, processor, model, device)
        all_features.append(batch_embeddings.astype(np.float32))

        batch_rows = labeled_rois.iloc[start : start + len(batch_paths)]
        for batch_index, (_, row) in enumerate(batch_rows.iterrows()):
            width, height = batch_sizes[batch_index]
            metadata_rows.append(
                {
                    "feature_index": start + batch_index,
                    "feature_type": "dinov3_cls_token",
                    "roi_uid": row["roi_uid"],
                    "label": row["label"],
                    "notes": row.get("detailed_label", ""),
                    "detailed_label": row.get("detailed_label", ""),
                    "bildname": row["bildname"],
                    "roi_nummer": row["roi_nummer"],
                    "sample": row["sample"],
                    "group_id": row["sample"],
                    "roi_index": int(row["roi_index"]),
                    "object": row["object"],
                    "split": row["split"],
                    "image_path": row["crop_path"],
                    "crop_path": row["crop_path"],
                    "original_sample": row["sample"],
                    "region_max_score": float(row.get("region_max_score", 0.0)),
                    "region_mass": float(row.get("region_mass", 0.0)),
                    "primary_peak_score": float(row.get("primary_peak_score", 0.0)),
                    "width": int(width),
                    "height": int(height),
                    "model_id": args.model_id,
                    "embedding_dim": int(batch_embeddings.shape[1]),
                }
            )

        print(f"Processed {min(start + len(batch_paths), len(crop_paths))}/{len(crop_paths)} labeled ROI crops")

    feature_array = np.concatenate(all_features, axis=0).astype(np.float32)
    features_file = output_dir / "roi_features_mean.npy"
    metadata_file = output_dir / "roi_feature_table.csv"
    summary_file = output_dir / "summary.json"
    np.save(features_file, feature_array)
    write_csv(metadata_rows, metadata_file)
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "roi_metadata_csv": str(roi_metadata_csv),
            "labels_file": str(labels_file),
            "model_id": args.model_id,
            "num_labeled_rois": int(len(metadata_rows)),
            "num_groups": int(pd.Series(labeled_rois["sample"]).nunique()),
            "class_counts": labeled_rois["label"].value_counts().sort_index().to_dict(),
            "features_file": str(features_file),
            "metadata_file": str(metadata_file),
            "feature_shape": list(feature_array.shape),
        },
        summary_file,
    )

    print(f"Saved features: {features_file}")
    print(f"Saved metadata: {metadata_file}")
    print(f"Saved summary: {summary_file}")
    print(f"Feature shape: {feature_array.shape}")


if __name__ == "__main__":
    main()
