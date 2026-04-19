from __future__ import annotations

import argparse
import csv
import json
from math import ceil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from component_memory_bank.data_io import load_patch_features, load_patch_scores, load_run_samples
from extract_labeled_roi_cls_features_dinov3 import (
    clean_label,
    default_output_dir as default_cls_output_dir,
    ensure_dir,
    extract_cls_embeddings,
    load_dinov3,
    load_labels_table,
    load_roi_table,
    prepare_labeled_roi_table,
    resolve_device,
    write_csv,
    write_json,
)


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
            "Extract DINOv3 ROI CLS features plus the mean feature vector of the top 1% most anomalous "
            "patches inside the ROI bounding box, then write them in the standard ROI feature format."
        )
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--roi-metadata-csv", type=Path, default=DEFAULT_ROI_METADATA_CSV)
    parser.add_argument("--labels-file", type=Path, default=DEFAULT_LABELS_FILE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--valid-labels", type=str, nargs="*", default=["2D", "3D"])
    parser.add_argument("--hf-token-env", type=str, default="HF_TOKEN")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.01,
        help="Fraction of bbox patches used for the anomaly-patch feature mean. Minimum one patch.",
    )
    return parser.parse_args()


def default_output_dir(experiment_dir: Path, explicit_output_dir: Path | None) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()
    return (experiment_dir / "cls_plus_top1p_bbox_patch_features_labeled").resolve()


def _build_sample_map(experiment_dir: Path, seed: int = 0) -> dict[str, object]:
    return {sample.sample: sample for sample in load_run_samples(experiment_dir, seed=seed)}


def _load_sample_patch_data(sample_map: dict[str, object], sample_name: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    sample = sample_map[sample_name]
    features, grid_size = load_patch_features(sample)
    scores = load_patch_scores(sample)
    if tuple(scores.shape) != tuple(grid_size):
        raise ValueError(
            f"Grid mismatch for sample {sample_name}: feature grid {grid_size}, score grid {scores.shape}"
        )
    return features, scores, grid_size


def _bbox_top_feature_mean(
    roi_row: pd.Series,
    sample_map: dict[str, object],
    cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]],
    top_percent: float,
) -> tuple[np.ndarray, dict[str, object]]:
    sample_name = str(roi_row["sample"])
    if sample_name not in cache:
        cache[sample_name] = _load_sample_patch_data(sample_map, sample_name)

    features_flat, score_grid, grid_size = cache[sample_name]
    grid_rows, grid_cols = grid_size
    feature_dim = int(features_flat.shape[1])
    features_grid = features_flat.reshape(grid_rows, grid_cols, feature_dim)

    row_min = int(roi_row["region_row_min"])
    row_max = int(roi_row["region_row_max"])
    col_min = int(roi_row["region_col_min"])
    col_max = int(roi_row["region_col_max"])

    if not (0 <= row_min <= row_max < grid_rows and 0 <= col_min <= col_max < grid_cols):
        raise ValueError(
            f"Invalid ROI bbox in patch space for {sample_name} roi {roi_row['roi_index']}: "
            f"rows {row_min}-{row_max}, cols {col_min}-{col_max}, grid {grid_size}"
        )

    bbox_scores = score_grid[row_min : row_max + 1, col_min : col_max + 1]
    bbox_features = features_grid[row_min : row_max + 1, col_min : col_max + 1, :]
    flat_scores = bbox_scores.reshape(-1)
    flat_features = bbox_features.reshape(-1, feature_dim)

    num_bbox_patches = int(flat_scores.shape[0])
    top_k = max(1, int(ceil(num_bbox_patches * float(top_percent))))
    top_indices = np.argsort(-flat_scores)[:top_k]
    selected_features = flat_features[top_indices]
    selected_scores = flat_scores[top_indices]
    pooled_feature = selected_features.mean(axis=0).astype(np.float32)

    bbox_width = int(col_max - col_min + 1)
    selected_rows = []
    selected_cols = []
    selected_patch_indices = []
    for flat_index in top_indices.tolist():
        local_row = flat_index // bbox_width
        local_col = flat_index % bbox_width
        global_row = row_min + local_row
        global_col = col_min + local_col
        selected_rows.append(global_row)
        selected_cols.append(global_col)
        selected_patch_indices.append(global_row * grid_cols + global_col)

    meta = {
        "bbox_patch_count": num_bbox_patches,
        "top1p_patch_count": top_k,
        "top1p_patch_rows": ";".join(str(v) for v in selected_rows),
        "top1p_patch_cols": ";".join(str(v) for v in selected_cols),
        "top1p_patch_indices": ";".join(str(v) for v in selected_patch_indices),
        "top1p_patch_scores": ";".join(f"{float(v):.8f}" for v in selected_scores.tolist()),
        "top1p_patch_score_max": float(selected_scores.max()),
        "top1p_patch_score_mean": float(selected_scores.mean()),
    }
    return pooled_feature, meta


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
    sample_map = _build_sample_map(experiment_dir, seed=0)
    sample_patch_cache: dict[str, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}

    all_cls_features: list[np.ndarray] = []
    all_top_patch_features: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    crop_paths = [Path(path) for path in labeled_rois["crop_path"].tolist()]
    for start in range(0, len(crop_paths), args.batch_size):
        batch_paths = crop_paths[start : start + args.batch_size]
        batch_cls_embeddings, batch_sizes = extract_cls_embeddings(batch_paths, processor, model, device)
        all_cls_features.append(batch_cls_embeddings.astype(np.float32))

        batch_rows = labeled_rois.iloc[start : start + len(batch_paths)]
        for batch_index, (_, row) in enumerate(batch_rows.iterrows()):
            width, height = batch_sizes[batch_index]
            top_patch_feature, top_patch_meta = _bbox_top_feature_mean(
                roi_row=row,
                sample_map=sample_map,
                cache=sample_patch_cache,
                top_percent=float(args.top_percent),
            )
            all_top_patch_features.append(top_patch_feature)

            metadata_rows.append(
                {
                    "feature_index": start + batch_index,
                    "feature_type": "dinov3_cls_plus_top1p_bbox_patch_mean",
                    "roi_uid": row["roi_uid"],
                    "label": clean_label(row["label"]),
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
                    "embedding_dim_cls": int(batch_cls_embeddings.shape[1]),
                    "embedding_dim_top1p_patch_mean": int(top_patch_feature.shape[0]),
                    "embedding_dim_total": int(batch_cls_embeddings.shape[1] + top_patch_feature.shape[0]),
                    "top_percent": float(args.top_percent),
                    "region_row_min": int(row["region_row_min"]),
                    "region_row_max": int(row["region_row_max"]),
                    "region_col_min": int(row["region_col_min"]),
                    "region_col_max": int(row["region_col_max"]),
                    **top_patch_meta,
                }
            )

        print(f"Processed {min(start + len(batch_paths), len(crop_paths))}/{len(crop_paths)} labeled ROI crops")

    cls_feature_array = np.concatenate(all_cls_features, axis=0).astype(np.float32)
    top_patch_feature_array = np.stack(all_top_patch_features, axis=0).astype(np.float32)
    combined_feature_array = np.concatenate([cls_feature_array, top_patch_feature_array], axis=1).astype(np.float32)

    features_file = output_dir / "roi_features_mean.npy"
    metadata_file = output_dir / "roi_feature_table.csv"
    summary_file = output_dir / "summary.json"
    np.save(features_file, combined_feature_array)
    write_csv(metadata_rows, metadata_file)
    write_json(
        {
            "experiment_dir": str(experiment_dir),
            "roi_metadata_csv": str(roi_metadata_csv),
            "labels_file": str(labels_file),
            "model_id": args.model_id,
            "top_percent": float(args.top_percent),
            "num_labeled_rois": int(len(metadata_rows)),
            "num_groups": int(pd.Series(labeled_rois["sample"]).nunique()),
            "class_counts": labeled_rois["label"].astype(str).value_counts().sort_index().to_dict(),
            "features_file": str(features_file),
            "metadata_file": str(metadata_file),
            "feature_shape": list(combined_feature_array.shape),
            "embedding_dim_cls": int(cls_feature_array.shape[1]),
            "embedding_dim_top1p_patch_mean": int(top_patch_feature_array.shape[1]),
            "embedding_dim_total": int(combined_feature_array.shape[1]),
        },
        summary_file,
    )

    print(f"Saved features: {features_file}")
    print(f"Saved metadata: {metadata_file}")
    print(f"Saved summary: {summary_file}")
    print(f"Feature shape: {combined_feature_array.shape}")


if __name__ == "__main__":
    raise SystemExit(main())
